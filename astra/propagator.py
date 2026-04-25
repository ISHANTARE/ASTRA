# astra/propagator.py
"""ASTRA Core Numerical Propagator — Segmented Cowell's Method.

Implements a mission-operations–grade numerical orbit propagator using
Cowell's direct integration with a Dormand-Prince **DOP853** (8th-order,
error-estimate 5th/3rd order) adaptive-step integrator via SciPy's
``solve_ivp(method='DOP853')``.

**Features**

- **6-DOF coast arcs:** Two-body + J2/J3/J4 + drag + 3rd-body gravity.
- **7-DOF powered arcs:** Attitude-steered thrust with Tsiolkovsky-coupled mass depletion.
- **Segmented orchestrator:** Slices propagation at engine ignition/cutoff boundaries so
  the integrator never steps across a force-model discontinuity.
- **High-fidelity data:** JPL DE421 Sun/Moon via Skyfield; empirical atmospheric density
  from F10.7 and Ap (replacing a static exponential model).

**Force model includes**

- Two-body Keplerian gravity
- J2, J3, J4 zonal harmonic perturbations (WGS84)
- Empirical atmospheric drag (NRLMSISE-00 with space weather)
- Solar third-body point-mass perturbation (JPL DE421)
- Lunar third-body point-mass perturbation (JPL DE421)
- Finite continuous thrust (7-DOF powered arcs)

**Numba / IEEE-754**

JIT kernels use ``@njit(fastmath=True)`` (MTH-16), which allows reordering and
fused operations that can differ slightly from the pure-Python
``_acceleration`` path. For validation, compare integrated trajectories
or segment-level energy, not bitwise-identical acceleration samples.

**SRP / PHY-18**

Cannonball SRP scales flux from 1 AU; uses a **conical Earth umbra/penumbra**
geometry (planar intersection model) to continuously scale solar pressure through
twilight regions — see ``DragConfig.srp_cylindrical_shadow`` (named for legacy compatibility).

**References**

- Vallado, D. A. (2013). *Fundamentals of Astrodynamics and Applications.*
- Montenbruck & Gill (2000). *Satellite Orbits.*
- Park et al. (2021). JPL Planetary Ephemerides DE440/DE441.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Any

import numpy as np
import numpy.polynomial.chebyshev as cheb
from astra._numba_compat import njit
from scipy.integrate import solve_ivp

from astra.constants import (
    EARTH_EQUATORIAL_RADIUS_KM,
    EARTH_MU_KM3_S2,
    EARTH_OMEGA_RAD_S,
    J2,
    J3,
    J4,
    SECONDS_PER_DAY,
    SUN_MU_KM3_S2,
    MOON_MU_KM3_S2,
    G0_STD,
    AU_KM as _AU_KM_CONST,
    SRP_P0_N_M2 as _SRP_P0,
)
from astra.log import get_logger
from astra.frames import _build_vnb_matrix_njit, _build_rtn_matrix_njit
from astra.models import FiniteBurn

logger = get_logger(__name__)


@njit(fastmath=True, cache=True)
def _srp_illumination_factor_planar_njit(
    r_km: np.ndarray,
    r_sun_km: np.ndarray,
    earth_radius_km: float = 6378.137,
    sun_radius_km: float = 695700.0,
) -> float:
    """LEGACY: Conical Earth umbra/penumbra factor using planar circle-circle geometry.

    Preserved for regression comparison against ``_srp_illumination_factor_dual_cone_njit``.
    Do NOT use in the force model — use the dual-cone function instead.

    **Limitation (FM-1B):** Applies planar geometry to what is fundamentally a
    spherical-cap intersection problem. Error is O(α²) — negligible in LEO but
    up to 8% in HEO/GEO and during grazing transits. The first derivative dν/dγ
    is discontinuous at the umbra/penumbra boundary, injecting force impulse
    artifacts into the numerical integrator.

    References:
        Montenbruck & Gill, *Satellite Orbits*, §3.4.2 (Springer, 2000) — see
        the note on the planar-approximation error.
    """
    d_sun_sat = r_sun_km - r_km
    d_sun_sat_mag = np.linalg.norm(d_sun_sat)
    r_mag = np.linalg.norm(r_km)

    if r_mag < 1.0 or d_sun_sat_mag < 1.0:
        return 1.0

    alpha = math.asin(min(1.0, earth_radius_km / r_mag))
    beta  = math.asin(min(1.0, sun_radius_km  / d_sun_sat_mag))

    cos_gamma = np.dot(-r_km, d_sun_sat) / (r_mag * d_sun_sat_mag)
    if cos_gamma >  1.0: cos_gamma =  1.0
    if cos_gamma < -1.0: cos_gamma = -1.0
    gamma = math.acos(cos_gamma)

    if gamma >= alpha + beta:
        return 1.0
    if gamma <= alpha - beta:
        return 0.0

    x = (gamma * gamma + alpha * alpha - beta * beta) / (2.0 * gamma)
    y = math.sqrt(max(0.0, alpha * alpha - x * x))
    overlap_area = (
        alpha * alpha * math.acos(x / alpha)
        + beta * beta * math.acos((gamma - x) / beta)
        - gamma * y
    )
    nu = 1.0 - overlap_area / (math.pi * beta * beta)
    if nu < 0.0: nu = 0.0
    if nu > 1.0: nu = 1.0
    return nu


@njit(fastmath=True, cache=True)
def _srp_illumination_factor_dual_cone_njit(
    r_km: np.ndarray,
    r_sun_km: np.ndarray,
    earth_radius_km: float = 6378.137,
    sun_radius_km: float = 695700.0,
) -> float:
    """Dual-cone SRP illumination factor ν ∈ [0, 1] (FM-1B fix).

    Computes the fraction of the solar disk visible from the satellite,
    accounting for Earth's occultation. Supersedes the old planar formula by
    correctly handling three additional cases:

      1. **Annular eclipse** (β > α): when the Sun's apparent disk is larger
         than Earth's (valid at very high altitudes), the old code returned
         ν = 0 incorrectly. Now returns ν = 1 − (α/β)² (ratio of disk areas).
      2. **γ ≈ 0 degenerate guard**: prevents division-by-zero in the
         penumbra formula when Earth and Sun centres are coincident.
      3. **β > α umbra gate**: the old ``gamma <= alpha - beta`` condition
         is negative when β > α, so it never triggered for annular geometries.
         Now both eclipse cases are gated explicitly.

    The penumbra formula is the standard Montenbruck & Gill §3.4.2 (eq. 3.84–
    3.87) planar circle-circle intersection — the industry reference for
    all Earth-orbit SRP shadow models, including Vallado (2013) Algorithm 34.
    For Earth-orbiting satellites, β ≈ 0.266° (tiny), so the planar
    approximation error for the Sun disk is O(β²/12) ≈ 1.7×10⁻⁷ — below
    IEEE-754 double precision significance and irrelevant operationally.

    Geometry (as seen from satellite):
        α = arcsin(R_Earth / |r_sat|)   apparent Earth semi-radius
        β = arcsin(R_Sun   / |r_sat−r_sun|)  apparent Sun semi-radius
        γ = angular separation between Earth and Sun disk centres

        Case 1 — Full sunlight:   γ ≥ α + β           → ν = 1
        Case 2 — Full umbra:      γ ≤ α − β, α ≥ β    → ν = 0
        Case 2b — Annular eclipse: γ ≤ β − α, β > α   → ν = 1 − (α/β)²
        Case 3 — Penumbra:
            x  = (γ² + β² − α²) / (2γ)
            y  = √(β² − x²)
            A  = α²·arccos((γ−x)/α) + β²·arccos(x/β) − γ·y
            ν  = 1 − A / (π·β²)

    Args:
        r_km: Geocentric satellite position vector (km), shape (3,).
        r_sun_km: Geocentric Sun position vector (km), shape (3,).
        earth_radius_km: Earth equatorial radius (km). Default WGS-84 6378.137.
        sun_radius_km: Solar mean radius (km). Default 695700.0.

    Returns:
        ν ∈ [0.0, 1.0]: 0 = full eclipse, 1 = full sunlight, (0,1) = penumbra.
    """
    d_sun_sat     = r_sun_km - r_km
    d_sun_sat_mag = np.linalg.norm(d_sun_sat)
    r_mag         = np.linalg.norm(r_km)

    if r_mag < 1.0 or d_sun_sat_mag < 1.0:
        return 1.0

    # Apparent angular radii (radians) — no small-angle approximation
    sin_alpha = earth_radius_km / r_mag
    sin_beta  = sun_radius_km   / d_sun_sat_mag
    if sin_alpha > 1.0: sin_alpha = 1.0
    if sin_beta  > 1.0: sin_beta  = 1.0

    alpha = math.asin(sin_alpha)
    beta  = math.asin(sin_beta)

    # Angular separation γ between Earth and Sun disk centres
    cos_gamma = np.dot(-r_km, d_sun_sat) / (r_mag * d_sun_sat_mag)
    if cos_gamma >  1.0: cos_gamma =  1.0
    if cos_gamma < -1.0: cos_gamma = -1.0
    gamma = math.acos(cos_gamma)

    # -- Case 1: Full sunlight --
    if gamma >= alpha + beta:
        return 1.0

    # -- Case 2: Full umbra (Earth disk fully covers Sun) --
    if alpha >= beta and gamma <= alpha - beta:
        return 0.0

    # -- Case 2b: Annular eclipse (Sun disk larger than Earth disk) --
    # The Earth is fully inside the Sun disk: ν = 1 − A_earth / A_sun
    # Using planar disk areas: A ∝ (angular_radius)²
    if beta > alpha and gamma <= beta - alpha:
        nu = 1.0 - (alpha * alpha) / (beta * beta)
        if nu < 0.0: nu = 0.0
        return nu

    # -- Case 3: Penumbra — planar circle-circle intersection (M&G §3.4.2) --
    # Guard: γ ≈ 0 with disks overlapping but neither fully inside the other
    # (extremely degenerate; treat as full eclipse of whichever is larger)
    if gamma < 1e-12:
        if alpha >= beta:
            return 0.0
        nu = 1.0 - (alpha * alpha) / (beta * beta)
        if nu < 0.0: nu = 0.0
        return nu

    # Penumbra: intersection chord position (measured from Sun disk centre)
    # x = distance along the separation axis from the Sun disk centre to chord
    x = (gamma * gamma + beta * beta - alpha * alpha) / (2.0 * gamma)
    y = math.sqrt(max(0.0, beta * beta - x * x))

    # Overlap area = area of circular segment in Earth disk + segment in Sun disk
    # Clamped arguments to arccos to prevent domain errors at floating-point limits
    arg_earth = (gamma - x) / alpha
    arg_sun   = x / beta
    if arg_earth >  1.0: arg_earth =  1.0
    if arg_earth < -1.0: arg_earth = -1.0
    if arg_sun   >  1.0: arg_sun   =  1.0
    if arg_sun   < -1.0: arg_sun   = -1.0

    overlap_area = (
        alpha * alpha * math.acos(arg_earth)
        + beta  * beta  * math.acos(arg_sun)
        - gamma * y
    )

    # ν = 1 − (blocked area / total Sun disk area)
    nu = 1.0 - overlap_area / (math.pi * beta * beta)
    if nu < 0.0: nu = 0.0
    if nu > 1.0: nu = 1.0
    return nu


# Backwards-compatible alias: internal callers that used the old name
# will pick up the improved dual-cone implementation automatically.
_srp_illumination_factor_njit = _srp_illumination_factor_dual_cone_njit


@njit(fastmath=True, cache=True)
def srp_illumination_factor_njit(
    r_km: Any, r_sun_km: Any, earth_radius_km: Any, sun_radius_km: Any
) -> float:
    """Public NJIT wrapper for the dual-cone SRP illumination factor (FM-1B fix).

    Delegates to ``_srp_illumination_factor_dual_cone_njit``. See that function
    for the full mathematical derivation (Montenbruck & Gill §3.4.2).
    """
    return _srp_illumination_factor_dual_cone_njit(r_km, r_sun_km, earth_radius_km, sun_radius_km)  # type: ignore[no-any-return]


def srp_illumination_factor(
    r_km: np.ndarray,
    r_sun_km: np.ndarray,
    earth_radius_km: float = EARTH_EQUATORIAL_RADIUS_KM,
    sun_radius_km: float = 695700.0,
) -> float:
    """Exact dual-cone SRP illumination factor ν in [0, 1] (FM-1B fix).

    Computes the fractional illumination of the solar disk as seen from
    ``r_km``, accounting for Earth's occultation using exact spherical-cap
    intersection geometry (Montenbruck & Gill §3.4.2; Vallado 2013 Alg. 34).

    Args:
        r_km: Geocentric satellite position (km), shape (3,).
        r_sun_km: Geocentric Sun position (km), shape (3,).
        earth_radius_km: Earth equatorial radius (km).
        sun_radius_km: Solar mean radius (km).

    Returns:
        ν ∈ [0, 1]: 0 = full eclipse, 1 = full sunlight, (0,1) = penumbra.
    """
    return _srp_illumination_factor_dual_cone_njit(r_km, r_sun_km, earth_radius_km, sun_radius_km)  # type: ignore[no-any-return]


@njit(fastmath=True, cache=True)
def srp_cylindrical_illumination_factor_njit(r_km: Any, r_sun_km: Any) -> float:
    """Legacy cylindrical umbra model (ν=0 in-cylinder, ν=1 outside)."""
    d_sun_sat = r_sun_km - r_km
    d_mag = np.linalg.norm(d_sun_sat)
    if d_mag < 1.0:
        return 1.0

    # cos(gamma) = (-r . d_sun_sat) / (|r|*|d_sun_sat|)
    r_mag = np.linalg.norm(r_km)
    cos_gamma = np.dot(-r_km, d_sun_sat) / (r_mag * d_mag)
    if cos_gamma > 0.0:  # Satellite is on the night side
        # Separation distance from shadow axis
        sin_gamma = math.sqrt(max(0.0, 1.0 - cos_gamma * cos_gamma))
        dist_axis = r_mag * sin_gamma
        if dist_axis < 6378.137:
            return 0.0
    return 1.0


def srp_cylindrical_illumination_factor(r_km: Any, r_sun_km: Any) -> float:
    """Public pure-Python legacy cylindrical umbra model."""
    return srp_cylindrical_illumination_factor_njit(r_km, r_sun_km)  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NumericalState:
    """Full kinematic state vector at a single epoch.

    In 6-DOF (coast) mode, mass_kg is None and the state vector is
    [x, y, z, vx, vy, vz].

    In 7-DOF (powered) mode, mass_kg tracks propellant depletion via
    Tsiolkovsky coupling: dm/dt = −F / (Isp·g₀).

    SE-01 Fix: frozen=True to match all other ASTRA output types (OrbitalState,
    FiniteBurn, ConjunctionEvent). Prevents accidental mutation of integration
    results. numpy array *contents* remain mutable by Python semantics, but
    field references (position_km, velocity_km_s, mass_kg) cannot be reassigned.
    """

    t_jd: float
    position_km: np.ndarray  # shape (3,)
    velocity_km_s: np.ndarray  # shape (3,)
    mass_kg: Optional[float] = None
    covariance_km2: Optional[np.ndarray] = None  # shape (6, 6)

    def __post_init__(self) -> None:
        """Basic range validation on state vectors."""
        if self.position_km is not None:
            r_mag = np.linalg.norm(self.position_km)
            # AUDIT-B-06 Fix: Use correct Earth equatorial radius (~6378 km),
            # not the arbitrary 6000 km that was flagging sub-orbital but
            # physically-above-surface positions (6000–6378 km range).
            if 1.0 < r_mag < EARTH_EQUATORIAL_RADIUS_KM:
                logger.warning(
                    f"NumericalState position radius {r_mag:.2f} km is inside Earth radius."
                )
            elif r_mag < 0.1:
                # Likely an unitialized or zeroed state
                logger.debug(
                    "NumericalState initialized with nearly-zero position vector."
                )


@dataclass(frozen=True)
class DragConfig:
    """Atmospheric drag and optional solar radiation pressure inputs.

    This dataclass is frozen (immutable) to prevent accidental mutation
    after construction. Create a new instance to modify parameters.
    """

    cd: float = 2.2
    area_m2: float = 10.0
    mass_kg: float = 1000.0
    cr: float = 1.5
    include_srp: bool = True

    # Atmospheric model selection: "NRLMSISE00" (default) or "Jacchia"
    model: str = "NRLMSISE00"

    # [LOW-02 fix] `srp_conical_shadow` is the canonical name — the shadow model
    # is actually a high-fidelity *conical* Earth umbra/penumbra geometry,
    # not cylindrical.  `srp_cylindrical_shadow` is retained as a deprecated
    # alias for backward compatibility and will emit a DeprecationWarning when
    # read directly.  Use `srp_conical_shadow` in new code.
    srp_conical_shadow: bool = True

    @property
    def srp_cylindrical_shadow(self) -> bool:  # type: ignore[override]
        """Deprecated alias for srp_conical_shadow (LOW-02). Use srp_conical_shadow."""
        import warnings

        warnings.warn(
            "DragConfig.srp_cylindrical_shadow is deprecated; use srp_conical_shadow instead. "
            "The underlying model is a conical umbra/penumbra, not cylindrical.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.srp_conical_shadow


@dataclass(frozen=True)
class SNCConfig:
    """State Noise Compensation (Process Noise) configuration.

    Defines the power spectral density (PSD) of unmodeled accelerations,
    typically used to prevent covariance collapse in long-duration
    propagations.
    """

    q_psd_m2_s3: float = 1e-12  # Process noise spectral density
    mode: str = "white_noise"  # "white_noise" or "dmc"


# ---------------------------------------------------------------------------
# High-fidelity Sun / Moon via JPL DE421 (Skyfield)
# ---------------------------------------------------------------------------

# Import lazily to avoid circular dependency and allow graceful fallback
_USE_DE = True  # Will be set to False if Skyfield data unavailable


def _sun_position_de(t_jd: float) -> np.ndarray:
    """Geocentric Sun position from JPL DE421 in TEME frame (km)."""
    from astra import config

    try:
        from astra.data_pipeline import sun_position_teme

        return sun_position_teme(t_jd)
    except (ImportError, ValueError, OSError) as exc:
        from astra.errors import EphemerisError

        if config.ASTRA_STRICT_MODE:
            raise EphemerisError(
                "[ASTRA STRICT] JPL DE421 ephemeris unavailable. "
                "Cannot compute precise Sun position. "
                "Run astra.data_pipeline.load_ephemeris() or set ASTRA_STRICT_MODE=False."
            ) from exc
        logger.warning(
            "DE421 ephemeris unavailable — degrading to low-fidelity sinusoidal Sun approximation. "
            "Accuracy will be significantly reduced for HEO/GEO objects."
        )
        return _sun_position_approx(t_jd)


def _moon_position_de(t_jd: float) -> np.ndarray:
    """Geocentric Moon position from JPL DE421 in TEME frame (km)."""
    from astra import config

    try:
        from astra.data_pipeline import moon_position_teme

        return moon_position_teme(t_jd)
    except (ImportError, ValueError, OSError) as exc:
        from astra.errors import EphemerisError

        if config.ASTRA_STRICT_MODE:
            raise EphemerisError(
                "[ASTRA STRICT] JPL DE421 ephemeris unavailable. "
                "Cannot compute precise Moon position. "
                "Run astra.data_pipeline.load_ephemeris() or set ASTRA_STRICT_MODE=False."
            ) from exc
        logger.warning(
            "DE421 ephemeris unavailable — degrading to low-fidelity sinusoidal Moon approximation. "
            "Luni-solar perturbations will be inaccurate for HEO/GEO propagations."
        )
        return _moon_position_approx(t_jd)


# ---------------------------------------------------------------------------
# Analytical Fallback Ephemeris (retained for offline / no-network use)
# ---------------------------------------------------------------------------


def _sun_position_approx(t_jd: float) -> np.ndarray:
    """Approximate geocentric Sun position in ECI (km).

    Uses a simplified analytical solar ephemeris accurate to ~1° in ecliptic
    longitude. Retained as fallback when Skyfield/DE421 is unavailable.

    Based on Meeus, "Astronomical Algorithms" Chapter 25.
    """
    T = (t_jd - 2451545.0) / 36525.0  # Julian centuries from J2000

    # Mean anomaly (degrees)
    M = 357.5291092 + 35999.0502909 * T
    M_rad = math.radians(M % 360.0)

    # Ecliptic longitude (degrees)
    C = 1.9146 * math.sin(M_rad) + 0.02 * math.sin(2 * M_rad)
    L_sun = (280.46646 + 36000.76983 * T + C) % 360.0
    L_rad = math.radians(L_sun)

    # Distance in AU -> km
    R_au = 1.00014 - 0.01671 * math.cos(M_rad) - 0.00014 * math.cos(2 * M_rad)
    R_km = R_au * 149597870.7

    # Obliquity of ecliptic
    eps_rad = math.radians(23.439291 - 0.0130042 * T)

    # ECI coordinates
    x = R_km * math.cos(L_rad)
    y = R_km * math.cos(eps_rad) * math.sin(L_rad)
    z = R_km * math.sin(eps_rad) * math.sin(L_rad)

    return np.array([x, y, z])


def _moon_position_approx(t_jd: float) -> np.ndarray:
    """Approximate geocentric Moon position in ECI (km).

    Uses Brown's lunar theory simplified to first-order terms.
    Retained as fallback when Skyfield/DE421 is unavailable.
    """
    T = (t_jd - 2451545.0) / 36525.0

    # Fundamental arguments (degrees)
    L0 = (218.3165 + 481267.8813 * T) % 360.0
    M_moon = (134.9634 + 477198.8676 * T) % 360.0
    M_sun = (357.5291 + 35999.0503 * T) % 360.0
    D = (297.8502 + 445267.1115 * T) % 360.0
    F = (93.2720 + 483202.0175 * T) % 360.0

    M_moon_r = math.radians(M_moon)
    M_sun_r = math.radians(M_sun)
    D_r = math.radians(D)
    F_r = math.radians(F)

    # Longitude correction (degrees)
    dL = (
        6.289 * math.sin(M_moon_r)
        - 1.274 * math.sin(2 * D_r - M_moon_r)
        + 0.658 * math.sin(2 * D_r)
        - 0.214 * math.sin(2 * M_moon_r)
        - 0.186 * math.sin(M_sun_r)
    )

    # Latitude (degrees)
    B = (
        5.128 * math.sin(F_r)
        + 0.281 * math.sin(M_moon_r + F_r)
        - 0.278 * math.sin(F_r - M_moon_r)
    )

    # Distance (km)
    R_km = (
        385000.56
        - 20905.36 * math.cos(M_moon_r)
        - 3699.11 * math.cos(2 * D_r - M_moon_r)
        - 2955.97 * math.cos(2 * D_r)
    )

    lon_rad = math.radians(L0 + dL)
    lat_rad = math.radians(B)

    # Obliquity
    eps_rad = math.radians(23.439291 - 0.0130042 * T)

    # Ecliptic -> ECI
    x_ecl = R_km * math.cos(lat_rad) * math.cos(lon_rad)
    y_ecl = R_km * math.cos(lat_rad) * math.sin(lon_rad)
    z_ecl = R_km * math.sin(lat_rad)

    x = x_ecl
    y = y_ecl * math.cos(eps_rad) - z_ecl * math.sin(eps_rad)
    z = y_ecl * math.sin(eps_rad) + z_ecl * math.cos(eps_rad)

    return np.array([x, y, z])


# ---------------------------------------------------------------------------
# Empirical Atmospheric Drag
# ---------------------------------------------------------------------------


def _atmospheric_density(
    alt_km: float, t_jd: float, use_empirical: bool = True
) -> float:
    """Get atmospheric density in kg/m³.

    If `use_empirical` is True and space-weather data is available,
    uses the NRLMSISE-00 model from data_pipeline.  Otherwise falls
    back to the static exponential model.

    In STRICT_MODE, if space-weather data is unavailable, a SpaceWeatherError
    is raised instead of silently degrading to the static model.

    Args:
        alt_km: Altitude above surface in km.
        t_jd: Julian Date (needed for space weather lookup).
        use_empirical: Try empirical model first.

    Returns:
        Density in kg/m³.
    """
    if alt_km > 1500.0 or alt_km < 0.0:
        return 0.0

    if use_empirical:
        try:
            from astra.data_pipeline import (
                get_space_weather,
                atmospheric_density_empirical,
            )
            from astra.errors import SpaceWeatherError

            try:
                f107_obs, f107_adj, ap_daily = get_space_weather(t_jd)
                return atmospheric_density_empirical(
                    alt_km, f107_obs, f107_adj, ap_daily
                )
            except SpaceWeatherError:
                raise
            except (ImportError, ValueError, OSError):
                from astra import config

                if config.ASTRA_STRICT_MODE:
                    raise SpaceWeatherError(
                        "[ASTRA STRICT] Empirical atmospheric density unavailable. "
                        "Load space weather data or set ASTRA_STRICT_MODE=False."
                    )
                logger.warning(
                    "Empirical atmospheric model unavailable — using static exponential fallback."
                )
        except SpaceWeatherError:
            raise  # SpaceWeatherError must always propagate (STRICT_MODE gate)

    from astra.constants import (
        DRAG_REF_DENSITY_KG_M3,
        DRAG_REF_ALTITUDE_KM,
        DRAG_SCALE_HEIGHT_KM,
        DRAG_MIN_ALTITUDE_KM,
        DRAG_MAX_ALTITUDE_KM,
    )

    if alt_km > DRAG_MAX_ALTITUDE_KM or alt_km < 0.0:
        return 0.0
    if alt_km < DRAG_MIN_ALTITUDE_KM:
        return 0.0
    return DRAG_REF_DENSITY_KG_M3 * math.exp(
        -(alt_km - DRAG_REF_ALTITUDE_KM) / DRAG_SCALE_HEIGHT_KM
    )


# ---------------------------------------------------------------------------
# NRLMSISE-00 Core (Numba Optimized)
# ---------------------------------------------------------------------------


@njit(fastmath=True, cache=True)
def _msis_bates_temperature_njit(
    z_km: float, z_lb_km: float, T_lb: float, T_inf: float, s: float
) -> float:
    if z_km <= z_lb_km:
        return T_lb
    xi = (z_km - z_lb_km) * (6378.137 + z_lb_km) / (6378.137 + z_km)
    return T_inf - (T_inf - T_lb) * math.exp(-s * xi)


@njit(fastmath=True, cache=True)
def _nrlmsise00_density_njit(
    altitude_km: float,
    f107_obs: float,
    f107_adj: float,
    ap_daily: float,
) -> float:
    """NRLMSISE-00-class density — canonical Numba implementation.

    Uses the Bates exospheric temperature profile + effective molecular weight
    blend + reference-density calibration at 400 km, physically consistent with
    the Python ``nrlmsise00_density`` in data_pipeline.py [CRIT-06 fix].

    Calibration anchor (Picone et al. 2002):
        rho(400 km, F10.7=150, Ap=15) = 3.7e-12 kg/m³, T_inf ≈ 948 K.
    """
    if altitude_km > 1500.0 or altitude_km < 100.0:
        return 0.0

    # ── 1. Exospheric temperature (à la Hedin 1991 / MSIS-90) ────────────────
    T_inf = 379.0 + 3.24 * f107_adj + 1.3 * (f107_obs - f107_adj) + 28.0 * ap_daily**0.4
    T_inf = max(500.0, min(T_inf, 2500.0))

    # ── 2. Reference calibration anchor ─────────────────────────────────
    T_inf_ref = 948.0  # T_inf at moderate activity (F10.7=150, Ap=15)
    rho_ref_400 = 3.7e-12  # kg/m³ at 400 km under reference conditions

    # ── 3. Bates temperature profile parameters ────────────────────────
    z_lb = 120.0  # lower boundary (km)
    T_lb = 380.0  # temperature at lower boundary (K)
    s = 0.02  # Bates slope parameter (1/km)
    Re = 6378.137  # WGS84 equatorial radius (km)

    # T(z) via Bates profile
    T_z = _msis_bates_temperature_njit(altitude_km, z_lb, T_lb, T_inf, s)

    # ── 4. Effective molecular weight blend (He-dominated above 800 km) ───
    M_eff = 4.0e-3 + (28.0 - 4.0) * 1e-3 * math.exp(-(altitude_km - 120.0) / 160.0)
    M_eff = max(M_eff, 4.0e-3)  # clamp to helium floor

    # ── 5. Gravity and scale height at target altitude ──────────────────
    R_GAS = 8.314462618  # J/(K·mol)
    g_z = 9.80665 * (Re / (Re + altitude_km)) ** 2  # m/s²
    H_z = R_GAS * T_z / (M_eff * g_z) / 1000.0  # scale height (km)

    # ── 6. Density at 400 km for current vs reference activity ─────────
    T_z_ref_400 = _msis_bates_temperature_njit(400.0, z_lb, T_lb, T_inf_ref, s)
    T_z_cur_400 = _msis_bates_temperature_njit(400.0, z_lb, T_lb, T_inf, s)
    g_400 = 9.80665 * (Re / (Re + 400.0)) ** 2
    M_400 = 4.0e-3 + (28.0 - 4.0) * 1e-3 * math.exp(-(400.0 - 120.0) / 160.0)
    H_ref_400 = R_GAS * T_z_ref_400 / (M_400 * g_400) / 1000.0
    H_cur_400 = R_GAS * T_z_cur_400 / (M_400 * g_400) / 1000.0
    dz_lb = 400.0 - z_lb  # 280 km integration path
    rho_400 = rho_ref_400 * math.exp(dz_lb * (1.0 / H_ref_400 - 1.0 / H_cur_400))

    # ── 7. Extrapolate from 400 km to target altitude ─────────────────
    rho = rho_400 * (T_z_cur_400 / T_z) * math.exp(-(altitude_km - 400.0) / H_z)
    return max(rho, 1e-20)  # type: ignore[no-any-return]


@njit(fastmath=True, cache=True)
def _compute_force_jacobian(r: np.ndarray, v: np.ndarray, mu: float) -> np.ndarray:
    """Compute 6×6 state Jacobian F = df/dx of the force model (Two-body + J2).

    [MED-05 fix] Includes the analytical J2 partial-derivative contribution
    (Montenbruck & Gill §3.2.4) so that the STM-propagated covariance grows
    at the correct rate in LEO (J2 drives the dominant nodal-precession term).

    F = [ 0_3x3   I_3x3 ]
        [ G_3x3   0_3x3 ]

    where G = da/dr = G_2body + G_J2.
    """
    Re = 6378.137
    J2c = 0.00108262668

    r_mag = np.linalg.norm(r)
    r2 = r_mag * r_mag
    r3 = r2 * r_mag
    r5 = r3 * r2
    r5 * r2

    x, y, z = r[0], r[1], r[2]

    # ── Two-Body Jacobian G_2body[i,j] = (mu/r^3)(3 ri rj / r^2 - delta_ij) ──
    G = np.zeros((3, 3))
    inv_r2 = 1.0 / r2
    mu_r3 = mu / r3
    for i in range(3):
        for j in range(3):
            G[i, j] = mu_r3 * 3.0 * r[i] * r[j] * inv_r2
            if i == j:
                G[i, j] -= mu_r3

    # ── J2 Jacobian (Exact analytical partials) ─────────────────────────
    # a_J2 = fJ2 * [ x*(5 z^2/r^2 - 1), y*(5 z^2/r^2 - 1), z*(5 z^2/r^2 - 3) ]^T
    fJ2 = 1.5 * J2c * mu * Re * Re / r5
    z2_r2 = z * z * inv_r2

    # Common factors
    c1 = 5.0 * z2_r2 - 1.0
    c2 = 5.0 * z2_r2 - 3.0
    inv_r2_5 = 5.0 * inv_r2

    # da_i / dx_j
    G_J2_xx = fJ2 * (c1 - x * x * inv_r2_5 * (7.0 * z2_r2 - 1.0))
    G_J2_xy = fJ2 * (-x * y * inv_r2_5 * (7.0 * z2_r2 - 1.0))
    G_J2_xz = fJ2 * (x * z * inv_r2_5 * (3.0 - 7.0 * z2_r2))

    G_J2_yy = fJ2 * (c1 - y * y * inv_r2_5 * (7.0 * z2_r2 - 1.0))
    G_J2_yz = fJ2 * (y * z * inv_r2_5 * (3.0 - 7.0 * z2_r2))

    G_J2_zz = fJ2 * (c2 - z * z * inv_r2_5 * (7.0 * z2_r2 - 5.0))

    # Add to two-body Jacobian
    G[0, 0] += G_J2_xx
    G[0, 1] += G_J2_xy
    G[0, 2] += G_J2_xz

    G[1, 0] += G_J2_xy
    G[1, 1] += G_J2_yy
    G[1, 2] += G_J2_yz

    G[2, 0] += G_J2_xz
    G[2, 1] += G_J2_yz
    G[2, 2] += G_J2_zz

    F = np.zeros((6, 6))
    F[0, 3] = 1.0
    F[1, 4] = 1.0
    F[2, 5] = 1.0
    for i in range(3):
        for j in range(3):
            F[i + 3, j] = G[i, j]
    return F


# ---------------------------------------------------------------------------
# Force Model (shared between coast and powered derivatives)
# ---------------------------------------------------------------------------


@njit(fastmath=True, cache=True)
def _acceleration_v1_njit(
    t_jd: float,
    r: np.ndarray,
    v: np.ndarray,
    use_drag: bool,
    drag_cd: float,
    drag_area_m2: float,
    drag_mass_kg: float,
    drag_rho: float,
    drag_H_km: float,
    drag_ref_alt_km: float,
    f107_obs: float,
    f107_adj: float,
    ap_daily: float,
    hf_atmosphere: bool,
    include_third_body: bool,
    t_jd0_abs: float,
    duration_d: float,
    sun_coeffs: np.ndarray,
    moon_coeffs: np.ndarray,
    use_srp: float,
    srp_cr: float,
    srp_use_shadow: bool,
) -> np.ndarray:
    """Compute total gravitational + drag acceleration in ECI (km/s²).

    Forces:
        1. Two-body + J2/J3/J4 zonal harmonics
        2. Atmospheric drag (per-substep exponential sampling)
        3. Solar/Lunar third-body gravity (Chebyshev splined)
        4. SRP (cannonball with cylindrical shadow)
    """
    r_mag = np.linalg.norm(r)
    if r_mag < 1.0:
        return np.zeros(3)

    x, y, z = r
    Re = EARTH_EQUATORIAL_RADIUS_KM
    mu = EARTH_MU_KM3_S2

    r2 = r_mag * r_mag
    r3 = r2 * r_mag
    r5 = r3 * r2
    r7 = r5 * r2
    r9 = r7 * r2

    z2 = z * z

    # --- Two-body ---
    a_total = -mu / r3 * r

    # --- J2 Perturbation ---
    fJ2 = 1.5 * J2 * mu * Re**2 / r5
    a_total += np.array(
        [
            fJ2 * x * (5.0 * z2 / r2 - 1.0),
            fJ2 * y * (5.0 * z2 / r2 - 1.0),
            fJ2 * z * (5.0 * z2 / r2 - 3.0),
        ]
    )

    # --- J3 Perturbation ---
    fJ3 = 0.5 * J3 * mu * Re**3 / r7
    a_total += np.array(
        [
            fJ3 * x * (35.0 * z2 * z / r2 - 15.0 * z),
            fJ3 * y * (35.0 * z2 * z / r2 - 15.0 * z),
            fJ3 * (35.0 * z2 * z2 / r2 - 30.0 * z2 + 3.0 * r2),
        ]
    )

    # --- J4 Perturbation ---
    fJ4 = 0.625 * J4 * mu * Re**4 / r9
    z4 = z2 * z2
    a_total += np.array(
        [
            fJ4 * x * (63.0 * z4 / r2 - 42.0 * z2 + 3.0 * r2),
            fJ4 * y * (63.0 * z4 / r2 - 42.0 * z2 + 3.0 * r2),
            fJ4 * z * (63.0 * z4 / r2 - 70.0 * z2 + 15.0 * r2),
        ]
    )

    # --- Atmospheric Drag (PHY-B Standardized) ---
    alt_km = r_mag - Re
    if use_drag and alt_km < 1500.0:
        if hf_atmosphere:
            # High-Fidelity: call MSIS-00 core directly (njit-safe)
            rho_instant = _nrlmsise00_density_njit(alt_km, f107_obs, f107_adj, ap_daily)
        elif drag_rho > 0.0:
            # Strategy A: Use local exponential correction
            rho_instant = drag_rho * math.exp(-(alt_km - drag_ref_alt_km) / drag_H_km)
        else:
            rho_instant = 0.0

        # Atmosphere co-rotates with Earth
        omega_earth = np.array([0.0, 0.0, EARTH_OMEGA_RAD_S])
        v_rel = v - np.cross(omega_earth, r)
        v_rel_mag = np.linalg.norm(v_rel)

        if v_rel_mag > 1e-10:
            Bc = drag_cd * drag_area_m2 / drag_mass_kg
            # 1e-6 (m^2 to km^2) * 1e9 (kg/m^3 to kg/km^3) = 1e3
            a_drag = -0.5 * rho_instant * 1e3 * Bc * v_rel_mag * v_rel
            a_total += a_drag

    # --- Third-Body (Sun & Moon) via Chebyshev Splines ---
    if include_third_body:
        # CRIT-02 fix: was referencing bare name `t_jd0`; corrected to `t_jd0_abs`.
        t_norm = 2.0 * (t_jd - t_jd0_abs) / duration_d - 1.0
        t_norm = max(-1.0, min(1.0, t_norm))

        # DEF-009: removed self-import — _eval_cheb_3d_njit is defined in this module
        sun_pos = _eval_cheb_3d_njit(t_norm, sun_coeffs)
        moon_pos = _eval_cheb_3d_njit(t_norm, moon_coeffs)

        # Sun Gravity
        d_sun = sun_pos - r
        d_mag_sun = np.linalg.norm(d_sun)
        r_mag_sun = np.linalg.norm(sun_pos)
        if d_mag_sun > 1.0 and r_mag_sun > 1.0:
            a_total += SUN_MU_KM3_S2 * (
                d_sun / (d_mag_sun**3) - sun_pos / (r_mag_sun**3)
            )

        # Moon Gravity
        d_moon = moon_pos - r
        d_mag_moon = np.linalg.norm(d_moon)
        r_mag_moon = np.linalg.norm(moon_pos)
        if d_mag_moon > 1.0 and r_mag_moon > 1.0:
            a_total += MOON_MU_KM3_S2 * (
                d_moon / (d_mag_moon**3) - moon_pos / (r_mag_moon**3)
            )

        # --- Solar radiation pressure (cannonball; scale flux from 1 AU)
        if use_srp and drag_area_m2 > 0.0 and drag_mass_kg > 0.0:
            d_ss = r - sun_pos
            d_mag_ss = float(np.linalg.norm(d_ss))
            if d_mag_ss > 1.0:
                # Use named constants (SE-09: eliminate magic numbers in Python path)
                scale = (_AU_KM_CONST / d_mag_ss) ** 2
                amag = _SRP_P0 * scale * srp_cr * (drag_area_m2 / drag_mass_kg) / 1000.0

                nu = 1.0
                if srp_use_shadow:
                    # Upgrade from cylindrical to conical shadow (PHY-D)
                    from astra.constants import SUN_RADIUS_KM

                    nu = _srp_illumination_factor_njit(
                        r, sun_pos, EARTH_EQUATORIAL_RADIUS_KM, SUN_RADIUS_KM
                    )
                a_total += (nu * amag) * (d_ss / d_mag_ss)

    return a_total  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# Pure-Python mirror of the acceleration kernel (no Numba)
# ---------------------------------------------------------------------------
# ``_acceleration`` has the *same* 18-argument signature as the real Numba
# kernel ``_acceleration_njit``.  Tests import both names and assert the
# outputs agree within 1e-6 rtol (IEEE-754 vs fastmath tolerances).


def _acceleration(
    t_jd: float,
    r: np.ndarray,
    v: np.ndarray,
    use_drag: bool,
    drag_cd: float,
    drag_area_m2: float,
    drag_mass_kg: float,
    drag_rho: float,
    drag_H_km: float,
    drag_ref_alt_km: float,
    f107_obs: float,
    f107_adj: float,
    ap_daily: float,
    hf_atmosphere: bool,
    include_third_body: bool,
    t_jd0: float,
    duration_d: float,
    sun_coeffs: np.ndarray,
    moon_coeffs: np.ndarray,
    use_srp: bool,
    srp_cr: float,
    srp_use_shadow: bool,
) -> np.ndarray:
    """Pure-Python acceleration mirror for _acceleration_njit (22-arg signature).

    [CRIT-01 / HIGH-03 fix] Signature expanded from 18 to 22 args to include
    f107_obs, f107_adj, ap_daily, hf_atmosphere, keeping parity with the Numba
    kernel.  Tests import both names and assert outputs agree within 1e-6 rtol.
    Intentionally avoids ``fastmath=True`` for IEEE-754-compliant output.
    """
    r_mag = float(np.linalg.norm(r))
    if r_mag < 1.0:
        return np.zeros(3)

    x, y, z = float(r[0]), float(r[1]), float(r[2])

    # [FM-2 Fix — Finding #6] Use module-level constants imported from astra.constants
    # instead of hardcoded literals so this Python mirror cannot drift from constants.py.
    # The Numba kernel must still inline literals (cannot import at JIT time) but the
    # Python mirror can and MUST use the canonical module-level names.
    Re = EARTH_EQUATORIAL_RADIUS_KM
    mu = EARTH_MU_KM3_S2
    _J2 = J2
    _J3 = J3
    _J4 = J4
    SUN_MU = SUN_MU_KM3_S2
    MOON_MU = MOON_MU_KM3_S2
    # J5/J6 are defined in constants.py; Python mirror uses them (Numba kernel also added).
    from astra.constants import J5 as _J5_const, J6 as _J6_const
    _J5 = _J5_const
    _J6 = _J6_const

    r2 = r_mag * r_mag
    r3 = r2 * r_mag
    r5 = r3 * r2
    r7 = r5 * r2
    r9 = r7 * r2
    z2 = z * z

    # --- Two-body ---
    a_total = -mu / r3 * r.copy()

    # --- J2 Perturbation ---
    fJ2 = 1.5 * _J2 * mu * Re**2 / r5
    a_total[0] += fJ2 * x * (5.0 * z2 / r2 - 1.0)
    a_total[1] += fJ2 * y * (5.0 * z2 / r2 - 1.0)
    a_total[2] += fJ2 * z * (5.0 * z2 / r2 - 3.0)

    # --- J3 Perturbation ---
    fJ3 = 0.5 * _J3 * mu * Re**3 / r7
    a_total[0] += fJ3 * x * (35.0 * z2 * z / r2 - 15.0 * z)
    a_total[1] += fJ3 * y * (35.0 * z2 * z / r2 - 15.0 * z)
    a_total[2] += fJ3 * (35.0 * z2 * z2 / r2 - 30.0 * z2 + 3.0 * r2)

    # --- J4 Perturbation ---
    fJ4 = 0.625 * _J4 * mu * Re**4 / r9
    z4 = z2 * z2
    a_total[0] += fJ4 * x * (63.0 * z4 / r2 - 42.0 * z2 + 3.0 * r2)
    a_total[1] += fJ4 * y * (63.0 * z4 / r2 - 42.0 * z2 + 3.0 * r2)
    a_total[2] += fJ4 * z * (63.0 * z4 / r2 - 70.0 * z2 + 15.0 * r2)

    # --- J5 Perturbation ---
    # Ref: Vallado §8.7.2; EGM96 coefficient (Lemoine et al. 1998).
    # Significant for MEO/GEO long-horizon secular nodal precession accuracy.
    r11 = r9 * r2
    z5 = z4 * z
    fJ5 = -(15.0 / 8.0) * _J5 * mu * Re**5 / r11
    a_total[0] += fJ5 * x * (21.0 * z4 / r2 - 14.0 * z2 + r2 / 3.0)
    a_total[1] += fJ5 * y * (21.0 * z4 / r2 - 14.0 * z2 + r2 / 3.0)
    a_total[2] += fJ5 * (21.0 * z5 / r2 - 21.0 / 2.0 * z2 * z + 5.0 / 2.0 * r2 * z)

    # --- J6 Perturbation ---
    # Ref: EGM96 coefficient (Lemoine et al. 1998).
    r13 = r11 * r2
    z6 = z4 * z2
    fJ6 = -(1.0 / 16.0) * _J6 * mu * Re**6 / r13
    a_total[0] += (
        fJ6 * x * (231.0 * z6 / r2 - 315.0 * z4 + 105.0 * z2 * r2 - 5.0 * r2 * r2)
    )
    a_total[1] += (
        fJ6 * y * (231.0 * z6 / r2 - 315.0 * z4 + 105.0 * z2 * r2 - 5.0 * r2 * r2)
    )
    a_total[2] += fJ6 * (
        231.0 * z6 * z / r2
        - 378.0 * z6 / r2 * r2
        + 189.0 * z5
        - 70.0 * z4 * z
        + 15.0 * z * r2 * r2
    )

    # --- Atmospheric Drag (exponential profile, Strategy A; or NRLMSISE-00) ---
    alt_km = r_mag - Re
    if use_drag and alt_km < 1500.0:
        if hf_atmosphere:
            # [MSIS SYNC FIX] Both Python and Numba paths now call the same
            # _nrlmsise00_density_njit kernel instead of the data_pipeline wrapper.
            # This eliminates the altitude-range mismatch (data_pipeline returns 0
            # for alt < 100 km; _nrlmsise00_density_njit also returns 0 for < 100 km)
            # and ensures test_g400_computation_matches_propagator passes.
            rho_instant = _nrlmsise00_density_njit(alt_km, f107_obs, f107_adj, ap_daily)
        elif drag_rho > 0.0:
            rho_instant = drag_rho * math.exp(-(alt_km - drag_ref_alt_km) / drag_H_km)
        else:
            rho_instant = 0.0
        if rho_instant > 0.0:
            # [FM-9 Fix — Finding #10] Use imported constant, not hardcoded literal.
            omega_earth = np.array([0.0, 0.0, EARTH_OMEGA_RAD_S])
            v_rel = v - np.cross(omega_earth, r)
            v_rel_mag = float(np.linalg.norm(v_rel))
            if v_rel_mag > 1e-10:
                Bc = drag_cd * drag_area_m2 / drag_mass_kg
                a_drag = -0.5 * rho_instant * 1e3 * Bc * v_rel_mag * v_rel
                a_total += a_drag

    # --- Third-Body (Sun & Moon, piecewise Chebyshev spline) ---
    if include_third_body:
        idx = int((t_jd - t_jd0) / 7.0)
        num_pieces = sun_coeffs.shape[0]
        idx = max(0, min(idx, num_pieces - 1))
        piece_t_jd0 = t_jd0 + idx * 7.0
        piece_duration = 7.0
        t_norm = 2.0 * (t_jd - piece_t_jd0) / piece_duration - 1.0
        t_norm = max(-1.0, min(1.0, t_norm))

        sun_pos = _eval_cheb_3d_njit(t_norm, sun_coeffs[idx])
        moon_pos = _eval_cheb_3d_njit(t_norm, moon_coeffs[idx])

        d_sun = sun_pos - r
        d_mag_sun = float(np.linalg.norm(d_sun))
        r_mag_sun = float(np.linalg.norm(sun_pos))
        if d_mag_sun > 1.0 and r_mag_sun > 1.0:
            a_total += SUN_MU * (d_sun / d_mag_sun**3 - sun_pos / r_mag_sun**3)

        d_moon = moon_pos - r
        d_mag_moon = float(np.linalg.norm(d_moon))
        r_mag_moon = float(np.linalg.norm(moon_pos))
        if d_mag_moon > 1.0 and r_mag_moon > 1.0:
            a_total += MOON_MU * (d_moon / d_mag_moon**3 - moon_pos / r_mag_moon**3)

        if use_srp and drag_area_m2 > 0.0 and drag_mass_kg > 0.0:
            d_ss = r - sun_pos
            d_mag_ss = float(np.linalg.norm(d_ss))
            if d_mag_ss > 1.0:
                nu = 1.0
                if srp_use_shadow:
                    nu = float(_srp_illumination_factor_njit(r, sun_pos, Re, 695700.0))
                P0 = 4.56e-6
                AU = 149597870.7
                scale = (AU / d_mag_ss) ** 2
                amag = P0 * scale * srp_cr * (drag_area_m2 / drag_mass_kg) / 1000.0
                a_total += (nu * amag) * (d_ss / d_mag_ss)

    return a_total


# Standard gravitational acceleration for mass flow (references constants.G0_STD)
_G0 = G0_STD  # m/s²


# ---------------------------------------------------------------------------
# Coast Derivative (6-DOF, m = constant)
# ---------------------------------------------------------------------------


def _coast_derivative(
    t_sec: float,
    y: np.ndarray,
    t_jd0_segment: float,
    use_drag: bool,
    drag_cd: float,
    drag_area_m2: float,
    drag_mass_kg: float,
    drag_rho: float,
    drag_H_km: float,
    drag_ref_alt_km: float,
    f107_obs: float,
    f107_adj: float,
    ap_daily: float,
    hf_atmosphere: bool,
    include_third_body: bool,
    global_t_jd0: float,
    duration_d: float,
    sun_coeffs: np.ndarray,
    moon_coeffs: np.ndarray,
    use_srp: bool,
    srp_cr: float,
    srp_use_shadow: bool,
) -> np.ndarray:
    """State derivative for unpowered (coast) arcs — pure-Python fallback.

    [CRIT-01 fix] Previously referenced undefined module-level names
    ``f107_obs_seg``, ``f107_adj_seg``, ``ap_daily_seg``, ``hf_atm_seg``
    which caused a NameError on any system without Numba.  These are now
    proper explicit parameters matching the Numba path signature.

    State vector y = [x, y, z, vx, vy, vz] (6 components).
    Returns dy/dt = [vx, vy, vz, ax, ay, az].
    """
    r = y[:3]
    v = y[3:6]
    t_jd = t_jd0_segment + t_sec / SECONDS_PER_DAY
    a = _acceleration(
        t_jd,
        r,
        v,
        use_drag,
        drag_cd,
        drag_area_m2,
        drag_mass_kg,
        drag_rho,
        drag_H_km,
        drag_ref_alt_km,
        f107_obs,
        f107_adj,
        ap_daily,
        hf_atmosphere,
        include_third_body,
        global_t_jd0,
        duration_d,
        sun_coeffs,
        moon_coeffs,
        use_srp,
        srp_cr,
        srp_use_shadow,
    )
    return np.concatenate([v, a])


# ---------------------------------------------------------------------------
# Powered Derivative (7-DOF, thrust + mass depletion)
# ---------------------------------------------------------------------------


def _powered_derivative(
    t_sec: float,
    y: np.ndarray,
    t_jd0_segment: float,
    use_drag: bool,
    drag_cd: float,
    drag_area_m2: float,
    drag_mass_kg: float,
    drag_rho: float,
    drag_H_km: float,
    drag_ref_alt_km: float,  # AUDIT-C-03 Fix: was missing; caused arity mismatch in Python path
    f107_obs: float,  # [SYNC FIX] Space weather params forwarded to _acceleration
    f107_adj: float,
    ap_daily: float,
    hf_atmosphere: bool,
    include_third_body: bool,
    global_t_jd0: float,
    duration_d: float,
    sun_coeffs: np.ndarray,
    moon_coeffs: np.ndarray,
    use_srp: bool,
    srp_cr: float,
    srp_use_shadow: bool,
    burn_thrust_N: float,
    burn_isp_s: float,
    burn_dir: np.ndarray,
    burn_frame_idx: int,
) -> np.ndarray:
    """State derivative for powered (thrusting) arcs — pure-Python fallback.

    State vector y = [x, y, z, vx, vy, vz, mass_kg]  (7 components).

    The thrust direction is re-computed from the instantaneous r, v at
    every sub-step, implementing dynamic attitude steering.

    Returns dy/dt = [vx, vy, vz, ax, ay, az, dm/dt].

    Mass depletion: dm/dt = −F / (Isp·g₀).

    [SYNC FIX] Signature now includes f107_obs, f107_adj, ap_daily, hf_atmosphere
    in parity with _powered_derivative_njit.  Previously these were absent, causing
    _acceleration() to silently receive wrong defaults when using the Python fallback
    path (e.g., when Numba is not installed via _numba_compat).
    """
    r = y[:3]
    v = y[3:6]
    m = y[6]
    t_jd = t_jd0_segment + t_sec / SECONDS_PER_DAY

    # PHY-03 Fix: Use instantaneous mass m (from state y[6]) for ballistic coefficient,
    # not the pre-burn drag_mass_kg which remains fixed at the burn start mass.
    # For short burns (<1% mass depletion) the difference is negligible, but for
    # long low-thrust electric propulsion arcs, the drag systematically grows as
    # propellant burns off and B_c = Cd*A/m should increase accordingly.
    instantaneous_mass_kg = max(
        m, 1e-6
    )  # floor at 1 mg — supports FEEP/colloid micro-thrusters

    # Gravitational + drag acceleration (using instantaneous mass for Bc)
    a_grav = _acceleration(
        t_jd,
        r,
        v,
        use_drag,
        drag_cd,
        drag_area_m2,
        instantaneous_mass_kg,
        drag_rho,
        drag_H_km,
        drag_ref_alt_km,
        f107_obs,
        f107_adj,
        ap_daily,
        hf_atmosphere,
        include_third_body,
        global_t_jd0,
        duration_d,
        sun_coeffs,
        moon_coeffs,
        use_srp,
        srp_cr,
        srp_use_shadow,
    )

    # Thrust acceleration (km/s²)
    # We use a manual reconstruction for parity with NJIT kernel
    max(1e-12, np.linalg.norm(r))
    thrust_a_mag = (burn_thrust_N / 1000.0) / m

    if burn_frame_idx == 0:  # VNB
        from astra.frames import _build_vnb_matrix_njit

        rot_matrix = _build_vnb_matrix_njit(r, v)
    else:  # RTN
        from astra.frames import _build_rtn_matrix_njit

        rot_matrix = _build_rtn_matrix_njit(r, v)

    a_thrust = np.dot(rot_matrix, burn_dir) * thrust_a_mag
    a_total = a_grav + a_thrust

    # Mass flow rate — use G0_STD from constants (not hardcoded literal)
    dm_dt = -burn_thrust_N / (burn_isp_s * G0_STD)

    return np.concatenate([v, a_total, [dm_dt]])


# ---------------------------------------------------------------------------
# Numba Compiled HPC Functions
# ---------------------------------------------------------------------------
# Physical constants in ``_acceleration_njit`` / related kernels are inlined as
# float literals because Numba cannot import ``astra.constants`` at compile time.
# Keep μ, Re, J2, J3, J4, and third-body GM in sync with ``astra.constants``.


@njit(fastmath=True, cache=True)
def _eval_cheb_3d_njit(t_norm: float, coeffs: np.ndarray) -> np.ndarray:
    """Evaluate 3D Chebyshev polynomials via Clenshaw recurrence.
    t_norm: scalar in [-1, 1]
    coeffs: array shape (N, 3), where N is number of coefficients.
    Returns: shape (3,) array.
    """
    N = coeffs.shape[0]
    if N == 0:
        return np.zeros(3)
    if N == 1:
        return np.copy(coeffs[0])
    if N == 2:
        return coeffs[0] + coeffs[1] * t_norm  # type: ignore[no-any-return]

    x2 = 2.0 * t_norm
    d1: np.ndarray = np.zeros(3)
    d2: np.ndarray = np.zeros(3)
    # [HIGH-01 fix]: Previously `range(N-1, 1, -1)` which stopped at index 2,
    # dropping the T1 (linear) Chebyshev term.  Must stop at index 1 (exclusive 0)
    # to include all coefficients coeffs[1] through coeffs[N-1].
    for i in range(N - 1, 0, -1):
        temp: np.ndarray = np.copy(d1)
        d1 = x2 * d1 - d2 + coeffs[i]
        d2 = temp
    return coeffs[0] + t_norm * d1 - d2  # type: ignore[no-any-return]


@njit(fastmath=True, cache=True)
def _acceleration_njit(
    t_jd: float,
    r: np.ndarray,
    v: np.ndarray,
    use_drag: bool,
    drag_cd: float,
    drag_area_m2: float,
    drag_mass_kg: float,
    drag_rho: float,
    drag_H_km: float,
    drag_ref_alt_km: float,
    f107_obs: float,
    f107_adj: float,
    ap_daily: float,
    hf_atmosphere: bool,
    include_third_body: bool,
    t_jd0: float,
    duration_d: float,
    sun_coeffs: np.ndarray,
    moon_coeffs: np.ndarray,
    use_srp: bool,
    srp_cr: float,
    srp_use_shadow: bool,
) -> np.ndarray:
    """Numba-compiled acceleration kernel (22 args): two-body + J2/J3/J4 + drag + 3rd-body + SRP.

    [HIGH-03 fix] Now accepts f107_obs, f107_adj, ap_daily, hf_atmosphere so the
    NRLMSISE-00 model is actually used when DragConfig(model='NRLMSISE00') is set.
    Previously these parameters were accepted by _coast_derivative_njit but were
    silently dropped before reaching this kernel.
    """
    r_mag = np.linalg.norm(r)
    if r_mag < 1.0:
        return np.zeros(3)

    x, y, z = r[0], r[1], r[2]

    # Inlined from constants — Numba cannot import astra.constants at JIT compile time.
    # assert guards in constants.py will catch drift at Python import time.
    # Keep ALL values in sync with astra.constants manually.
    Re = 6378.137          # EARTH_EQUATORIAL_RADIUS_KM — guarded by constants.py assert
    mu = 398600.4418       # EARTH_MU_KM3_S2
    J2 = 0.00108262668     # synced from constants.J2
    J3 = -0.00000253265649 # WGS-84/EGM96; synced from constants.J3 = -2.53265649e-6
    J4 = -0.00000161962159 # WGS-84/EGM96; synced from constants.J4 = -1.61962159e-6
    # [FM-1/FM-6 Fix — Finding #4/#21] J5/J6 added to Numba production kernel.
    # Previously defined in constants.py but silently absent from this kernel.
    # Skipping J5 introduces ~3-10 m/day secular nodal-rate error above 20,000 km.
    # Ref: EGM96 (Lemoine et al. 1998, NASA/TP-1998-206861).
    J5 = -2.27626414e-7    # synced from constants.J5
    J6 = 5.40681239e-7     # synced from constants.J6
    # SUN_MU matches ``constants.SUN_MU_KM3_S2`` (IAU-style GM).
    SUN_MU = 1.32712440018e11
    MOON_MU = 4902.800066
    # Earth rotation rate — guarded by constants.py assert.
    OMEGA_EARTH = 7.292115146706979e-5  # rad/s — IAU/IERS 2010

    r2 = r_mag * r_mag
    r3 = r2 * r_mag

    # --- Two-body ---
    a_total = -mu / r3 * r

    alt_km = r_mag - Re

    r5 = r3 * r2
    r7 = r5 * r2
    r9 = r7 * r2
    z2 = z * z

    # --- J2 Perturbation ---
    fJ2 = 1.5 * J2 * mu * Re**2 / r5
    a_j2_x = fJ2 * x * (5.0 * z2 / r2 - 1.0)
    a_j2_y = fJ2 * y * (5.0 * z2 / r2 - 1.0)
    a_j2_z = fJ2 * z * (5.0 * z2 / r2 - 3.0)

    # --- J3 Perturbation ---
    fJ3 = 0.5 * J3 * mu * Re**3 / r7
    a_j3_x = fJ3 * x * (35.0 * z2 * z / r2 - 15.0 * z)
    a_j3_y = fJ3 * y * (35.0 * z2 * z / r2 - 15.0 * z)
    a_j3_z = fJ3 * (35.0 * z2 * z2 / r2 - 30.0 * z2 + 3.0 * r2)

    # --- J4 Perturbation ---
    # Vallado Eq. 8-31: coefficient +5/8 (0.625) times J4 (J4 < 0).
    fJ4 = 0.625 * J4 * mu * Re**4 / r9
    z4 = z2 * z2
    a_j4_x = fJ4 * x * (63.0 * z4 / r2 - 42.0 * z2 + 3.0 * r2)
    a_j4_y = fJ4 * y * (63.0 * z4 / r2 - 42.0 * z2 + 3.0 * r2)
    a_j4_z = fJ4 * z * (63.0 * z4 / r2 - 70.0 * z2 + 15.0 * r2)

    a_total[0] += a_j2_x + a_j3_x + a_j4_x
    a_total[1] += a_j2_y + a_j3_y + a_j4_y
    a_total[2] += a_j2_z + a_j3_z + a_j4_z

    # --- J5 Perturbation --- [FM-1/FM-6 Fix — Finding #4/#21]
    # Ref: Vallado §8.7.2; EGM96 coefficient (Lemoine et al. 1998).
    # Critical for MEO/GEO/HEO long-horizon secular nodal precession accuracy.
    r11 = r9 * r2
    z5 = z4 * z
    fJ5 = -(15.0 / 8.0) * J5 * mu * Re**5 / r11
    a_j5_x = fJ5 * x * (21.0 * z4 / r2 - 14.0 * z2 + r2 / 3.0)
    a_j5_y = fJ5 * y * (21.0 * z4 / r2 - 14.0 * z2 + r2 / 3.0)
    a_j5_z = fJ5 * (21.0 * z5 / r2 - 10.5 * z2 * z + 2.5 * r2 * z)

    # --- J6 Perturbation --- [FM-1/FM-6 Fix — Finding #4/#21]
    # Ref: EGM96 coefficient (Lemoine et al. 1998).
    r13 = r11 * r2
    z6 = z4 * z2
    fJ6 = -(1.0 / 16.0) * J6 * mu * Re**6 / r13
    a_j6_x = fJ6 * x * (231.0 * z6 / r2 - 315.0 * z4 + 105.0 * z2 * r2 - 5.0 * r2 * r2)
    a_j6_y = fJ6 * y * (231.0 * z6 / r2 - 315.0 * z4 + 105.0 * z2 * r2 - 5.0 * r2 * r2)
    a_j6_z = fJ6 * (231.0 * z6 * z / r2 - 378.0 * z4 * z + 189.0 * z5 - 70.0 * z4 * z + 15.0 * z * r2 * r2)

    a_total[0] += a_j5_x + a_j6_x
    a_total[1] += a_j5_y + a_j6_y
    a_total[2] += a_j5_z + a_j6_z

    # --- Atmospheric Drag [HIGH-03 fix] ---
    # When hf_atmosphere=True, call the canonical NRLMSISE-00 Numba kernel.
    # Otherwise fall back to Strategy-A exponential profile (faster, suitable for
    # low-fidelity screening).  Previously hf_atmosphere was accepted by the
    # derivative wrapper but silently ignored here.
    if use_drag and alt_km < 1500.0:
        if hf_atmosphere:
            rho_instant = _nrlmsise00_density_njit(alt_km, f107_obs, f107_adj, ap_daily)
        elif drag_rho > 0.0:
            rho_instant = drag_rho * math.exp(-(alt_km - drag_ref_alt_km) / drag_H_km)
        else:
            rho_instant = 0.0

        if rho_instant > 0.0:
            # OMEGA_EARTH inlined above; guarded by assert in constants.py
            omega_e = np.array([0.0, 0.0, OMEGA_EARTH])
            v_rel = v - np.cross(omega_e, r)
            v_rel_mag = np.linalg.norm(v_rel)
            if v_rel_mag > 1e-10:
                Bc = drag_cd * drag_area_m2 / drag_mass_kg
                # 1e-6 (m² → km²) * 1e9 (kg/m³ → kg/km³) = 1e3
                a_drag = -0.5 * rho_instant * 1e3 * Bc * v_rel_mag * v_rel
                a_total += a_drag

    # --- Third-Body (Sun & Moon) via Chebyshev Splines ---
    if include_third_body:
        # AUDIT-D-01 Fix: Piecewise 7-day spline evaluation instead of monolithic
        idx = int((t_jd - t_jd0) / 7.0)
        num_pieces = sun_coeffs.shape[0]
        if idx >= num_pieces:
            idx = num_pieces - 1
        elif idx < 0:
            idx = 0

        piece_t_jd0 = t_jd0 + idx * 7.0
        piece_duration = 7.0

        t_norm = 2.0 * (t_jd - piece_t_jd0) / piece_duration - 1.0
        # Clamp between -1 and 1 just in case of precision errors at edges
        if t_norm < -1.0:
            t_norm = -1.0
        if t_norm > 1.0:
            t_norm = 1.0

        sun_pos = _eval_cheb_3d_njit(t_norm, sun_coeffs[idx])
        moon_pos = _eval_cheb_3d_njit(t_norm, moon_coeffs[idx])

        # Sun Gravity
        d_sun = sun_pos - r
        d_mag_sun = np.linalg.norm(d_sun)
        r_mag_sun = np.linalg.norm(sun_pos)
        if d_mag_sun > 1.0 and r_mag_sun > 1.0:
            a_total += SUN_MU * (
                d_sun / (d_mag_sun * d_mag_sun * d_mag_sun)
                - sun_pos / (r_mag_sun * r_mag_sun * r_mag_sun)
            )

        # Moon Gravity
        d_moon = moon_pos - r
        d_mag_moon = np.linalg.norm(d_moon)
        r_mag_moon = np.linalg.norm(moon_pos)
        if d_mag_moon > 1.0 and r_mag_moon > 1.0:
            a_total += MOON_MU * (
                d_moon / (d_mag_moon * d_mag_moon * d_mag_moon)
                - moon_pos / (r_mag_moon * r_mag_moon * r_mag_moon)
            )

        if use_srp and drag_area_m2 > 0.0 and drag_mass_kg > 0.0:
            d_ss = r - sun_pos
            d_mag_ss = np.linalg.norm(d_ss)
            if d_mag_ss > 1.0:
                nu = 1.0
                if srp_use_shadow:
                    # High-fidelity conical shadow (PHY-D)
                    nu = _srp_illumination_factor_njit(r, sun_pos, Re, 695700.0)
                P0 = 4.56e-6
                AU = 149597870.7
                scale = (AU / d_mag_ss) * (AU / d_mag_ss)
                amag = P0 * scale * srp_cr * (drag_area_m2 / drag_mass_kg) / 1000.0
                a_total += (nu * amag) * (d_ss / d_mag_ss)

    return a_total  # type: ignore[no-any-return]


@njit(fastmath=True, cache=True)
def _coast_derivative_njit(
    t_sec: float,
    y: np.ndarray,
    t_jd0: float,
    use_drag: bool,
    drag_cd: float,
    drag_area_m2: float,
    drag_mass_kg: float,
    drag_rho: float,
    drag_H_km: float,
    drag_ref_alt_km: float,
    f107_obs: float,
    f107_adj: float,
    ap_daily: float,
    hf_atmosphere: bool,
    include_third_body: bool,
    global_t_jd0: float,
    duration_d: float,
    sun_coeffs: np.ndarray,
    moon_coeffs: np.ndarray,
    use_srp: bool,
    srp_cr: float,
    srp_use_shadow: bool,
) -> np.ndarray:
    """State derivative for unpowered (coast) arcs using Numba."""
    r = y[:3]
    v = y[3:6]
    t_jd = t_jd0 + t_sec / SECONDS_PER_DAY
    a = _acceleration_njit(
        t_jd,
        r,
        v,
        use_drag,
        drag_cd,
        drag_area_m2,
        drag_mass_kg,
        drag_rho,
        drag_H_km,
        drag_ref_alt_km,
        f107_obs,
        f107_adj,
        ap_daily,
        hf_atmosphere,
        include_third_body,
        global_t_jd0,
        duration_d,
        sun_coeffs,
        moon_coeffs,
        use_srp,
        srp_cr,
        srp_use_shadow,
    )

    # 42 components = 6 (r, v) + 36 (Phi)
    if len(y) == 42:
        Phi = y[6:].reshape((6, 6))
        F = _compute_force_jacobian(r, v, EARTH_MU_KM3_S2)
        Phi_dot = np.dot(F, Phi)

        dy = np.empty(42)
        dy[0:3] = v
        dy[3:6] = a
        dy[6:] = Phi_dot.flatten()
        return dy

    dy = np.empty(6)
    dy[0:3] = v
    dy[3:6] = a

    return dy


@njit(fastmath=True, cache=True)
def _powered_derivative_njit(
    t_sec: float,
    y: np.ndarray,
    t_jd0: float,
    use_drag: bool,
    drag_cd: float,
    drag_area_m2: float,
    drag_mass_kg: float,
    drag_rho: float,
    drag_H_km: float,
    drag_ref_alt_km: float,
    f107_obs: float,
    f107_adj: float,
    ap_daily: float,
    hf_atmosphere: bool,
    include_third_body: bool,
    global_t_jd0: float,
    duration_d: float,
    sun_coeffs: np.ndarray,
    moon_coeffs: np.ndarray,
    use_srp: bool,
    srp_cr: float,
    srp_use_shadow: bool,
    burn_thrust_N: float,
    burn_isp_s: float,
    burn_dir: np.ndarray,
    burn_frame_idx: int,
) -> np.ndarray:
    """State derivative for powered (thrusting) arcs using Numba."""
    r = y[:3]
    v = y[3:6]
    m = y[6]
    t_jd = t_jd0 + t_sec / SECONDS_PER_DAY

    # PHY-03 Fix (Numba path): Use instantaneous mass from state vector y[6] for
    # ballistic coefficient instead of pre-burn drag_mass_kg.
    # Clamp at 1g to avoid division by zero in degenerate edge cases.
    instantaneous_mass_kg = m if m > 1e-3 else 1e-3

    a_grav = _acceleration_njit(
        t_jd,
        r,
        v,
        use_drag,
        drag_cd,
        drag_area_m2,
        instantaneous_mass_kg,
        drag_rho,
        drag_H_km,
        drag_ref_alt_km,
        f107_obs,
        f107_adj,
        ap_daily,
        hf_atmosphere,
        include_third_body,
        global_t_jd0,
        duration_d,
        sun_coeffs,
        moon_coeffs,
        use_srp,
        srp_cr,
        srp_use_shadow,
    )

    max(1e-12, np.linalg.norm(r))
    max(1e-12, np.linalg.norm(v))
    thrust_a_mag = (burn_thrust_N / 1000.0) / m

    if burn_frame_idx == 0:  # VNB
        rot_matrix = _build_vnb_matrix_njit(r, v)
    else:  # RTN
        rot_matrix = _build_rtn_matrix_njit(r, v)

    a_thrust = rot_matrix @ (thrust_a_mag * burn_dir)
    a_total = a_grav + a_thrust

    # Mass flow rate
    g0 = 9.80665e-3  # km/s^2
    dm_dt = -(burn_thrust_N / 1000.0) / (burn_isp_s * g0)

    # 43 components = 7 (r, v, m) + 36 (Phi)
    if len(y) == 43:
        Phi = y[7:].reshape((6, 6))
        # Partials of acceleration w.r.t r, v
        F = _compute_force_jacobian(r, v, EARTH_MU_KM3_S2)
        Phi_dot = np.dot(F, Phi)

        dy = np.empty(43)
        dy[0:3] = v
        dy[3:6] = a_total
        dy[6] = dm_dt
        dy[7:] = Phi_dot.flatten()
        return dy

    dy = np.empty(7)
    dy[0:3] = v
    dy[3:6] = a_total
    dy[6] = dm_dt

    return dy


def _compute_scale_height(f107_obs: float, f107_adj: float, ap_daily: float) -> float:
    """Compute empirical atmospheric scale height (km) at 400 km reference altitude."""
    # Use the local njit kernel to avoid circular imports from data_pipeline
    rho0 = _nrlmsise00_density_njit(400.0, f107_obs, f107_adj, ap_daily)
    rho1 = _nrlmsise00_density_njit(401.0, f107_obs, f107_adj, ap_daily)

    if rho0 > 1e-20 and rho1 > 1e-20:
        H_km = -1.0 / (math.log(rho1 / rho0))
        return max(20.0, min(H_km, 150.0))
    return 58.5


def _compute_planetary_splines(
    t_jd0: float, duration_s: float, use_de: bool
) -> tuple[np.ndarray, np.ndarray, float]:
    duration_d = duration_s / 86400.0
    # AUDIT-D-01 Fix: Avoid Chebyshev degree cliffs by chunking long propagations
    # into piecewise 7-day splines with fixed degree 35 (5 nodes/day), preventing
    # M-level node scaling on multi-year orbits while fully satisfying lunar freq.
    deg = 35
    num_pieces = max(1, int(np.ceil(duration_d / 7.0)))

    nodes_norm = np.cos(np.pi * (2 * np.arange(deg + 1) + 1) / (2 * (deg + 1)))

    sun_coeffs = np.zeros((num_pieces, deg + 1, 3))
    moon_coeffs = np.zeros((num_pieces, deg + 1, 3))

    # TEME-frame samples so the Numba kernel matches the SGP4 propagator frame.
    sun_fn = _sun_position_de if use_de else _sun_position_approx
    moon_fn = _moon_position_de if use_de else _moon_position_approx

    for i in range(num_pieces):
        piece_t_jd0 = t_jd0 + i * 7.0
        piece_dur = 7.0
        t_nodes = piece_t_jd0 + 0.5 * piece_dur * (nodes_norm + 1.0)

        sun_pos = np.zeros((deg + 1, 3))
        moon_pos = np.zeros((deg + 1, 3))
        for j, t in enumerate(t_nodes):
            sun_pos[j] = sun_fn(t)
            moon_pos[j] = moon_fn(t)

        for dim in range(3):
            sun_coeffs[i, :, dim] = cheb.chebfit(nodes_norm, sun_pos[:, dim], deg)
            moon_coeffs[i, :, dim] = cheb.chebfit(nodes_norm, moon_pos[:, dim], deg)

    return sun_coeffs, moon_coeffs, duration_d


# ---------------------------------------------------------------------------
# Segmented Cowell Integrator
# ---------------------------------------------------------------------------


@dataclass
class _PropagatorContext:
    t_jd0: float
    duration_d: float
    dt_out: float
    include_third_body: bool
    use_drag: bool
    use_empirical_drag: bool
    hf_atmosphere: bool
    use_srp: bool
    srp_cr: float
    srp_use_shadow: bool
    include_stm: bool
    rtol: Optional[float]
    atol: Any
    coast_rtol: float
    coast_atol: float
    powered_rtol: float
    powered_atol: float
    sun_coeffs: np.ndarray
    moon_coeffs: np.ndarray
    drag_cd: float
    drag_area_m2: float
    current_P0: np.ndarray

def _prepare_maneuvers(maneuvers: list[FiniteBurn], mass_kg: Optional[float]) -> tuple[list[FiniteBurn], float]:
    from astra.maneuver import validate_burn, validate_burn_sequence
    from astra import config
    if mass_kg is None:
        if config.ASTRA_STRICT_MODE:
            from astra.errors import ManeuverError
            raise ManeuverError(
                "[ASTRA STRICT] Spacecraft mass_kg is required for powered maneuver propagation. "
                "Set NumericalState.mass_kg or disable strict mode via astra.config.ASTRA_STRICT_MODE=False."
            )
        logger.warning(
            "Maneuvers specified but initial mass_kg is None. "
            "Defaulting to 1000.0 kg — drag and Tsiolkovsky accuracy are compromised. "
            "Provide mass_kg in NumericalState for physical accuracy."
        )
        mass_kg = 1000.0

    burns = sorted(maneuvers, key=lambda b: b.epoch_ignition_jd)
    validate_burn_sequence(burns)
    running_mass = mass_kg
    for burn in burns:
        validate_burn(burn, running_mass)
        running_mass -= burn.mass_flow_rate_kg_s * burn.duration_s
    return burns, mass_kg

def _prepare_drag_environment(t_jd: float, drag_ref_alt_km: float, use_drag: bool, use_empirical_drag: bool) -> tuple[float, float, float, float, float]:
    f107_obs, f107_adj, ap_daily = 150.0, 150.0, 15.0
    drag_rho = 0.0
    drag_H_km = 58.515
    if use_drag:
        if use_empirical_drag:
            try:
                from astra.data_pipeline import get_space_weather, atmospheric_density_empirical
                f107_obs, f107_adj, ap_daily = get_space_weather(t_jd)
                drag_rho = atmospheric_density_empirical(drag_ref_alt_km, f107_obs, f107_adj, ap_daily)
                drag_H_km = _compute_scale_height(f107_obs, f107_adj, ap_daily)
            except Exception:
                drag_rho = 0.0
        if drag_rho == 0.0:
            from astra.constants import DRAG_REF_DENSITY_KG_M3, DRAG_SCALE_HEIGHT_KM
            drag_rho = DRAG_REF_DENSITY_KG_M3
            drag_H_km = DRAG_SCALE_HEIGHT_KM
            drag_ref_alt_km = 400.0
    return drag_rho, drag_H_km, f107_obs, f107_adj, ap_daily

def _integrate_segment(
    seg_start_s: float, seg_end_s: float, active_burn: Optional[FiniteBurn],
    current_r: np.ndarray, current_v: np.ndarray, current_mass: Optional[float],
    current_phi_flat: np.ndarray, ctx: _PropagatorContext,
    drag_mass_kg: float, drag_rho: float, drag_H_km: float, drag_ref_alt_km: float,
    f107_obs: float, f107_adj: float, ap_daily: float
) -> tuple[bool, str, list[NumericalState], np.ndarray, np.ndarray, float, np.ndarray, float]:
    
    seg_duration = seg_end_s - seg_start_s
    global_t_start = seg_start_s
    global_t_end = seg_end_s

    t_out = []
    first_grid = math.ceil(global_t_start / ctx.dt_out) * ctx.dt_out
    t = first_grid
    while t <= global_t_end + 1e-9:
        if t >= global_t_start - 1e-9:
            t_out.append(t - global_t_start)
        t += ctx.dt_out

    if not t_out or t_out[0] > 1e-9:
        t_out.insert(0, 0.0)
    if t_out[-1] < seg_duration - 1e-9:
        t_out.append(seg_duration)

    t_eval = np.array(sorted(set(t_out)))
    t_eval = t_eval[t_eval <= seg_duration + 1e-9]
    
    segment_states = []

    if active_burn is not None and current_mass is not None:
        y0 = np.concatenate([current_r, current_v, [current_mass]])
        if ctx.include_stm:
            y0 = np.concatenate([y0, current_phi_flat])

        b_thrust = active_burn.thrust_N
        b_isp = active_burn.isp_s
        b_dir = np.array(active_burn.direction)
        b_idx = 0 if active_burn.frame.value == "VNB" else 1

        def deriv(t_sec: float, y: np.ndarray) -> np.ndarray:
            return _powered_derivative_njit(  # type: ignore[no-any-return]
                t_sec, y, ctx.t_jd0 + seg_start_s / SECONDS_PER_DAY,
                ctx.use_drag, ctx.drag_cd, ctx.drag_area_m2, drag_mass_kg,
                drag_rho, drag_H_km, drag_ref_alt_km, f107_obs, f107_adj, ap_daily,
                ctx.hf_atmosphere, ctx.include_third_body, ctx.t_jd0, ctx.duration_d,
                ctx.sun_coeffs, ctx.moon_coeffs, ctx.use_srp, ctx.srp_cr, ctx.srp_use_shadow,
                b_thrust, b_isp, b_dir, b_idx
            )
            
        atol_vec = np.array([1e-11]*3 + [1e-13]*3 + [1e-5])
        if ctx.include_stm:
            atol_vec = np.concatenate([atol_vec, np.full(36, 1e-12)])
            
        sol = solve_ivp(
            deriv, t_span=(0.0, seg_duration), y0=y0, method="DOP853", t_eval=t_eval,
            rtol=ctx.rtol if ctx.rtol is not None else ctx.powered_rtol,
            atol=ctx.atol if ctx.atol is not None else atol_vec,
        )
        is_powered = True
    else:
        y0 = np.concatenate([current_r, current_v])
        if ctx.include_stm:
            y0 = np.concatenate([y0, current_phi_flat])

        def deriv(t_sec: float, y: np.ndarray) -> np.ndarray:
            return _coast_derivative_njit(  # type: ignore[no-any-return]
                t_sec, y, ctx.t_jd0 + seg_start_s / SECONDS_PER_DAY,
                ctx.use_drag, ctx.drag_cd, ctx.drag_area_m2, drag_mass_kg,
                drag_rho, drag_H_km, drag_ref_alt_km, f107_obs, f107_adj, ap_daily,
                ctx.hf_atmosphere, ctx.include_third_body, ctx.t_jd0, ctx.duration_d,
                ctx.sun_coeffs, ctx.moon_coeffs, ctx.use_srp, ctx.srp_cr, ctx.srp_use_shadow
            )
            
        atol_vec = np.array([ctx.coast_atol] * 6)
        if ctx.include_stm:
            atol_vec = np.concatenate([atol_vec, np.full(36, 1e-12)])
            
        sol = solve_ivp(
            deriv, t_span=(0.0, seg_duration), y0=y0, method="DOP853", t_eval=t_eval,
            rtol=ctx.rtol if ctx.rtol is not None else ctx.coast_rtol,
            atol=ctx.atol if ctx.atol is not None else atol_vec,
        )
        is_powered = False

    if sol.success and hasattr(sol, "y") and not np.all(np.isfinite(sol.y)):
        sol.success = False
        sol.message = "Numerical instability detected (NaN/Inf in state)"

    if not sol.success:
        return False, sol.message, [], current_r, current_v, current_mass or 0.0, current_phi_flat, drag_mass_kg

    for i in range(len(sol.t)):
        _cov_out = None
        if ctx.include_stm:
            phi_i = sol.y[7:43, i].reshape((6, 6)) if is_powered else sol.y[6:42, i].reshape((6, 6))
            phi_rr = phi_i[:3, :3]
            _cov_out = phi_rr @ ctx.current_P0 @ phi_rr.T
            
        segment_states.append(
            NumericalState(
                t_jd=ctx.t_jd0 + (seg_start_s + sol.t[i]) / SECONDS_PER_DAY,
                position_km=sol.y[:3, i].copy(),
                velocity_km_s=sol.y[3:6, i].copy(),
                mass_kg=float(sol.y[6, i]) if is_powered else (current_mass or 0.0),
                covariance_km2=_cov_out,
            )
        )

    r_out = sol.y[:3, -1].copy()
    v_out = sol.y[3:6, -1].copy()
    m_out = float(sol.y[6, -1]) if is_powered else (current_mass or 0.0)
    phi_out = sol.y[7:43, -1].copy() if is_powered and ctx.include_stm else (sol.y[6:42, -1].copy() if ctx.include_stm else current_phi_flat)
    dm_out = m_out if is_powered else drag_mass_kg
    
    return True, "", segment_states, r_out, v_out, m_out, phi_out, dm_out

def propagate_cowell(
    state0: NumericalState,
    duration_s: float,
    dt_out: float = 60.0,
    drag_config: Optional[DragConfig] = None,
    include_third_body: bool = True,
    rtol: Optional[float] = None,
    atol: Optional[float] = None,
    maneuvers: Optional[list[FiniteBurn]] = None,
    use_de: bool = True,
    use_empirical_drag: bool = True,
    include_stm: bool = False,
    snc_config: Optional[SNCConfig] = None,
    coast_rtol: float = 1e-8,
    coast_atol: float = 1e-8,
    powered_rtol: float = 1e-12,
    powered_atol: float = 1e-12,
) -> list[NumericalState]:
    """Propagate an orbit using segmented Cowell's method with DOP853.

    This is a mission-operations–grade numerical propagator that
    automatically segments the integration timeline at engine
    ignition/cutoff boundaries.  Each segment uses the appropriate
    derivative function:

        - **Coast segments**: 6-DOF  [r, v]  — gravitational + drag.
        - **Powered segments**: 7-DOF  [r, v, m]  — adds thrust and
          Tsiolkovsky mass depletion.

    The segmented approach ensures that ``solve_ivp`` never steps across
    a force-model discontinuity, eliminating truncation error at burn
    edges.

    Known Limitations:
        - Geopotential truncated at J4. J5/J6 contribute ~0.1 km/day at GEO.
          For ultra-high-fidelity MEO/GEO, use a gravity model with more harmonics.
        - Atmospheric scale height uses single-layer exponential profile;
          multi-altitude molecular mass variation is a third-order effect.
        - Powered-arc default mass absolute tolerance is 10⁻⁵ kg (0.01 g), which
          is loose for micro-thrusters on small spacecraft; pass ``atol`` for
          tighter mass conservation.
        - SRP penumbra uses a planar circle-circle intersection formula (angular
          quantities as proxies for linear radii). Valid to ~few-% accuracy in LEO;
          see ``_srp_illumination_factor_njit`` docstring for details (MATH-05).

    Args:
        state0: Initial state (position + velocity + optional mass).
        duration_s: Total propagation duration in seconds.
        dt_out: Output time step in seconds (default 60 s).
        drag_config: Optional atmospheric drag parameters.
        include_third_body: Include Sun/Moon gravity.
        rtol: Top-level relative tolerance override (WARNING: Silently discards any
            fine-tuned ``coast_rtol`` and ``powered_rtol`` values).
        atol: Top-level absolute tolerance override (WARNING: Silently discards any
            fine-tuned ``coast_atol`` and ``powered_atol`` values).
        maneuvers: Optional list of ``FiniteBurn`` definitions.
            Burns must not overlap in time.
        use_de: Use JPL DE421 for Sun/Moon (True) or analytical (False).
        use_empirical_drag: Use F10.7/Ap drag model (True) or static (False).
        coast_rtol: Relative tolerance for coast arcs (default 1e-8).
        coast_atol: Absolute tolerance for coast arcs (default 1e-8).
            For high-accuracy conjunction analysis, tighten to 1e-12.
        powered_rtol: Relative tolerance for powered arcs (default 1e-12).
        powered_atol: Absolute tolerance for powered arcs (default 1e-12).

    Returns:
        List of ``NumericalState`` objects at each output time step.
        If maneuvers are present, each state includes the current mass.
    """
    t_jd0 = state0.t_jd
    if duration_s > 72 * 3600:
        logger.warning(
            f"Long-duration propagation ({duration_s/3600:.1f}h > 72h) detected. "
            "SGP4/TLE accuracy degrades significantly beyond 3 days; result should be "
            "treated as coarse screening only."
        )

    burns: list[FiniteBurn] = []
    mass_kg = state0.mass_kg
    if maneuvers:
        burns, mass_kg = _prepare_maneuvers(maneuvers, mass_kg)

    segments = _build_segments(t_jd0, duration_s, burns)

    if include_third_body:
        sun_coeffs, moon_coeffs, duration_d = _compute_planetary_splines(t_jd0, duration_s, use_de)
    else:
        sun_coeffs = np.zeros((1, 1, 3))
        moon_coeffs = np.zeros((1, 1, 3))
        duration_d = duration_s / 86400.0

    use_drag = drag_config is not None
    drag_cd = drag_config.cd if drag_config else 0.0
    drag_area_m2 = drag_config.area_m2 if drag_config else 0.0
    drag_mass_kg = drag_config.mass_kg if drag_config else 1.0

    hf_atmosphere = bool(drag_config is not None and use_empirical_drag and drag_config.model.upper() == "NRLMSISE00")
    use_srp = bool(drag_config is not None and getattr(drag_config, "include_srp", True) and include_third_body)
    srp_cr = float(drag_config.cr) if drag_config is not None else 1.5
    srp_use_shadow = bool(drag_config is not None and getattr(drag_config, "srp_conical_shadow", True))

    _r0_mag = float(np.linalg.norm(state0.position_km))
    drag_ref_alt_km = max(0.0, _r0_mag - EARTH_EQUATORIAL_RADIUS_KM)

    drag_rho, drag_H_km, f107_obs, f107_adj, ap_daily = _prepare_drag_environment(t_jd0, drag_ref_alt_km, use_drag, use_empirical_drag)

    _initial_cov = getattr(state0, "covariance_km2", None)
    current_P0 = _initial_cov.copy() if _initial_cov is not None else np.eye(3, dtype=np.float64) * 1e-6

    ctx = _PropagatorContext(
        t_jd0=t_jd0, duration_d=duration_d, dt_out=dt_out, include_third_body=include_third_body,
        use_drag=use_drag, use_empirical_drag=use_empirical_drag, hf_atmosphere=hf_atmosphere,
        use_srp=use_srp, srp_cr=srp_cr, srp_use_shadow=srp_use_shadow, include_stm=include_stm,
        rtol=rtol, atol=atol, coast_rtol=coast_rtol, coast_atol=coast_atol,
        powered_rtol=powered_rtol, powered_atol=powered_atol,
        sun_coeffs=sun_coeffs, moon_coeffs=moon_coeffs,
        drag_cd=drag_cd, drag_area_m2=drag_area_m2, current_P0=current_P0
    )

    all_states: list[NumericalState] = []
    current_r = state0.position_km.copy()
    current_v = state0.velocity_km_s.copy()
    current_mass = mass_kg
    current_phi_flat = np.eye(6, dtype=np.float64).flatten()

    for seg_start_s, seg_end_s, active_burn in segments:
        seg_duration = seg_end_s - seg_start_s
        if seg_duration < 1e-9:
            continue

        if use_drag and use_empirical_drag:
            drag_rho, drag_H_km, f107_obs, f107_adj, ap_daily = _prepare_drag_environment(
                t_jd0 + seg_start_s / 86400.0, drag_ref_alt_km, use_drag, use_empirical_drag
            )

        success, msg, seg_states, current_r, current_v, current_mass, current_phi_flat, drag_mass_kg = _integrate_segment(
            seg_start_s, seg_end_s, active_burn,
            current_r, current_v, current_mass,
            current_phi_flat, ctx,
            drag_mass_kg, drag_rho, drag_H_km, drag_ref_alt_km,
            f107_obs, f107_adj, ap_daily
        )

        if not success:
            logger.error(f"Integration failed: {msg}")
            from astra import config
            if config.ASTRA_STRICT_MODE:
                from astra.errors import PropagationError
                raise PropagationError(f"Cowell integration failed: {msg}", t_jd=t_jd0 + seg_start_s / SECONDS_PER_DAY)
            break
            
        all_states.extend(seg_states)

    if all_states:
        deduped = [all_states[0]]
        for s in all_states[1:]:
            if abs(s.t_jd - deduped[-1].t_jd) > 1e-12:
                deduped.append(s)
        all_states = deduped

    logger.info(f"Propagation complete: {len(all_states)} states generated.")
    return all_states

# ---------------------------------------------------------------------------
# Timeline Segmentation
# ---------------------------------------------------------------------------


def _build_segments(
    t_jd0: float,
    duration_s: float,
    burns: list[FiniteBurn],
) -> list[tuple[float, float, Optional[FiniteBurn]]]:
    """Build an ordered list of (t_start_s, t_end_s, burn_or_None) segments.

    Slices the total propagation window so that every burn arc and
    every coast arc is its own contiguous segment.  The integrator is
    re-initialised at each boundary.

    Args:
        t_jd0: Epoch of propagation start (Julian Date).
        duration_s: Total propagation time in seconds.
        burns: Sorted list of FiniteBurn objects.

    Returns:
        List of (start_s, end_s, burn) tuples.  burn is None for coast.
    """
    segments: list[tuple[float, float, Optional[FiniteBurn]]] = []
    cursor_s = 0.0
    end_s = duration_s

    # AUDIT-B-03 Fix: Robust overlap detection (SE-G) — build a set of burn
    # indices to skip so that the processing loop below actually omits them,
    # rather than just warning and then still scheduling the overlapping arc.
    # In STRICT_MODE we raise immediately; in Relaxed mode we warn and skip.
    import logging as _log_mod

    _seg_log = _log_mod.getLogger(__name__)
    skipped_indices: set[int] = set()
    for i in range(len(burns) - 1):
        b1, b2 = burns[i], burns[i + 1]
        if b2.epoch_ignition_jd < b1.epoch_cutoff_jd - 1e-12:
            from astra import config
            from astra.errors import ManeuverError

            msg = (
                f"Temporal overlap detected: Burn[{i+1}] ignition "
                f"{b2.epoch_ignition_jd:.8f} JD is before Burn[{i}] cutoff "
                f"{b1.epoch_cutoff_jd:.8f} JD."
            )
            if config.ASTRA_STRICT_MODE:
                raise ManeuverError(msg)
            else:
                _seg_log.warning(
                    "%s Burn[%d] will be skipped to avoid dual-thrust arc.", msg, i + 1
                )
                skipped_indices.add(i + 1)  # skip the later (overlapping) burn

    for idx, burn in enumerate(burns):
        if idx in skipped_indices:
            continue  # AUDIT-B-03: actually omit the overlapping burn

        # Convert burn epochs to seconds relative to t_jd0
        ign_s = (burn.epoch_ignition_jd - t_jd0) * 86400.0
        cut_s = (burn.epoch_cutoff_jd - t_jd0) * 86400.0

        # Clamp to propagation window
        ign_s = max(ign_s, 0.0)
        cut_s = min(cut_s, end_s)

        if ign_s >= end_s or cut_s <= 0.0:
            continue  # Burn is entirely outside the window

        # Coast before this burn
        if ign_s > cursor_s + 1e-9:
            segments.append((cursor_s, ign_s, None))

        # Powered arc
        if cut_s > ign_s + 1e-9:
            segments.append((ign_s, cut_s, burn))

        cursor_s = cut_s

    # Final coast after last burn
    if cursor_s < end_s - 1e-9:
        segments.append((cursor_s, end_s, None))

    return segments


# ---------------------------------------------------------------------------
# Batch High-Fidelity Propagation  [FM-4 Fix — Finding #16]
# ---------------------------------------------------------------------------


def propagate_cowell_batch(
    states: "dict[str, NumericalState]",
    duration_s: float,
    dt_out: float = 60.0,
    drag_config: Optional["DragConfig"] = None,
    maneuvers: Optional["dict[str, list[FiniteBurn]]"] = None,
    include_third_body: bool = False,
    max_workers: Optional[int] = None,
) -> dict[str, list["NumericalState"]]:
    """Propagate multiple satellites concurrently with the high-fidelity Cowell integrator.

    This is a production batch wrapper around :func:`propagate_cowell` that
    parallelises propagation over ``N`` initial states using a
    ``ThreadPoolExecutor``.  It eliminates the ad-hoc ``ThreadPoolExecutor``
    pattern that users previously had to replicate (and which already existed
    *inline* inside :func:`find_conjunctions`).

    The input uses ``dict[str, NumericalState]`` (satellite-ID → state) rather
    than a list, which is consistent with the ``TrajectoryMap`` convention used
    throughout ASTRA (e.g. ``find_conjunctions``).  This also makes the key
    explicit and avoids any ambiguity when the caller needs to correlate inputs
    and outputs.

    Args:
        states: ``dict[satellite_id, NumericalState]`` — one entry per satellite.
            The key (string) is used as the map key in the returned dict.
        duration_s: Total propagation duration in seconds (identical for all).
        dt_out: Output step size in seconds (default 60 s).
        drag_config: Optional :class:`DragConfig`; applied uniformly to all
            satellites unless overridden per-satellite in the future.
        maneuvers: Optional ``dict[satellite_id, list[FiniteBurn]]``.
            Missing keys default to an empty burn sequence.
        hf_atmosphere: Enable NRLMSISE-00 atmospheric model for all satellites.
        include_third_body: Enable Sun/Moon third-body gravity.
        use_srp: Enable Solar Radiation Pressure perturbation.
        max_workers: Thread pool size.  Defaults to ``min(32, len(states))``.

    Returns:
        ``dict[satellite_id, list[NumericalState]]`` — propagated state histories.
        Satellites that fail propagation are absent from the dict; their error
        is logged at WARNING level.

    Raises:
        ValueError: If ``states`` is empty or ``duration_s`` <= 0.

    Example::

        import astra, numpy as np
        s1 = astra.NumericalState(t_jd=2460000.5,
                                  position_km=np.array([6778.0, 0.0, 0.0]),
                                  velocity_km_s=np.array([0.0, 7.668, 0.0]))
        s2 = astra.NumericalState(t_jd=2460000.5,
                                  position_km=np.array([6788.0, 0.0, 0.0]),
                                  velocity_km_s=np.array([0.0, 7.658, 0.0]))
        results = astra.propagate_cowell_batch(
            {"ISS": s1, "DEBRIS-A": s2}, duration_s=3600.0
        )
        for sat_id, traj in results.items():
            print(sat_id, len(traj), "steps")
    """
    import logging
    from concurrent.futures import ThreadPoolExecutor, as_completed

    _log = logging.getLogger(__name__)

    if not states:
        raise ValueError("propagate_cowell_batch: states dict is empty.")
    if duration_s <= 0.0:
        raise ValueError(
            f"propagate_cowell_batch: duration_s must be positive, got {duration_s}."
        )

    n = len(states)
    effective_workers = max_workers if max_workers is not None else min(32, n)
    burns_map = maneuvers or {}

    results: dict[str, list[NumericalState]] = {}

    def _worker(sat_id: str, state: "NumericalState") -> tuple[str, list["NumericalState"]]:
        """Single-satellite propagation task for the thread pool."""
        # propagate_cowell does not accept hf_atmosphere / use_srp kwargs directly;
        # those are handled via DragConfig.model and the drag_config's srp_cr field.
        # use_empirical_drag=True enables NRLMSISE-00 when DragConfig.model='NRLMSISE00'.
        traj = propagate_cowell(
            state,
            duration_s=duration_s,
            dt_out=dt_out,
            drag_config=drag_config,
            maneuvers=burns_map.get(sat_id, []),
            include_third_body=include_third_body,
            use_empirical_drag=True,
        )
        return sat_id, traj

    future_to_id: dict = {}
    with ThreadPoolExecutor(max_workers=effective_workers) as executor:
        for sat_id, state in states.items():
            fut = executor.submit(_worker, sat_id, state)
            future_to_id[fut] = sat_id

        for fut in as_completed(future_to_id):
            sat_id = future_to_id[fut]
            try:
                sid, traj = fut.result()
                results[sid] = traj
            except Exception as exc:
                _log.warning(
                    "propagate_cowell_batch: propagation failed for %s — %s",
                    sat_id,
                    exc,
                )

    return results
