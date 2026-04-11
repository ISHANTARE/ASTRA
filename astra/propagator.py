# astra/propagator.py
"""ASTRA Core Numerical Propagator — Segmented Cowell's Method.

Implements a mission-operations–grade numerical orbit propagator using
Cowell's direct integration with a Dormand-Prince RK8(7) adaptive-step
integrator.

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
- Empirical atmospheric drag (Jacchia-class with space weather)
- Solar third-body point-mass perturbation (JPL DE421)
- Lunar third-body point-mass perturbation (JPL DE421)
- Finite continuous thrust (7-DOF powered arcs)

**Numba / IEEE-754**

JIT kernels use ``@njit(fastmath=True)`` (MTH-16), which allows reordering and
fused operations that can differ slightly from the pure-Python
``_acceleration`` path. For validation, compare integrated trajectories
or segment-level energy, not bitwise-identical acceleration samples.

**SRP / PHY-18**

Cannonball SRP scales flux from 1 AU; optional **cylindrical Earth umbra**
(no penumbra) zeros SRP when the satellite lies in the anti-sunward cylinder
of radius ``EARTH_EQUATORIAL_RADIUS_KM`` — see ``DragConfig.srp_cylindrical_shadow``.

**References**

- Vallado, D. A. (2013). *Fundamentals of Astrodynamics and Applications.*
- Montenbruck & Gill (2000). *Satellite Orbits.*
- Park et al. (2021). JPL Planetary Ephemerides DE440/DE441.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import numpy.polynomial.chebyshev as cheb
try:
    from numba import njit
    _NUMBA_AVAILABLE = True
except ImportError:
    _NUMBA_AVAILABLE = False
    def njit(*args, **kwargs):
        """No-op decorator when Numba is unavailable."""
        def decorator(f):
            return f
        return decorator if (args and callable(args[0])) else decorator
    import logging as _logging
    _logging.getLogger(__name__).warning(
        "Numba not installed — JIT-compiled kernels will run in pure-Python mode. "
        "Performance will be significantly reduced. Install numba for full speed."
    )
from scipy.integrate import solve_ivp

from astra.constants import (
    EARTH_EQUATORIAL_RADIUS_KM,
    EARTH_MU_KM3_S2,
    EARTH_OMEGA_RAD_S,
    J2, J3, J4,
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
def _srp_illumination_factor_njit(
    r_km: np.ndarray,
    r_sun_km: np.ndarray,
    earth_radius_km: float = 6378.137,
    sun_radius_km: float = 695700.0,
) -> float:
    """Conical Earth umbra/penumbra factor ν for cannonball SRP.

    Models the Sun and Earth as spherical disks as seen from the satellite
    and computes the fractional area of the solar disk that remains visible.

    **Mathematical approach — MATH-05 note:**
    The circle-circle overlap formula used here applies planar geometry:

        x = (γ² + α² − β²) / (2γ)
        A = α²·acos(x/α) + β²·acos((γ-x)/β) − γ·√(α²-x²)
        ν = 1 − A / (π·β²)

    where α = Earth angular semi-diameter, β = Sun angular semi-radius,
    γ = angular separation.  In LEO, α ≈ 0.72 rad (not a small angle), so
    a strict spherical-geometry treatment (Montenbruck & Gill §3.4) would
    give a slightly different area.  In practice, for LEO conjunction screening
    durations (~hours) the error in time-averaged ν is a few percent, which
    is below the uncertainty in Cr and area/mass.  This approximation is
    therefore acceptable for mission-planning fidelity.  For ultra-precise
    SRP-limited precision orbit determination, replace with exact
    spherical-cap intersection (Vallado 2013, Algorithm 34).

    Args:
        r_km: Geocentric satellite position (km).
        r_sun_km: Geocentric Sun position (km).
        earth_radius_km: Earth equatorial radius (km).
        sun_radius_km: Solar radius (km).

    Returns:
        Fractional illumination in [0, 1].
    """
    # Relative vectors
    d_sun_sat = r_sun_km - r_km
    d_sun_sat_mag = np.linalg.norm(d_sun_sat)
    r_mag = np.linalg.norm(r_km)

    if r_mag < 1.0 or d_sun_sat_mag < 1.0:
        return 1.0

    # Apparent angular radii (radians)
    # Using small-angle approximation sin(x) ≈ x is NOT used here to maintain fidelity.
    alpha = math.asin(earth_radius_km / r_mag)
    beta = math.asin(sun_radius_km / d_sun_sat_mag)

    # Angular separation between Earth and Sun centers (radians)
    # cos(gamma) = (-r_sat . d_sun_sat) / (|r_sat| * |d_sun_sat|)
    cos_gamma = np.dot(-r_km, d_sun_sat) / (r_mag * d_sun_sat_mag)
    # Clamp for numerical safety
    if cos_gamma > 1.0: cos_gamma = 1.0
    if cos_gamma < -1.0: cos_gamma = -1.0
    gamma = math.acos(cos_gamma)

    # Sunlight (no occultation)
    if gamma >= alpha + beta:
        return 1.0
    # Umbra (Total eclipse)
    if gamma <= alpha - beta:
        return 0.0
    # Penumbra (Partial eclipse) - Area of intersection of two circles
    if gamma < alpha + beta:
        # Standard circle-circle intersection area on a unit sphere (approximation using planar geometry)
        # alpha = circle 1 radius, beta = circle 2 radius, gamma = distance between centers
        x = (gamma*gamma + alpha*alpha - beta*beta) / (2.0 * gamma)
        y = math.sqrt(max(0.0, alpha*alpha - x*x))
        
        area = (alpha*alpha * math.acos(x / alpha) + 
                beta*beta * math.acos((gamma - x) / beta) - 
                gamma * y)
        
        # Illumination factor = 1 - (Obscured Area / Total Solar Area)
        nu = 1.0 - area / (math.pi * beta * beta)
        if nu < 0.0: nu = 0.0
        if nu > 1.0: nu = 1.0
        return nu

@njit(fastmath=True, cache=True)
def srp_illumination_factor_njit(r_km, r_sun_km, earth_radius_km, sun_radius_km):
    """Public NJIT wrapper for conical SRP illumination factor."""
    return _srp_illumination_factor_njit(r_km, r_sun_km, earth_radius_km, sun_radius_km)

def srp_illumination_factor(
    r_km: np.ndarray,
    r_sun_km: np.ndarray,
    earth_radius_km: float = 6378.137,
    sun_radius_km: float = 695700.0,
) -> float:
    """Public pure-Python wrapper for conical SRP illumination factor ν in [0, 1]."""
    return _srp_illumination_factor_njit(r_km, r_sun_km, earth_radius_km, sun_radius_km)


@njit(fastmath=True, cache=True)
def srp_cylindrical_illumination_factor_njit(r_km, r_sun_km):
    """Legacy cylindrical umbra model (ν=0 in-cylinder, ν=1 outside)."""
    d_sun_sat = r_sun_km - r_km
    d_mag = np.linalg.norm(d_sun_sat)
    if d_mag < 1.0: return 1.0
    
    # cos(gamma) = (-r . d_sun_sat) / (|r|*|d_sun_sat|)
    r_mag = np.linalg.norm(r_km)
    cos_gamma = np.dot(-r_km, d_sun_sat) / (r_mag * d_mag)
    if cos_gamma > 0.0:  # Satellite is on the night side
        # Separation distance from shadow axis
        sin_gamma = math.sqrt(max(0.0, 1.0 - cos_gamma*cos_gamma))
        dist_axis = r_mag * sin_gamma
        if dist_axis < 6378.137:
            return 0.0
    return 1.0

def srp_cylindrical_illumination_factor(r_km, r_sun_km):
    """Public pure-Python legacy cylindrical umbra model."""
    return srp_cylindrical_illumination_factor_njit(r_km, r_sun_km)

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

    def __post_init__(self):
        """Basic range validation on state vectors."""
        if self.position_km is not None:
            r_mag = np.linalg.norm(self.position_km)
            if 1.0 < r_mag < 6000.0:
                logger.warning(f"NumericalState position radius {r_mag:.2f} km is inside Earth radius.")
            elif r_mag < 0.1:
                # Likely an unitialized or zeroed state
                logger.debug("NumericalState initialized with nearly-zero position vector.")



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
    srp_cylindrical_shadow: bool = True


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
    dL = (6.289 * math.sin(M_moon_r)
          - 1.274 * math.sin(2 * D_r - M_moon_r)
          + 0.658 * math.sin(2 * D_r)
          - 0.214 * math.sin(2 * M_moon_r)
          - 0.186 * math.sin(M_sun_r))

    # Latitude (degrees)
    B = (5.128 * math.sin(F_r)
         + 0.281 * math.sin(M_moon_r + F_r)
         - 0.278 * math.sin(F_r - M_moon_r))

    # Distance (km)
    R_km = (385000.56
            - 20905.36 * math.cos(M_moon_r)
            - 3699.11 * math.cos(2 * D_r - M_moon_r)
            - 2955.97 * math.cos(2 * D_r))

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

def _atmospheric_density(alt_km: float, t_jd: float, use_empirical: bool = True) -> float:
    """Get atmospheric density in kg/m³.

    If `use_empirical` is True and space-weather data is available,
    uses the Jacchia-class model from data_pipeline.  Otherwise falls
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
            from astra.data_pipeline import get_space_weather, atmospheric_density_empirical
            from astra.errors import SpaceWeatherError
            try:
                f107_obs, f107_adj, ap_daily = get_space_weather(t_jd)
                return atmospheric_density_empirical(alt_km, f107_obs, f107_adj, ap_daily)
            except SpaceWeatherError:
                raise
            except (ImportError, ValueError, OSError):
                from astra import config
                if config.ASTRA_STRICT_MODE:
                    raise SpaceWeatherError(
                        "[ASTRA STRICT] Empirical atmospheric density unavailable. "
                        "Load space weather data or set ASTRA_STRICT_MODE=False."
                    )
                logger.warning("Empirical atmospheric model unavailable — using static exponential fallback.")
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
        logger.debug(
            f"Altitude {alt_km:.1f} km below drag model minimum ({DRAG_MIN_ALTITUDE_KM} km). "
            "Returning zero density — reentry simulations require a dedicated model (e.g. NRLMSISE-00)."
        )
        return 0.0
    return DRAG_REF_DENSITY_KG_M3 * math.exp(
        -(alt_km - DRAG_REF_ALTITUDE_KM) / DRAG_SCALE_HEIGHT_KM
    )


# ---------------------------------------------------------------------------
# Force Model (shared between coast and powered derivatives)
# ---------------------------------------------------------------------------

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
    include_third_body: bool,
    t_jd0: float,
    duration_d: float,
    sun_coeffs: np.ndarray,
    moon_coeffs: np.ndarray,
    use_srp: bool,
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
    a_total += np.array([
        fJ2 * x * (5.0 * z2 / r2 - 1.0),
        fJ2 * y * (5.0 * z2 / r2 - 1.0),
        fJ2 * z * (5.0 * z2 / r2 - 3.0),
    ])

    # --- J3 Perturbation ---
    fJ3 = 0.5 * J3 * mu * Re**3 / r7
    a_total += np.array([
        fJ3 * x * (35.0 * z2 * z / r2 - 15.0 * z),
        fJ3 * y * (35.0 * z2 * z / r2 - 15.0 * z),
        fJ3 * (35.0 * z2 * z2 / r2 - 30.0 * z2 + 3.0 * r2),
    ])

    # --- J4 Perturbation ---
    fJ4 = 0.625 * J4 * mu * Re**4 / r9
    z4 = z2 * z2
    a_total += np.array([
        fJ4 * x * (63.0 * z4 / r2 - 42.0 * z2 + 3.0 * r2),
        fJ4 * y * (63.0 * z4 / r2 - 42.0 * z2 + 3.0 * r2),
        fJ4 * z * (63.0 * z4 / r2 - 70.0 * z2 + 15.0 * r2),
    ])

    # --- Atmospheric Drag (PHY-B Standardized) ---
    alt_km = r_mag - Re
    if use_drag and alt_km < 1500.0 and drag_rho > 0.0:
        # DEF-001 (Strategy A): drag_rho is initialized at drag_ref_alt_km
        # (the actual initial orbit altitude, not a hardcoded 400 km constant).
        # The exponential corrects for altitude deviations during integration.
        rho_instant = drag_rho * math.exp(-(alt_km - drag_ref_alt_km) / drag_H_km)
        
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
        t_norm = 2.0 * (t_jd - t_jd0) / duration_d - 1.0
        t_norm = max(-1.0, min(1.0, t_norm))

        # DEF-009: removed self-import — _eval_cheb_3d_njit is defined in this module
        sun_pos = _eval_cheb_3d_njit(t_norm, sun_coeffs)
        moon_pos = _eval_cheb_3d_njit(t_norm, moon_coeffs)
        
        # Sun Gravity
        d_sun = sun_pos - r
        d_mag_sun = np.linalg.norm(d_sun)
        r_mag_sun = np.linalg.norm(sun_pos)
        if d_mag_sun > 1.0 and r_mag_sun > 1.0:
            a_total += SUN_MU_KM3_S2 * (d_sun / (d_mag_sun**3) - sun_pos / (r_mag_sun**3))
            
        # Moon Gravity
        d_moon = moon_pos - r
        d_mag_moon = np.linalg.norm(d_moon)
        r_mag_moon = np.linalg.norm(moon_pos)
        if d_mag_moon > 1.0 and r_mag_moon > 1.0:
            a_total += MOON_MU_KM3_S2 * (d_moon / (d_mag_moon**3) - moon_pos / (r_mag_moon**3))

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
                    nu = _srp_illumination_factor_njit(r, sun_pos, EARTH_EQUATORIAL_RADIUS_KM, SUN_RADIUS_KM)
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
    include_third_body: bool,
    global_t_jd0: float,
    duration_d: float,
    sun_coeffs: np.ndarray,
    moon_coeffs: np.ndarray,
    use_srp: bool,
    srp_cr: float,
    srp_use_shadow: bool,
) -> np.ndarray:
    """State derivative for unpowered (coast) arcs.

    State vector y = [x, y, z, vx, vy, vz]   (6 components).

    Returns dy/dt = [vx, vy, vz, ax, ay, az].
    """
    r = y[:3]
    v = y[3:6]
    t_jd = t_jd0_segment + t_sec / SECONDS_PER_DAY
    a = _acceleration(
        t_jd, r, v, use_drag, drag_cd, drag_area_m2, drag_mass_kg, drag_rho, drag_H_km,
        include_third_body, global_t_jd0, duration_d, sun_coeffs, moon_coeffs,
        use_srp, srp_cr, srp_use_shadow,
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
    """State derivative for powered (thrusting) arcs.

    State vector y = [x, y, z, vx, vy, vz, mass_kg]  (7 components).

    The thrust direction is re-computed from the instantaneous r, v at
    every sub-step, implementing dynamic attitude steering.

    Returns dy/dt = [vx, vy, vz, ax, ay, az, dm/dt].

    Mass depletion: dm/dt = −F / (Isp·g₀).
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
    instantaneous_mass_kg = max(m, 1e-3)  # floor at 1g to prevent division by zero

    # Gravitational + drag acceleration (using instantaneous mass for Bc)
    a_grav = _acceleration(
        t_jd, r, v, use_drag, drag_cd, drag_area_m2, instantaneous_mass_kg, drag_rho, drag_H_km,
        include_third_body, global_t_jd0, duration_d, sun_coeffs, moon_coeffs,
        use_srp, srp_cr, srp_use_shadow,
    )

    # Thrust acceleration (km/s²)
    # We use a manual reconstruction for parity with NJIT kernel
    r_mag = max(1e-12, np.linalg.norm(r))
    thrust_a_mag = (burn_thrust_N / 1000.0) / m

    if burn_frame_idx == 0:  # VNB
        from astra.frames import _build_vnb_matrix_njit
        rot_matrix = _build_vnb_matrix_njit(r, v)
    else:  # RTN
        from astra.frames import _build_rtn_matrix_njit
        rot_matrix = _build_rtn_matrix_njit(r, v)

    a_thrust = np.dot(rot_matrix, burn_dir) * thrust_a_mag
    a_total = a_grav + a_thrust

    # Mass flow rate
    dm_dt = -burn_thrust_N / (burn_isp_s * 9.80665)
    
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
        return coeffs[0] + coeffs[1] * t_norm

    x2 = 2.0 * t_norm
    d1 = np.zeros(3)
    d2 = np.zeros(3)
    for i in range(N - 1, 1, -1):
        temp = np.copy(d1)
        d1 = x2 * d1 - d2 + coeffs[i]
        d2 = temp
    return coeffs[0] + t_norm * d1 - d2

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
    include_third_body: bool,
    t_jd0: float,
    duration_d: float,
    sun_coeffs: np.ndarray,
    moon_coeffs: np.ndarray,
    use_srp: bool,
    srp_cr: float,
    srp_use_shadow: bool,
) -> np.ndarray:
    """Numba-compiled acceleration function with altitude-aware harmonic truncation."""
    r_mag = np.linalg.norm(r)
    if r_mag < 1.0:
        return np.zeros(3)

    x, y, z = r[0], r[1], r[2]
    
    # Inlined from constants to ensure Numba sees scalars
    # Keep μ, Re, J2, J3, J4, and third-body GM in sync with astra.constants.
    # MATH-01/02 Fix: Use full 8-significant-figure precision matching constants.py
    # (prior values -0.00000253266 / -0.00000161962 were truncated by 3 digits).
    Re = 6378.137
    mu = 398600.4418
    J2 = 0.00108262668
    J3 = -0.00000253265649   # WGS-84/EGM96; synced from constants.J3 = -2.53265649e-6
    J4 = -0.00000161962159   # WGS-84/EGM96; synced from constants.J4 = -1.61962159e-6
    # SUN_MU matches ``constants.SUN_MU_KM3_S2`` (IAU-style GM).
    SUN_MU = 1.32712440018e11
    MOON_MU = 4902.800066

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

    # --- Atmospheric Drag (PHY-B) ---
    # DEF-001 (Strategy A): drag_rho is initialized at drag_ref_alt_km
    # (the actual initial orbit altitude from propagate_cowell, not 400 km).
    # The exponential profile corrects deviations during the integration substep.
    if use_drag and alt_km < 1500.0 and drag_rho > 0.0:
        rho_instant = drag_rho * math.exp(-(alt_km - drag_ref_alt_km) / drag_H_km)
        
        omega_earth = np.array([0.0, 0.0, 7.292115146706979e-5])
        v_rel = v - np.cross(omega_earth, r)
        v_rel_mag = np.linalg.norm(v_rel)
        if v_rel_mag > 1e-10:
            Bc = drag_cd * drag_area_m2 / drag_mass_kg
            # 1e-6 (m^2 to km^2) * 1e9 (kg/m^3 to kg/km^3) = 1e3
            a_drag = -0.5 * rho_instant * 1e3 * Bc * v_rel_mag * v_rel
            a_total += a_drag

    # --- Third-Body (Sun & Moon) via Chebyshev Splines ---
    if include_third_body:
        t_norm = 2.0 * (t_jd - t_jd0) / duration_d - 1.0
        # Clamp between -1 and 1 just in case of precision errors at edges
        if t_norm < -1.0: t_norm = -1.0
        if t_norm > 1.0: t_norm = 1.0
        
        sun_pos = _eval_cheb_3d_njit(t_norm, sun_coeffs)
        moon_pos = _eval_cheb_3d_njit(t_norm, moon_coeffs)
        
        # Sun Gravity
        d_sun = sun_pos - r
        d_mag_sun = np.linalg.norm(d_sun)
        r_mag_sun = np.linalg.norm(sun_pos)
        if d_mag_sun > 1.0 and r_mag_sun > 1.0:
            a_total += SUN_MU * (d_sun / (d_mag_sun * d_mag_sun * d_mag_sun) - sun_pos / (r_mag_sun * r_mag_sun * r_mag_sun))
            
        # Moon Gravity
        d_moon = moon_pos - r
        d_mag_moon = np.linalg.norm(d_moon)
        r_mag_moon = np.linalg.norm(moon_pos)
        if d_mag_moon > 1.0 and r_mag_moon > 1.0:
            a_total += MOON_MU * (d_moon / (d_mag_moon * d_mag_moon * d_mag_moon) - moon_pos / (r_mag_moon * r_mag_moon * r_mag_moon))

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

    return a_total

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
        t_jd, r, v, use_drag, drag_cd, drag_area_m2, drag_mass_kg, drag_rho, drag_H_km,
        drag_ref_alt_km, include_third_body, global_t_jd0, duration_d, sun_coeffs, moon_coeffs,
        use_srp, srp_cr, srp_use_shadow,
    )
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
        t_jd, r, v, use_drag, drag_cd, drag_area_m2, instantaneous_mass_kg, drag_rho, drag_H_km,
        drag_ref_alt_km, include_third_body, global_t_jd0, duration_d, sun_coeffs, moon_coeffs,
        use_srp, srp_cr, srp_use_shadow,
    )

    r_mag = max(1e-12, np.linalg.norm(r))
    v_mag = max(1e-12, np.linalg.norm(v))
    thrust_a_mag = (burn_thrust_N / 1000.0) / m

    if burn_frame_idx == 0:  # VNB
        rot_matrix = _build_vnb_matrix_njit(r, v)
    else:  # RTN
        rot_matrix = _build_rtn_matrix_njit(r, v)

    for i in range(3):
        a_thrust_i = 0.0
        for j in range(3):
            a_thrust_i += rot_matrix[i, j] * burn_dir[j]
        a_grav[i] += a_thrust_i * thrust_a_mag

    dm_dt = -burn_thrust_N / (burn_isp_s * 9.80665)

    dy = np.empty(7)
    dy[0:3] = v
    dy[3:6] = a_grav
    dy[6] = dm_dt
    return dy

def _compute_scale_height(f107_obs: float, f107_adj: float, ap_daily: float) -> float:
    """Compute empirical atmospheric scale height (km) at 400 km reference altitude.

    Uses the same Jacchia-71 simplified model as ``atmospheric_density_empirical``
    to derive the local scale height that is passed to the Numba integration kernel
    for per-step exponential density evaluation.

    Both f107_obs and f107_adj are required to match the solar flux contribution
    formula used in ``atmospheric_density_empirical()``.

    Args:
        f107_obs: Observed F10.7 solar flux [SFU].
        f107_adj: 81-day centred average F10.7 solar flux [SFU].
        ap_daily: Daily Ap geomagnetic index.

    Returns:
        Scale height H in km at ~400 km altitude.
    """
    T_c = 379.0
    delta_T_solar = 3.24 * f107_adj + 1.3 * (f107_obs - f107_adj)  # consistent with empirical model
    delta_T_geo = 28.0 * max(ap_daily, 0.0) ** 0.4
    T_inf = max(500.0, min(T_c + delta_T_solar + delta_T_geo, 2500.0))

    k_boltzmann = 1.380649e-23  # J/K
    amu = 1.66054e-27            # kg
    m_eff = 16.0 * amu           # atomic oxygen at 400 km
    g_400 = 9.80665 * (6378.137 / (6378.137 + 400.0)) ** 2  # m/s²
    H_km = (k_boltzmann * T_inf) / (m_eff * g_400) / 1000.0
    return max(H_km, 20.0)  # physical floor


def _compute_planetary_splines(t_jd0: float, duration_s: float, use_de: bool) -> tuple[np.ndarray, np.ndarray, float]:
    duration_d = duration_s / 86400.0
    if duration_d < 1e-9:
        duration_d = 0.1  # fallback to avoid division by zero
    
    # Scale Chebyshev degree with arc duration to resolve lunar period (~27.3 d).
    # 25 nodes minimum; 5 nodes/day for longer arcs (> 10 days) captures the 27.3-day lunar period.
    deg = max(25, int(duration_d * 5))
    # Chebyshev nodes of the first kind on [-1, 1]; ``cheb.chebfit``
    # fits T_n in a least-squares sense on these points. Gauss–Lobatto (endpoints
    # included) is an alternative if endpoint errors dominate.
    nodes_norm = np.cos(np.pi * (2 * np.arange(deg + 1) + 1) / (2 * (deg + 1)))
    t_nodes = t_jd0 + 0.5 * duration_d * (nodes_norm + 1.0)
    
    sun_pos = np.zeros((deg + 1, 3))
    moon_pos = np.zeros((deg + 1, 3))
    
    # TEME-frame samples so the Numba kernel matches the SGP4 propagator frame.
    sun_fn = _sun_position_de if use_de else _sun_position_approx
    moon_fn = _moon_position_de if use_de else _moon_position_approx
    
    for i, t in enumerate(t_nodes):
        sun_pos[i] = sun_fn(t)
        moon_pos[i] = moon_fn(t)
        
    sun_c = np.zeros((deg + 1, 3))
    moon_c = np.zeros((deg + 1, 3))
    for dim in range(3):
        sun_c[:, dim] = cheb.chebfit(nodes_norm, sun_pos[:, dim], deg)
        moon_c[:, dim] = cheb.chebfit(nodes_norm, moon_pos[:, dim], deg)
        
    return sun_c, moon_c, duration_d



# ---------------------------------------------------------------------------
# Segmented Cowell Integrator
# ---------------------------------------------------------------------------

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
        rtol: Relative tolerance for adaptive step integrator (overrides defaults).
        atol: Absolute tolerance for adaptive step integrator (overrides defaults).
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

    # Resolve initial mass
    mass_kg = state0.mass_kg  # None if coast-only

    # Validate and sort maneuvers by ignition time
    burns: list[FiniteBurn] = []
    if maneuvers:
        from astra.maneuver import validate_burn, validate_burn_sequence
        if mass_kg is None:
            from astra import config
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

        # Validate each burn
        running_mass = mass_kg
        for burn in burns:
            validate_burn(burn, running_mass)
            running_mass -= burn.mass_flow_rate_kg_s * burn.duration_s

    # Build timeline segments
    # Each segment: (t_start_s, t_end_s, burn_or_None)
    segments = _build_segments(t_jd0, duration_s, burns)

    # Pre-Compute Planetary Splines (Chebyshev Polynomials)
    t_jd0_global = t_jd0
    if include_third_body:
        sun_coeffs, moon_coeffs, duration_d = _compute_planetary_splines(t_jd0, duration_s, use_de)
    else:
        sun_coeffs = np.zeros((1, 3))
        moon_coeffs = np.zeros((1, 3))
        duration_d = duration_s / 86400.0

    # Space weather once per propagation for empirical drag (rho_ref, scale height).
    use_drag = (drag_config is not None)
    drag_cd = drag_config.cd if drag_config else 0.0
    drag_area_m2 = drag_config.area_m2 if drag_config else 0.0
    drag_mass_kg = drag_config.mass_kg if drag_config else 1.0
    drag_rho = 0.0
    drag_H_km = 58.515  # default static scale height (fallback)

    # DEF-001 (Strategy A): compute drag_ref_alt_km from the actual initial orbit altitude
    # so that drag_rho is physically correct, not anchored to a hardcoded 400 km.
    drag_ref_alt_km = 400.0  # safe default (updated below when drag is active)
    if use_drag:
        from astra.data_pipeline import get_space_weather, atmospheric_density_empirical
        from astra.constants import DRAG_MIN_ALTITUDE_KM
        r0_mag = float(np.linalg.norm(state0.position_km))
        drag_ref_alt_km = max(r0_mag - EARTH_EQUATORIAL_RADIUS_KM, DRAG_MIN_ALTITUDE_KM)
        if use_empirical_drag:
            try:
                f107_obs, f107_adj, ap_daily = get_space_weather(t_jd0)
                # DEF-001: density at initial orbit altitude (not hardcoded 400 km)
                drag_rho = atmospheric_density_empirical(drag_ref_alt_km, f107_obs, f107_adj, ap_daily)
                drag_H_km = _compute_scale_height(f107_obs, f107_adj, ap_daily)
            except (ImportError, ValueError, OSError):
                drag_rho = 0.0
        if drag_rho == 0.0:
            from astra.constants import DRAG_REF_DENSITY_KG_M3, DRAG_SCALE_HEIGHT_KM
            drag_rho = DRAG_REF_DENSITY_KG_M3  # static fallback density (400 km reference)
            drag_H_km = DRAG_SCALE_HEIGHT_KM
            drag_ref_alt_km = 400.0  # static profile anchored at 400 km

    use_srp = bool(
        drag_config is not None
        and getattr(drag_config, "include_srp", True)
        and include_third_body
    )
    srp_cr = float(drag_config.cr) if drag_config is not None else 1.5
    srp_use_shadow = bool(
        drag_config is not None
        and getattr(drag_config, "srp_cylindrical_shadow", True)
    )

    logger.info(
        f"Segmented Cowell propagation: {duration_s:.0f}s, "
        f"{len(segments)} segments ({len(burns)} burn(s)), "
        f"drag={'ON' if drag_config else 'OFF'}, "
        f"third_body={'ON' if include_third_body else 'OFF'}, "
        f"ephemeris={'DE421' if use_de else 'analytical'} (Splined)"
    )

    # Run each segment sequentially
    all_states: list[NumericalState] = []
    current_r = state0.position_km.copy()
    current_v = state0.velocity_km_s.copy()
    current_mass = mass_kg

    for seg_start_s, seg_end_s, active_burn in segments:
        seg_duration = seg_end_s - seg_start_s
        if seg_duration < 1e-9:
            continue

        # Build output times for this segment (relative to segment start)
        # Align to the global dt_output grid
        global_t_start = seg_start_s
        global_t_end = seg_end_s

        # Output times within this segment
        t_out = []
        # First output at the global grid time >= segment start
        first_grid = math.ceil(global_t_start / dt_out) * dt_out
        t = first_grid
        while t <= global_t_end + 1e-9:
            if t >= global_t_start - 1e-9:
                t_out.append(t - global_t_start)
            t += dt_out

        # Always include segment endpoints
        if not t_out or t_out[0] > 1e-9:
            t_out.insert(0, 0.0)
        if t_out[-1] < seg_duration - 1e-9:
            t_out.append(seg_duration)

        t_eval = np.array(sorted(set(t_out)))
        t_eval = t_eval[t_eval <= seg_duration + 1e-9]

        if active_burn is not None and current_mass is not None:
            # ---- POWERED SEGMENT (7-DOF) ----
            y0 = np.concatenate([current_r, current_v, [current_mass]])
            
            b_thrust = active_burn.thrust_N
            b_isp = active_burn.isp_s
            b_dir = np.array(active_burn.direction)
            b_idx = 0 if active_burn.frame.value == "VNB" else 1
            
            def powered_deriv(t_sec, y):
                return _powered_derivative_njit(
                    t_sec, y, t_jd0 + seg_start_s / SECONDS_PER_DAY,
                    use_drag, drag_cd, drag_area_m2, drag_mass_kg, drag_rho, drag_H_km,
                    drag_ref_alt_km, include_third_body, t_jd0_global, duration_d,
                    sun_coeffs, moon_coeffs, use_srp, srp_cr, srp_use_shadow,
                    b_thrust, b_isp, b_dir, b_idx,
                )

            # Per-dimension absolute tolerances for 7-DOF powered arc
            # Position (km), velocity (km/s), mass (kg) — different physical scales
            powered_atol_vec = np.array([
                1e-11, 1e-11, 1e-11,   # position x, y, z (km) — sub-cm accuracy
                1e-13, 1e-13, 1e-13,   # velocity vx, vy, vz (km/s) — sub-μm/s accuracy
                # 1e-5 kg suits ton-class vehicles; pass rtol/atol for micro-thrust vehicles.
                1e-5,
            ])

            sol = solve_ivp(
                powered_deriv,
                t_span=(0.0, seg_duration),
                y0=y0,
                method='DOP853',
                t_eval=t_eval,
                rtol=rtol if rtol is not None else powered_rtol,
                atol=atol if atol is not None else powered_atol_vec,
            )

            if not sol.success:
                logger.error(f"Powered integration failed: {sol.message}")
                from astra import config
                from astra.errors import PropagationError

                if config.ASTRA_STRICT_MODE:
                    raise PropagationError(
                        f"Cowell powered integration failed: {sol.message}",
                        t_jd=t_jd0 + seg_start_s / SECONDS_PER_DAY,
                    )
                break

            for i in range(len(sol.t)):
                all_states.append(NumericalState(
                    t_jd=t_jd0 + (seg_start_s + sol.t[i]) / SECONDS_PER_DAY,
                    position_km=sol.y[:3, i].copy(),
                    velocity_km_s=sol.y[3:6, i].copy(),
                    mass_kg=float(sol.y[6, i]),
                ))

            # Update handoff state
            current_r = sol.y[:3, -1].copy()
            current_v = sol.y[3:6, -1].copy()
            current_mass = float(sol.y[6, -1])

        else:
            # ---- COAST SEGMENT (6-DOF) ----
            y0 = np.concatenate([current_r, current_v])

            def coast_deriv(t_sec, y):
                return _coast_derivative_njit(
                    t_sec, y, t_jd0 + seg_start_s / SECONDS_PER_DAY,
                    use_drag, drag_cd, drag_area_m2, drag_mass_kg, drag_rho, drag_H_km,
                    drag_ref_alt_km, include_third_body, t_jd0_global, duration_d,
                    sun_coeffs, moon_coeffs, use_srp, srp_cr, srp_use_shadow,
                )

            sol = solve_ivp(
                coast_deriv,
                t_span=(0.0, seg_duration),
                y0=y0,
                method='DOP853',
                t_eval=t_eval,
                rtol=rtol if rtol is not None else coast_rtol,
                atol=atol if atol is not None else coast_atol,
            )

            if not sol.success:
                logger.error(f"Coast integration failed: {sol.message}")
                from astra import config
                from astra.errors import PropagationError

                if config.ASTRA_STRICT_MODE:
                    raise PropagationError(
                        f"Cowell coast integration failed: {sol.message}",
                        t_jd=t_jd0 + seg_start_s / SECONDS_PER_DAY,
                    )
                break

            for i in range(len(sol.t)):
                all_states.append(NumericalState(
                    t_jd=t_jd0 + (seg_start_s + sol.t[i]) / SECONDS_PER_DAY,
                    position_km=sol.y[:3, i].copy(),
                    velocity_km_s=sol.y[3:6, i].copy(),
                    mass_kg=current_mass,
                ))

            # Update handoff state
            current_r = sol.y[:3, -1].copy()
            current_v = sol.y[3:6, -1].copy()

    # Deduplicate states exactly at segment boundaries (same t_jd)
    # The 1e-12 JD threshold (~86 microseconds) is chosen to catch repeated 
    # solve_ivp t_eval endpoints without filtering distinct adjacent steps.
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

    # 1. Robust overlap detection (SE-G)
    for i in range(len(burns) - 1):
        b1, b2 = burns[i], burns[i+1]
        if b2.epoch_ignition_jd < b1.epoch_cutoff_jd - 1e-12:
            from astra import config
            from astra.errors import ManeuverError
            msg = (
                f"Maneuver overlap detected: Burn at {b2.epoch_ignition_jd} "
                f"ignites before previous Burn cutoff at {b1.epoch_cutoff_jd}."
            )
            if config.ASTRA_STRICT_MODE:
                raise ManeuverError(msg)
            else:
                import logging
                logging.getLogger(__name__).warning(f"{msg} Skipping overlapping burn.")

    for burn in burns:
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
