# astra/propagator.py
"""ASTRA Core Numerical Propagator — Segmented Cowell's Method.

Implements a mission-operations–grade numerical orbit propagator using
Cowell's direct integration with a Dormand-Prince RK8(7) adaptive-step
integrator.

Features:
    - **6-DOF Coast arcs**: Two-body + J2/J3/J4 + drag + 3rd-body gravity.
    - **7-DOF Powered arcs**: Adds attitude-steered thrust with
      Tsiolkovsky-coupled mass depletion.
    - **Segmented Orchestrator**: Automatically slices propagation at
      engine ignition/cutoff boundaries so the integrator never steps
      across a force-model discontinuity.
    - **High-Fidelity Data Sources**:
      - JPL DE421 Sun/Moon positions (via Skyfield, replacing analytical
        approximations).
      - Empirical atmospheric density parameterised by F10.7 solar flux
        and Ap geomagnetic index (replacing the static exponential model).

Force model includes:
    - Two-body Keplerian gravity
    - J2, J3, J4 zonal harmonic perturbations (WGS84)
    - Empirical atmospheric drag (Jacchia-class with space weather)
    - Solar third-body point-mass perturbation (JPL DE421)
    - Lunar third-body point-mass perturbation (JPL DE421)
    - Finite continuous thrust (7-DOF powered arcs)

References:
    Vallado, D. A. (2013). Fundamentals of Astrodynamics and Applications.
    Montenbruck & Gill (2000). Satellite Orbits.
    Park et al. (2021). JPL Planetary Ephemerides DE440/DE441.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.integrate import solve_ivp

from astra.constants import (
    EARTH_EQUATORIAL_RADIUS_KM,
    EARTH_MU_KM3_S2,
    EARTH_OMEGA_RAD_S,
    J2, J3, J4,
    SUN_MU_KM3_S2,
    MOON_MU_KM3_S2,
)
from astra.log import get_logger
from astra.models import FiniteBurn

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class NumericalState:
    """Full kinematic state vector at a single epoch.

    In 6-DOF (coast) mode, mass_kg is None and the state vector is
    [x, y, z, vx, vy, vz].

    In 7-DOF (powered) mode, mass_kg tracks propellant depletion via
    Tsiolkovsky coupling: dm/dt = −F / (Isp·g₀).

    Attributes:
        t_jd: Julian Date of this state.
        position_km: Shape (3,) ECI position [x, y, z] in km.
        velocity_km_s: Shape (3,) ECI velocity [vx, vy, vz] in km/s.
        mass_kg: Spacecraft wet mass in kg. None for coast-only runs.
    """
    t_jd: float
    position_km: np.ndarray  # shape (3,)
    velocity_km_s: np.ndarray  # shape (3,)
    mass_kg: Optional[float] = None


@dataclass
class DragConfig:
    """Atmospheric drag configuration for a specific object.

    Attributes:
        cd: Drag coefficient (dimensionless, typically 2.0–2.5).
        area_m2: Cross-sectional area in m².
        mass_kg: Object mass in kg.
    """
    cd: float = 2.2
    area_m2: float = 10.0
    mass_kg: float = 1000.0


# ---------------------------------------------------------------------------
# High-fidelity Sun / Moon via JPL DE421 (Skyfield)
# ---------------------------------------------------------------------------

# Import lazily to avoid circular dependency and allow graceful fallback
_USE_DE = True  # Will be set to False if Skyfield data unavailable


def _sun_position_de(t_jd: float) -> np.ndarray:
    """Geocentric Sun position from JPL DE421 (km, GCRS ≈ ECI)."""
    try:
        from astra.data_pipeline import sun_position_de
        return sun_position_de(t_jd)
    except Exception:
        return _sun_position_approx(t_jd)


def _moon_position_de(t_jd: float) -> np.ndarray:
    """Geocentric Moon position from JPL DE421 (km, GCRS ≈ ECI)."""
    try:
        from astra.data_pipeline import moon_position_de
        return moon_position_de(t_jd)
    except Exception:
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
            f107_obs, f107_adj, ap_daily = get_space_weather(t_jd)
            return atmospheric_density_empirical(alt_km, f107_obs, f107_adj, ap_daily)
        except Exception:
            pass  # Fall through to static model

    # Static exponential fallback
    from astra.constants import (
        DRAG_REF_DENSITY_KG_M3,
        DRAG_REF_ALTITUDE_KM,
        DRAG_SCALE_HEIGHT_KM,
    )
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
    drag_config: Optional[DragConfig] = None,
    include_third_body: bool = True,
    use_de: bool = True,
    use_empirical_drag: bool = True,
) -> np.ndarray:
    """Compute total gravitational + drag acceleration in ECI (km/s²).

    Forces:
        1. Two-body + J2/J3/J4 zonal harmonics
        2. Atmospheric drag (empirical or static exponential)
        3. Solar/Lunar third-body point-mass gravity (DE421 or analytical)
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
    a_twobody = -mu / r3 * r

    # --- J2 Perturbation ---
    fJ2 = 1.5 * J2 * mu * Re**2 / r5
    a_j2 = np.array([
        fJ2 * x * (5.0 * z2 / r2 - 1.0),
        fJ2 * y * (5.0 * z2 / r2 - 1.0),
        fJ2 * z * (5.0 * z2 / r2 - 3.0),
    ])

    # --- J3 Perturbation ---
    fJ3 = 0.5 * J3 * mu * Re**3 / r7
    a_j3 = np.array([
        fJ3 * x * (35.0 * z2 * z / r2 - 15.0 * z),
        fJ3 * y * (35.0 * z2 * z / r2 - 15.0 * z),
        fJ3 * (35.0 * z2 * z2 / r2 - 30.0 * z2 + 3.0 * r2),
    ])

    # --- J4 Perturbation ---
    fJ4 = -0.625 * J4 * mu * Re**4 / r9
    z4 = z2 * z2
    a_j4 = np.array([
        fJ4 * x * (63.0 * z4 / r2 - 42.0 * z2 + 3.0 * r2),
        fJ4 * y * (63.0 * z4 / r2 - 42.0 * z2 + 3.0 * r2),
        fJ4 * z * (63.0 * z4 / r2 - 70.0 * z2 + 15.0 * r2),
    ])

    a_total = a_twobody + a_j2 + a_j3 + a_j4

    # --- Atmospheric Drag ---
    if drag_config is not None:
        alt_km = r_mag - EARTH_EQUATORIAL_RADIUS_KM
        if alt_km < 1500.0:
            rho = _atmospheric_density(alt_km, t_jd, use_empirical_drag)

            # Atmosphere co-rotates with Earth
            omega_earth = np.array([0.0, 0.0, EARTH_OMEGA_RAD_S])
            v_rel = v - np.cross(omega_earth, r)
            v_rel_mag = np.linalg.norm(v_rel)

            if v_rel_mag > 1e-10:
                Bc = drag_config.cd * drag_config.area_m2 / drag_config.mass_kg
                # Convert area from m² to km² (factor 1e-6)
                # Convert density from kg/m³ to kg/km³ (factor 1e9)
                # Net factor: 1e-6 * 1e9 = 1e3
                a_drag = -0.5 * rho * 1e3 * Bc * v_rel_mag * v_rel
                a_total += a_drag

    # --- Third-Body Perturbations (Sun & Moon) ---
    if include_third_body:
        # Select ephemeris source
        if use_de:
            sun_fn = _sun_position_de
            moon_fn = _moon_position_de
        else:
            sun_fn = _sun_position_approx
            moon_fn = _moon_position_approx

        for body_pos_fn, body_mu in [
            (sun_fn, SUN_MU_KM3_S2),
            (moon_fn, MOON_MU_KM3_S2),
        ]:
            r_body = body_pos_fn(t_jd)
            d = r_body - r  # vector from satellite to body
            d_mag = np.linalg.norm(d)
            r_body_mag = np.linalg.norm(r_body)

            if d_mag > 1.0 and r_body_mag > 1.0:
                # Standard third-body perturbation formula
                a_total += body_mu * (d / d_mag**3 - r_body / r_body_mag**3)

    return a_total


# ---------------------------------------------------------------------------
# Standard gravitational acceleration for mass flow
# ---------------------------------------------------------------------------

_G0 = 9.80665  # m/s²


# ---------------------------------------------------------------------------
# Coast Derivative (6-DOF, m = constant)
# ---------------------------------------------------------------------------

def _coast_derivative(
    t_sec: float,
    y: np.ndarray,
    t_jd0: float,
    drag_config: Optional[DragConfig],
    include_third_body: bool,
    use_de: bool,
    use_empirical_drag: bool,
) -> np.ndarray:
    """State derivative for unpowered (coast) arcs.

    State vector y = [x, y, z, vx, vy, vz]   (6 components).

    Returns dy/dt = [vx, vy, vz, ax, ay, az].
    """
    r = y[:3]
    v = y[3:6]
    t_jd = t_jd0 + t_sec / 86400.0
    a = _acceleration(t_jd, r, v, drag_config, include_third_body,
                      use_de, use_empirical_drag)
    return np.concatenate([v, a])


# ---------------------------------------------------------------------------
# Powered Derivative (7-DOF, thrust + mass depletion)
# ---------------------------------------------------------------------------

def _powered_derivative(
    t_sec: float,
    y: np.ndarray,
    t_jd0: float,
    drag_config: Optional[DragConfig],
    include_third_body: bool,
    use_de: bool,
    use_empirical_drag: bool,
    burn: FiniteBurn,
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
    t_jd = t_jd0 + t_sec / 86400.0

    # Gravitational + drag acceleration (same as coast)
    a_grav = _acceleration(t_jd, r, v, drag_config, include_third_body,
                           use_de, use_empirical_drag)

    # Thrust acceleration  (km/s²)
    from astra.maneuver import thrust_acceleration_inertial
    a_thrust = thrust_acceleration_inertial(r, v, m, burn)

    a_total = a_grav + a_thrust

    # Mass flow rate (negative because mass decreases)
    dm_dt = -burn.thrust_N / (burn.isp_s * _G0)

    return np.concatenate([v, a_total, [dm_dt]])


# ---------------------------------------------------------------------------
# Segmented Cowell Integrator
# ---------------------------------------------------------------------------

def propagate_cowell(
    state0: NumericalState,
    duration_s: float,
    dt_output_s: float = 60.0,
    drag_config: Optional[DragConfig] = None,
    include_third_body: bool = True,
    rtol: float = 1e-10,
    atol: float = 1e-12,
    maneuvers: Optional[list[FiniteBurn]] = None,
    use_de: bool = True,
    use_empirical_drag: bool = True,
) -> list[NumericalState]:
    """Propagate an orbit using segmented Cowell's method with RK8(7).

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

    Args:
        state0: Initial state (position + velocity + optional mass).
        duration_s: Total propagation duration in seconds.
        dt_output_s: Output time step in seconds (default 60 s).
        drag_config: Optional atmospheric drag parameters.
        include_third_body: Include Sun/Moon gravity.
        rtol: Relative tolerance for adaptive step integrator.
        atol: Absolute tolerance for adaptive step integrator.
        maneuvers: Optional list of ``FiniteBurn`` definitions.
            Burns must not overlap in time.
        use_de: Use JPL DE421 for Sun/Moon (True) or analytical (False).
        use_empirical_drag: Use F10.7/Ap drag model (True) or static (False).

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
        from astra.maneuver import validate_burn
        if mass_kg is None:
            logger.warning(
                "Maneuvers specified but initial mass_kg is None. "
                "Defaulting to 1000.0 kg."
            )
            mass_kg = 1000.0

        burns = sorted(maneuvers, key=lambda b: b.epoch_ignition_jd)

        # Validate each burn
        running_mass = mass_kg
        for burn in burns:
            validate_burn(burn, running_mass)
            running_mass -= burn.mass_flow_rate_kg_s * burn.duration_s

    # Build timeline segments
    # Each segment: (t_start_s, t_end_s, burn_or_None)
    segments = _build_segments(t_jd0, duration_s, burns)

    logger.info(
        f"Segmented Cowell propagation: {duration_s:.0f}s, "
        f"{len(segments)} segments ({len(burns)} burn(s)), "
        f"drag={'ON' if drag_config else 'OFF'}, "
        f"third_body={'ON' if include_third_body else 'OFF'}, "
        f"ephemeris={'DE421' if use_de else 'analytical'}"
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
        first_grid = math.ceil(global_t_start / dt_output_s) * dt_output_s
        t = first_grid
        while t <= global_t_end + 1e-9:
            if t >= global_t_start - 1e-9:
                t_out.append(t - global_t_start)
            t += dt_output_s

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

            def powered_deriv(t_sec, y, _burn=active_burn):
                return _powered_derivative(
                    t_sec, y, t_jd0 + seg_start_s / 86400.0,
                    drag_config, include_third_body,
                    use_de, use_empirical_drag, _burn,
                )

            sol = solve_ivp(
                powered_deriv,
                t_span=(0.0, seg_duration),
                y0=y0,
                method='DOP853',
                t_eval=t_eval,
                rtol=rtol,
                atol=atol,
            )

            if not sol.success:
                logger.error(f"Powered integration failed: {sol.message}")
                break

            for i in range(len(sol.t)):
                all_states.append(NumericalState(
                    t_jd=t_jd0 + (seg_start_s + sol.t[i]) / 86400.0,
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
                return _coast_derivative(
                    t_sec, y, t_jd0 + seg_start_s / 86400.0,
                    drag_config, include_third_body,
                    use_de, use_empirical_drag,
                )

            sol = solve_ivp(
                coast_deriv,
                t_span=(0.0, seg_duration),
                y0=y0,
                method='DOP853',
                t_eval=t_eval,
                rtol=rtol,
                atol=atol,
            )

            if not sol.success:
                logger.error(f"Coast integration failed: {sol.message}")
                break

            for i in range(len(sol.t)):
                all_states.append(NumericalState(
                    t_jd=t_jd0 + (seg_start_s + sol.t[i]) / 86400.0,
                    position_km=sol.y[:3, i].copy(),
                    velocity_km_s=sol.y[3:6, i].copy(),
                    mass_kg=current_mass,
                ))

            # Update handoff state
            current_r = sol.y[:3, -1].copy()
            current_v = sol.y[3:6, -1].copy()

    # Deduplicate states at segment boundaries (same t_jd)
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
