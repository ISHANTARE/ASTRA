# astra/propagator.py
"""ASTRA Core Numerical Propagator — Cowell's Method.

Implements a high-fidelity numerical orbit propagator using Cowell's direct
integration approach with a Dormand-Prince RK7(8) adaptive-step integrator.

Force model includes:
- Two-body Keplerian gravity
- J2, J3, J4 zonal harmonic perturbations (WGS84)
- Exponential atmospheric drag
- Solar third-body point-mass perturbation
- Lunar third-body point-mass perturbation

References:
    Vallado, D. A. (2013). Fundamentals of Astrodynamics and Applications.
    Montenbruck & Gill (2000). Satellite Orbits.
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
    DRAG_REF_DENSITY_KG_M3,
    DRAG_REF_ALTITUDE_KM,
    DRAG_SCALE_HEIGHT_KM,
)
from astra.log import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class NumericalState:
    """Full 6-DOF kinematic state vector at a single epoch.

    Attributes:
        t_jd: Julian Date of this state.
        position_km: Shape (3,) ECI position [x, y, z] in km.
        velocity_km_s: Shape (3,) ECI velocity [vx, vy, vz] in km/s.
    """
    t_jd: float
    position_km: np.ndarray  # shape (3,)
    velocity_km_s: np.ndarray  # shape (3,)


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
# Low-Fidelity Ephemeris (Sun & Moon position approximations)
# ---------------------------------------------------------------------------

def _sun_position_approx(t_jd: float) -> np.ndarray:
    """Approximate geocentric Sun position in ECI (km).

    Uses a simplified analytical solar ephemeris accurate to ~1° in ecliptic
    longitude. Sufficient for third-body perturbation forces.

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
    Accuracy: ~1° in longitude, ~0.5° in latitude.
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
# Force Model
# ---------------------------------------------------------------------------

def _acceleration(
    t_jd: float,
    r: np.ndarray,
    v: np.ndarray,
    drag_config: Optional[DragConfig] = None,
    include_third_body: bool = True,
) -> np.ndarray:
    """Compute total acceleration vector in ECI frame (km/s²).

    Forces:
        1. Two-body + J2/J3/J4 zonal harmonics
        2. Exponential atmospheric drag
        3. Solar/Lunar third-body point-mass gravity
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
        if alt_km < 1000.0:  # Drag negligible above ~1000 km
            rho = DRAG_REF_DENSITY_KG_M3 * math.exp(
                -(alt_km - DRAG_REF_ALTITUDE_KM) / DRAG_SCALE_HEIGHT_KM
            )

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
        for body_pos_fn, body_mu in [
            (_sun_position_approx, SUN_MU_KM3_S2),
            (_moon_position_approx, MOON_MU_KM3_S2),
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
# Cowell Integrator
# ---------------------------------------------------------------------------

def propagate_cowell(
    state0: NumericalState,
    duration_s: float,
    dt_output_s: float = 60.0,
    drag_config: Optional[DragConfig] = None,
    include_third_body: bool = True,
    rtol: float = 1e-10,
    atol: float = 1e-12,
) -> list[NumericalState]:
    """Propagate an orbit using Cowell's method with RK7(8) Dormand-Prince.

    This is a high-fidelity numerical propagator suitable for precise
    ephemeris generation where SGP4 analytical accuracy is insufficient.

    Args:
        state0: Initial state (position + velocity at epoch).
        duration_s: Propagation duration in seconds.
        dt_output_s: Output time step in seconds (default 60s).
        drag_config: Optional atmospheric drag parameters.
        include_third_body: Whether to include Sun/Moon gravity.
        rtol: Relative tolerance for adaptive step integrator.
        atol: Absolute tolerance for adaptive step integrator.

    Returns:
        List of NumericalState objects at each output time step.
    """
    y0 = np.concatenate([state0.position_km, state0.velocity_km_s])
    t_jd0 = state0.t_jd

    t_eval = np.arange(0.0, duration_s + 1e-9, dt_output_s)

    def derivatives(t_sec: float, y: np.ndarray) -> np.ndarray:
        r = y[:3]
        v = y[3:]
        t_jd = t_jd0 + t_sec / 86400.0
        a = _acceleration(t_jd, r, v, drag_config, include_third_body)
        return np.concatenate([v, a])

    logger.info(
        f"Cowell propagation: {duration_s:.0f}s, "
        f"drag={'ON' if drag_config else 'OFF'}, "
        f"third_body={'ON' if include_third_body else 'OFF'}"
    )

    sol = solve_ivp(
        derivatives,
        t_span=(0.0, duration_s),
        y0=y0,
        method='DOP853',  # 8th-order Dormand-Prince
        t_eval=t_eval,
        rtol=rtol,
        atol=atol,
        dense_output=True,
    )

    if not sol.success:
        logger.error(f"Integration failed: {sol.message}")
        return []

    states = []
    for i in range(len(sol.t)):
        states.append(NumericalState(
            t_jd=t_jd0 + sol.t[i] / 86400.0,
            position_km=sol.y[:3, i].copy(),
            velocity_km_s=sol.y[3:6, i].copy(),
        ))

    return states
