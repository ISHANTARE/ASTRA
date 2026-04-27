# astra/constants.py
"""ASTRA Core constants.
All simulation parameters, physical constants, orbital regime boundaries,
and conjunction thresholds used across the library. These are module-level
immutable values — no mutable state.
"""
from __future__ import annotations
# ---------------------------------------------------------------------------
# Simulation parameters
# ---------------------------------------------------------------------------
SIMULATION_WINDOW_HOURS: int = 24
SIMULATION_STEP_MINUTES: int = 5
SIMULATION_STEPS: int = 288  # 24 * 60 / 5
# ---------------------------------------------------------------------------
# Earth parameters
# ---------------------------------------------------------------------------
# Mean spherical radius — for astrodynamics use EARTH_EQUATORIAL_RADIUS_KM (WGS84).
# Using this where Re (equatorial) is expected introduces a ~7 km altitude error.
EARTH_RADIUS_KM: float = 6371.0
EARTH_EQUATORIAL_RADIUS_KM: float = 6378.137
EARTH_MU_KM3_S2: float = 398600.4418  # gravitational parameter (km³/s²)
EARTH_OMEGA_RAD_S: float = (
    7.292115146706979e-5  # Earth sidereal rotation rate (rad/s) — IAU/IERS 2010
)
# ---------------------------------------------------------------------------
# Geopotential Zonal Harmonics (WGS84)
# ---------------------------------------------------------------------------
J2: float = 1.08262668e-3
J3: float = -2.53265649e-6
J4: float = -1.61962159e-6
# J5 and J6 — EGM96/WGS-84 unnormalized zonal harmonics.
# Required for high-fidelity MEO/GEO/HEO long-horizon propagations.
# Skipping J5 introduces a quantifiable secular nodal-rate error above ~20,000 km.
# Reference: EGM96 coefficients (Lemoine et al. 1998, NASA/TP-1998-206861).
J5: float = -2.27626414e-7
J6: float = 5.40681239e-7
# ---------------------------------------------------------------------------
# Third-body gravitational parameters (km³/s²)
# ---------------------------------------------------------------------------
SUN_MU_KM3_S2: float = 1.32712440018e11
MOON_MU_KM3_S2: float = 4902.800066
SUN_EARTH_DISTANCE_KM: float = 1.496e8  # ~1 AU
MOON_EARTH_DISTANCE_KM: float = 384400.0
SUN_RADIUS_KM: float = 695700.0
# ---------------------------------------------------------------------------
# Atmospheric drag model (exponential)
# ---------------------------------------------------------------------------
DRAG_REF_DENSITY_KG_M3: float = 3.725e-12  # reference at 400 km
DRAG_REF_ALTITUDE_KM: float = 400.0
DRAG_SCALE_HEIGHT_KM: float = 58.515  # scale height for ~400 km
DRAG_MIN_ALTITUDE_KM: float = 100.0  # below this, exponential model is undefined
DRAG_MAX_ALTITUDE_KM: float = 1500.0  # above this, density is negligible
# ---------------------------------------------------------------------------
# Orbital regime boundaries (km altitude above Earth's surface)
# ---------------------------------------------------------------------------
LEO_MIN_KM: float = 160.0
LEO_MAX_KM: float = 2000.0
MEO_MIN_KM: float = 2000.0
MEO_MAX_KM: float = 35786.0
GEO_ALTITUDE_KM: float = 35786.0
# HEO is defined by eccentricity, not altitude. Use HEO_ECCENTRICITY_THRESHOLD.
# HEO_MIN_KM is retained for backward compatibility but is DEPRECATED.
HEO_MIN_KM: float = (
    35786.0  # DEPRECATED — conflatess GEO altitude with HEO; use HEO_ECCENTRICITY_THRESHOLD
)
HEO_ECCENTRICITY_THRESHOLD: float = (
    0.25  # e > 0.25 classifies an orbit as Highly Elliptical (Molniya etc.)
)
# ---------------------------------------------------------------------------
# Conjunction thresholds
# ---------------------------------------------------------------------------
CONJUNCTION_THRESHOLD_KM: float = 5.0  # fine-grained detection
COARSE_FILTER_THRESHOLD_KM: float = 50.0  # coarse pre-filter
# ---------------------------------------------------------------------------
# Collision probability reference volume (Chan formula)
# ---------------------------------------------------------------------------
COLLISION_VOLUME_SCALE_M: float = 10.0  # combined hard-body radius in meters
# ---------------------------------------------------------------------------
# TLE age thresholds (days)
# ---------------------------------------------------------------------------
TLE_AGE_LEO_MAX_DAYS: float = 7.0
TLE_AGE_DEFAULT_MAX_DAYS: float = 14.0
# ---------------------------------------------------------------------------
# Spatial grid parameters (used in conjunction detection)
# ---------------------------------------------------------------------------
GRID_ALTITUDE_CELL_KM: float = 100.0
GRID_ANGLE_CELL_DEG: float = 10.0
# ---------------------------------------------------------------------------
# Commonly-used physical and time scalars (eliminates magic-number literals)
# ---------------------------------------------------------------------------
SECONDS_PER_DAY: float = 86400.0
MINUTES_PER_DAY: float = 1440.0
G0_STD: float = 9.80665  # Standard gravitational acceleration at sea level (m/s²) — IAU
# ---------------------------------------------------------------------------
# Solar and Astronomical constants
# ---------------------------------------------------------------------------
AU_KM: float = 149597870.7  # Astronomical Unit (km) — IAU 2012
SRP_P0_N_M2: float = 4.56e-6  # Solar radiation pressure at 1 AU (N/m²)
# ---------------------------------------------------------------------------
# Physical scalars used inside Numba JIT kernels — must never drift.
# ---------------------------------------------------------------------------
#: Universal gas constant (J/K/mol) — inlined in propagator._nrlmsise00_density_njit.
R_GAS: float = 8.314462618
#: G0 expressed in km/s² for use inside Numba kernels that work in km.
#: propagator._powered_derivative_njit inlines ``g0 = 9.80665e-3`` directly;
#: this constant documents the conversion and is guarded below.
G0_STD_KM_S2: float = G0_STD * 1e-3  # 9.80665e-3 km/s²
# ---------------------------------------------------------------------------
# Compile-time guards: ensure Numba inlined literals stay in sync [LOW-01]
# The Numba JIT kernels in propagator.py, covariance.py, and frames.py cannot
# import this module at compile time, so they inline numeric literals.
# These assertions fire at Python import time to catch any drift.
# ---------------------------------------------------------------------------
assert SRP_P0_N_M2 == 4.56e-6, (
    f"SRP_P0_N_M2 ({SRP_P0_N_M2}) does not match the Numba inlined literal "
    "4.56e-6 in propagator.py._acceleration_njit. Update both in sync."
)
assert AU_KM == 149597870.7, (
    f"AU_KM ({AU_KM}) does not match the Numba inlined literal "
    "149597870.7 in propagator.py._acceleration_njit. Update both in sync."
)
# Earth rotation rate guard.
assert EARTH_OMEGA_RAD_S == 7.292115146706979e-5, (
    f"EARTH_OMEGA_RAD_S ({EARTH_OMEGA_RAD_S}) does not match the inlined literal "
    "7.292115146706979e-5 in propagator.py, covariance.py, frames.py. Update all in sync."
)
# WGS84 equatorial radius guard.
assert EARTH_EQUATORIAL_RADIUS_KM == 6378.137, (
    f"EARTH_EQUATORIAL_RADIUS_KM ({EARTH_EQUATORIAL_RADIUS_KM}) does not match the "
    "Numba inlined literal 6378.137 in frames.py.ecef_to_geodetic_wgs84. Update both in sync."
)
# Standard gravity guard.
assert G0_STD == 9.80665, (
    f"G0_STD ({G0_STD}) does not match the IAU standard 9.80665 m/s². "
    "Update constants.py and all import sites in sync."
)
# J2 zonal harmonic guard.
# propagator._acceleration_njit inlines J2 = 0.00108262668 at line ~636 (J2c).
# _compute_force_jacobian also inlines J2c = 0.00108262668 at line ~636.
# Both must match constants.J2 exactly.
assert abs(J2 - 1.08262668e-3) < 1e-15, (
    f"J2 ({J2!r}) diverged from canonical EGM96/WGS-84 value 1.08262668e-3. "
    "Update constants.py AND propagator.py J2 / J2c inlined literals in sync."
)
# Earth gravitational parameter guard.
# propagator._acceleration_njit inlines mu = 398600.4418.
assert EARTH_MU_KM3_S2 == 398600.4418, (
    f"EARTH_MU_KM3_S2 ({EARTH_MU_KM3_S2}) does not match the Numba inlined literal "
    "398600.4418 in propagator.py._acceleration_njit. Update both in sync."
)
# G0 in km/s² guard.
# propagator._powered_derivative_njit inlines ``g0 = 9.80665e-3`` (km/s²).
assert abs(G0_STD_KM_S2 - 9.80665e-3) < 1e-20, (
    f"G0_STD_KM_S2 ({G0_STD_KM_S2!r}) diverged from 9.80665e-3 km/s². "
    "Update constants.py AND propagator.py._powered_derivative_njit in sync."
)
# Universal gas constant guard.
# propagator._nrlmsise00_density_njit inlines R_GAS = 8.314462618 J/(K·mol).
assert abs(R_GAS - 8.314462618) < 1e-9, (
    f"R_GAS ({R_GAS!r}) diverged from CODATA 2018 value 8.314462618 J/(K·mol). "
    "Update constants.py AND propagator.py._nrlmsise00_density_njit in sync."
)
# Sun radius guard.
# propagator._acceleration_njit inlines 695700.0 for Sun radius in SRP shadow.
assert SUN_RADIUS_KM == 695700.0, (
    f"SUN_RADIUS_KM ({SUN_RADIUS_KM}) does not match the Numba inlined literal "
    "695700.0 in propagator.py._srp_illumination_factor_dual_cone_njit. Update both in sync."
)
