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
EARTH_RADIUS_KM: float = 6371.0
EARTH_EQUATORIAL_RADIUS_KM: float = 6378.137
EARTH_MU_KM3_S2: float = 398600.4418  # gravitational parameter (km³/s²)
EARTH_OMEGA_RAD_S: float = 7.2921159e-5  # Earth rotation rate (rad/s)

# ---------------------------------------------------------------------------
# Geopotential Zonal Harmonics (WGS84)
# ---------------------------------------------------------------------------
J2: float = 1.08262668e-3
J3: float = -2.53265649e-6
J4: float = -1.61962159e-6

# ---------------------------------------------------------------------------
# Third-body gravitational parameters (km³/s²)
# ---------------------------------------------------------------------------
SUN_MU_KM3_S2: float = 1.32712440018e11
MOON_MU_KM3_S2: float = 4902.800066
SUN_EARTH_DISTANCE_KM: float = 1.496e8  # ~1 AU
MOON_EARTH_DISTANCE_KM: float = 384400.0

# ---------------------------------------------------------------------------
# Atmospheric drag model (exponential)
# ---------------------------------------------------------------------------
DRAG_REF_DENSITY_KG_M3: float = 3.725e-12  # reference at 400 km
DRAG_REF_ALTITUDE_KM: float = 400.0
DRAG_SCALE_HEIGHT_KM: float = 58.515  # scale height for ~400 km

# ---------------------------------------------------------------------------
# Orbital regime boundaries (km altitude above Earth's surface)
# ---------------------------------------------------------------------------
LEO_MIN_KM: float = 160.0
LEO_MAX_KM: float = 2000.0
MEO_MIN_KM: float = 2000.0
MEO_MAX_KM: float = 35786.0
GEO_ALTITUDE_KM: float = 35786.0
HEO_MIN_KM: float = 35786.0

# ---------------------------------------------------------------------------
# Conjunction thresholds
# ---------------------------------------------------------------------------
CONJUNCTION_THRESHOLD_KM: float = 5.0      # fine-grained detection
COARSE_FILTER_THRESHOLD_KM: float = 50.0   # coarse pre-filter

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
