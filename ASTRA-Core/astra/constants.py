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
EARTH_MU_KM3_S2: float = 398600.4418  # gravitational parameter (km³/s²)

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
