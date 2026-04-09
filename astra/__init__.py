"""ASTRA Core package.

A pure Python computation engine for orbital analysis, conjunctions,
and spatial filtering.

**ASTRA-Core Engine** (*astra-core-engine*) — Copyright (c) 2026 Ishan Tare.
This file is part of ASTRA. If you use or adapt this code, retain this
copyright notice and provide attribution.

Supported Orbital Data Formats
--------------------------------
ASTRA-Core natively supports both the legacy TLE format and the modern
CCSDS OMM (Orbit Mean-Elements Message) standard. All physics functions
(propagation, conjunction analysis, filtering) accept both formats
transparently through the ``SatelliteState`` union type.

    ``SatelliteTLE``  — Legacy Two-Line Element strings. Fast, ubiquitous,
                        limited metadata.
    ``SatelliteOMM``  — Modern JSON format. Includes mass, RCS, and ballistic
                        coefficient. Recommended for new workflows.

Data Sources
------------
    CelesTrak (no account required):
        ``fetch_celestrak_group("starlink")``           → list[SatelliteTLE]
        ``fetch_celestrak_group_omm("starlink")``       → list[SatelliteOMM]

    Space-Track.org (free account required):
        ``fetch_spacetrack_group("starlink")``          → list[SatelliteOMM]
        Set env vars: SPACETRACK_USER, SPACETRACK_PASS
"""

from astra.version import __version__

# ---------------------------------------------------------------------------
# Conjunction Analysis
# ---------------------------------------------------------------------------
from astra.conjunction import (
    closest_approach,
    distance_3d,
    find_conjunctions,
    load_spacebook_covariance,
)

# ---------------------------------------------------------------------------
# Covariance & Collision Probability
# ---------------------------------------------------------------------------
from astra.covariance import (
    compute_collision_probability,
    compute_collision_probability_mc,
    estimate_covariance,
    propagate_covariance_stm,
    rotate_covariance_rtn_to_eci,
)

# ---------------------------------------------------------------------------
# CDM (Conjunction Data Message)
# ---------------------------------------------------------------------------
from astra.cdm import parse_cdm_xml, ConjunctionDataMessage

# ---------------------------------------------------------------------------
# Data Ingestion: CelesTrak (no account required)
# ---------------------------------------------------------------------------
from astra.data import (
    # TLE format (legacy, default)
    fetch_celestrak_active,
    fetch_celestrak_comprehensive,
    fetch_celestrak_group,
    # OMM format (modern, high-fidelity) — explicit discoverable names
    fetch_celestrak_active_omm,
    fetch_celestrak_comprehensive_omm,
    fetch_celestrak_group_omm,
)

# ---------------------------------------------------------------------------
# Data Ingestion: Space-Track.org (authenticated, free account required)
# ---------------------------------------------------------------------------
from astra.spacetrack import (
    fetch_spacetrack_active,
    fetch_spacetrack_group,
    fetch_spacetrack_satcat,
    spacetrack_logout,
)

# ---------------------------------------------------------------------------
# Data Ingestion: Spacebook (unauthenticated, COMSPOC)
# ---------------------------------------------------------------------------
from astra.spacebook import (
    fetch_xp_tle_catalog,
    fetch_historical_tle,
    fetch_tle_catalog,
    fetch_synthetic_covariance_stk,
    fetch_satcat_details,
    get_space_weather_sb,
    get_eop_sb,
)

# ---------------------------------------------------------------------------
# Debris Catalog Filtering
# ---------------------------------------------------------------------------
from astra.debris import (
    apply_filters,
    catalog_statistics,
    filter_altitude,
    filter_region,
    filter_time_window,
    make_debris_object,
)

# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------
from astra.errors import (
    AstraError,
    CoordinateError,
    EphemerisError,
    FilterError,
    InvalidTLEError,
    ManeuverError,
    PropagationError,
    SpaceWeatherError,
    SpacebookError,
    SpacebookLookupError,
)

# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------
from astra.models import (
    ConjunctionEvent,
    DebrisObject,
    FilterConfig,
    FiniteBurn,
    ManeuverFrame,
    Observer,
    OrbitalState,
    PassEvent,
    SatelliteTLE,
    SatelliteOMM,       # Modern OMM data model
    SatelliteState,     # Union[SatelliteTLE, SatelliteOMM] — accepted by all physics functions
    TrajectoryMap,
    VelocityMap,
    projected_area_m2,
)

# ---------------------------------------------------------------------------
# OMM Parser (modern format — recommended for new workflows)
# ---------------------------------------------------------------------------
from astra.omm import (
    load_omm_file,       # Load OMM JSON from local disk
    parse_omm_json,      # Parse OMM JSON string → list[SatelliteOMM]
    parse_omm_record,    # Parse single OMM dict → SatelliteOMM
    validate_omm,        # Validate OMM dict sanity before parsing
    xptle_to_satellite_omm, # Convert XP-TLE format into high-fidelity OMM
)

# ---------------------------------------------------------------------------
# STK Ephemeris Parser (Spacebook Synthetic Covariance output)
# ---------------------------------------------------------------------------
from astra.ocm import parse_stk_ephemeris

# ---------------------------------------------------------------------------
# TLE Parser (legacy format — backwards compatible)
# ---------------------------------------------------------------------------
from astra.tle import load_tle_catalog, parse_tle, validate_tle

# ---------------------------------------------------------------------------
# Orbit Propagation (accepts both TLE and OMM)
# ---------------------------------------------------------------------------
from astra.orbit import (
    ground_track,
    propagate_many,
    propagate_many_generator,
    propagate_orbit,
    propagate_trajectory,
)

# ---------------------------------------------------------------------------
# High-Fidelity Numerical Propagation
# ---------------------------------------------------------------------------
from astra.propagator import (
    NumericalState,
    DragConfig,
    propagate_cowell,
)

# ---------------------------------------------------------------------------
# Maneuver Planning
# ---------------------------------------------------------------------------
from astra.maneuver import (
    rotation_vnb_to_inertial,
    rotation_rtn_to_inertial,
    frame_to_inertial,
    thrust_acceleration_inertial,
    validate_burn,
    validate_burn_sequence,
)

# ---------------------------------------------------------------------------
# Space Weather & Atmospheric Pipeline
# ---------------------------------------------------------------------------
from astra.data_pipeline import (
    sun_position_de,
    sun_position_teme,
    moon_position_de,
    moon_position_teme,
    get_space_weather,
    load_space_weather,
    atmospheric_density_empirical,
)

# ---------------------------------------------------------------------------
# Visualization & Utilities
# ---------------------------------------------------------------------------
from astra.time import convert_time
from astra.utils import vincenty_distance, orbit_period, orbital_elements
from astra.visibility import passes_over_location, visible_from_location
from astra.spatial_index import SpatialIndex


# ---------------------------------------------------------------------------
# Public API Surface (__all__)
# ---------------------------------------------------------------------------
__all__ = [
    # --- TLE Parsing (legacy format) ---
    "parse_tle",
    "validate_tle",
    "load_tle_catalog",

    # --- OMM Parsing (modern format, recommended) ---
    "parse_omm_record",
    "parse_omm_json",
    "load_omm_file",
    "validate_omm",
    "xptle_to_satellite_omm",

    # --- STK Ephemeris Parser (Spacebook) ---
    "parse_stk_ephemeris",

    # --- Data Models ---
    "SatelliteTLE",
    "SatelliteOMM",
    "SatelliteState",
    "OrbitalState",
    "DebrisObject",
    "ConjunctionEvent",
    "Observer",
    "PassEvent",
    "FilterConfig",
    "ConjunctionDataMessage",

    # --- Data Ingestion: CelesTrak (no account required) ---
    "fetch_celestrak_active",           # → list[SatelliteTLE]
    "fetch_celestrak_group",            # → list[SatelliteTLE]
    "fetch_celestrak_comprehensive",    # → list[SatelliteTLE]
    "fetch_celestrak_active_omm",       # → list[SatelliteOMM]
    "fetch_celestrak_group_omm",        # → list[SatelliteOMM]
    "fetch_celestrak_comprehensive_omm",# → list[SatelliteOMM]

    # --- Data Ingestion: Space-Track.org (authenticated) ---
    "fetch_spacetrack_group",           # → list[SatelliteOMM] or list[SatelliteTLE]
    "fetch_spacetrack_active",          # → list[SatelliteOMM] or list[SatelliteTLE]
    "fetch_spacetrack_satcat",          # → list[dict]
    "spacetrack_logout",

    # --- Data Ingestion: Spacebook (unauthenticated) ---
    "fetch_xp_tle_catalog",             # → list[SatelliteTLE]
    "fetch_historical_tle",             # → list[SatelliteTLE]
    "fetch_tle_catalog",                # → list[SatelliteTLE]
    "fetch_synthetic_covariance_stk",   # → str
    "fetch_satcat_details",             # → dict
    "get_space_weather_sb",
    "get_eop_sb",

    # --- Orbit Propagation (✓ TLE | ✓ OMM) ---
    "propagate_orbit",
    "propagate_many",
    "propagate_many_generator",
    "propagate_trajectory",
    "ground_track",

    # --- Debris Catalog Filtering (✓ TLE | ✓ OMM) ---
    "filter_altitude",
    "filter_region",
    "filter_time_window",
    "catalog_statistics",
    "make_debris_object",
    "apply_filters",

    # --- Conjunction Analysis (✓ TLE | ✓ OMM) ---
    "distance_3d",
    "closest_approach",
    "find_conjunctions",
    "load_spacebook_covariance",
    "compute_collision_probability",
    "estimate_covariance",

    # --- CDM ---
    "parse_cdm_xml",

    # --- Visualization ---
    "plot_trajectories",

    # --- Visibility ---
    "visible_from_location",
    "passes_over_location",

    # --- Time & Utils ---
    "convert_time",
    "vincenty_distance",
    "orbital_elements",
    "orbit_period",

    # --- High-Fidelity Numerical Engine ---
    "compute_collision_probability_mc",
    "propagate_covariance_stm",
    "rotate_covariance_rtn_to_eci",
    "propagate_cowell",
    "SpatialIndex",

    # --- Maneuver & High-Fidelity Physics ---
    "ManeuverFrame",
    "FiniteBurn",
    "ManeuverError",
    "rotation_vnb_to_inertial",
    "rotation_rtn_to_inertial",
    "frame_to_inertial",
    "thrust_acceleration_inertial",
    "validate_burn",

    # --- Space Weather ---
    "sun_position_de",
    "moon_position_de",
    "get_space_weather",
    "load_space_weather",
    "atmospheric_density_empirical",

    # --- Errors ---
    "AstraError",
    "InvalidTLEError",
    "PropagationError",
    "FilterError",
    "CoordinateError",
    "ManeuverError",
    "SpaceWeatherError",
    "EphemerisError",
    "SpacebookError",
    "SpacebookLookupError",

    # --- Config & Mode Control ---
    "set_strict_mode",
    "validate_burn_sequence",
    "NumericalState",
    "DragConfig",
    "projected_area_m2",
]

import sys
from . import config
from astra.config import set_strict_mode


def __getattr__(name: str):
    """Lazy-load optional dependencies (e.g. Plotly for ``plot_trajectories``)."""
    if name == "plot_trajectories":
        try:
            from astra.plot import plot_trajectories as _plot_traj
        except ImportError as exc:
            raise ImportError(
                "plot_trajectories requires Plotly. Install with: "
                "pip install 'astra-core-engine[viz]' or pip install plotly>=5.18"
            ) from exc
        globals()["plot_trajectories"] = _plot_traj
        return _plot_traj
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


_BANNER_SHOWN = False

def _show_banner() -> None:
    """Print ASTRA startup banner to stderr. Fires once per process."""
    global _BANNER_SHOWN
    if _BANNER_SHOWN:
        return
    mode = "STRICT (Flight-Grade)" if config.ASTRA_STRICT_MODE else "Relaxed (Beginner-Friendly)"
    print(f"[ASTRA-Core v{__version__}] Mode: {mode}", file=sys.stderr)
    if not config.ASTRA_STRICT_MODE:
        print("[ASTRA-Core] -> Missing data will be estimated with warnings.", file=sys.stderr)
        print("[ASTRA-Core] -> Flight-grade: astra.config.ASTRA_STRICT_MODE = True", file=sys.stderr)
    print("[ASTRA-Core] -> Cache: ~/.astra/data | Creds: SPACETRACK_USER / SPACETRACK_PASS", file=sys.stderr)
    _BANNER_SHOWN = True

_show_banner()
