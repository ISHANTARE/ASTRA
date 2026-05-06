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
    find_conjunction_windows,
    ConjunctionWindow,
    load_spacebook_covariance,
    run_conjunction_sweep,
)

# ---------------------------------------------------------------------------
# Covariance & Collision Probability
# ---------------------------------------------------------------------------
from astra.covariance import (
    compute_collision_probability,
    compute_collision_probability_mc,
    compute_collision_probability_timeseries,
    estimate_covariance,
    propagate_covariance_stm,
    rotate_covariance_rtn_to_eci,
)

# ---------------------------------------------------------------------------
# CDM (Conjunction Data Message)
# ---------------------------------------------------------------------------
from astra.cdm import parse_cdm_xml, export_cdm_xml, parse_cdm_kvn, export_cdm_kvn, ConjunctionDataMessage

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
    refresh_satcat_cache,
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
    SGP4ErrorCode,  # Enum for SGP4 error codes with descriptions
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
    SatelliteOMM,  # Modern OMM data model
    SatelliteState,  # Union[SatelliteTLE, SatelliteOMM] — accepted by all physics functions
    TrajectoryMap,
    VelocityMap,
    projected_area_m2,
)

# ---------------------------------------------------------------------------
# OMM Parser (modern format — recommended for new workflows)
# ---------------------------------------------------------------------------
from astra.omm import (
    load_omm_file,  # Load OMM JSON from local disk
    parse_omm_json,  # Parse OMM JSON string → list[SatelliteOMM]
    parse_omm_record,  # Parse single OMM dict → SatelliteOMM
    validate_omm,  # Validate OMM dict sanity before parsing
    xptle_to_satellite_omm,  # Convert list[SatelliteTLE] XP-TLEs into OMMs
)

# ---------------------------------------------------------------------------
# OCM Parser (CCSDS Orbit Comprehensive Message — XML and KVN)
# ---------------------------------------------------------------------------
from astra.ocm import (
    parse_stk_ephemeris,   # Spacebook STK DotE → list[NumericalState]
    parse_ocm,             # Auto-detect XML vs KVN → list[NumericalState]
    parse_ocm_xml,         # CCSDS OCM XML → list[NumericalState]
    parse_ocm_kvn,         # CCSDS OCM KVN → list[NumericalState]
    export_ocm_xml,        # list[NumericalState] → CCSDS OCM XML string
)

# ---------------------------------------------------------------------------
# Coordinate Frame Transforms
# ---------------------------------------------------------------------------
from astra.frames import (
    teme_to_ecef,               # TEME → ECEF with optional Spacebook EOP correction
    ecef_to_geodetic_wgs84,     # ECEF (km) → (lat_deg, lon_deg, alt_km) WGS-84
    geodetic_to_ecef_wgs84,     # (lat_deg, lon_deg, alt_km) → ECEF (km) WGS-84
    get_eop_correction,         # Batch Spacebook EOP fetch for propagation time grids
)

# ---------------------------------------------------------------------------
# TLE Parser (legacy format — backwards compatible)
# ---------------------------------------------------------------------------
from astra.tle import load_tle_catalog, parse_tle, validate_tle

# ---------------------------------------------------------------------------
# Julian Date / Datetime Utilities
# ---------------------------------------------------------------------------
from astra.jdutil import (
    jd_utc_to_datetime,     # Julian Date → UTC-aware datetime
    datetime_utc_to_jd,     # UTC datetime → Julian Date
    jd_to_datetime,         # Alias for jd_utc_to_datetime
    datetime_to_jd,         # Alias for datetime_utc_to_jd
)

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
    SNCConfig,
    propagate_cowell,
    propagate_cowell_at_times,
    propagate_cowell_batch,
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
    plan_hohmann,
    plan_bielliptic,
    plan_inclination_change,
    compute_delta_v_budget,
    DeltaVBudget,
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
from astra.time import convert_time, prefetch_iers_data_async
from astra.utils import vincenty_distance, orbit_period, orbital_elements
from astra.visibility import passes_over_location, visible_from_location
from astra.spatial_index import SpatialIndex
from astra import constants  # expose as astra.constants.*

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
    # --- OCM Parser (CCSDS 502.0-B-2 — XML and KVN) ---
    "parse_stk_ephemeris",
    "parse_ocm",
    "parse_ocm_xml",
    "parse_ocm_kvn",
    "export_ocm_xml",
    # --- Data Models ---
    "SatelliteTLE",
    "SatelliteOMM",
    "SatelliteState",
    "OrbitalState",
    "DebrisObject",
    "ConjunctionEvent",
    "ConjunctionDataMessage",
    "Observer",
    "PassEvent",
    "FilterConfig",
    # --- Data Ingestion: CelesTrak (no account required) ---
    "fetch_celestrak_active",  # → list[SatelliteTLE]
    "fetch_celestrak_group",  # → list[SatelliteTLE]
    "fetch_celestrak_comprehensive",  # → list[SatelliteTLE]
    "fetch_celestrak_active_omm",  # → list[SatelliteOMM]
    "fetch_celestrak_group_omm",  # → list[SatelliteOMM]
    "fetch_celestrak_comprehensive_omm",  # → list[SatelliteOMM]
    # --- Data Ingestion: Space-Track.org (authenticated) ---
    "fetch_spacetrack_group",  # → list[SatelliteOMM] or list[SatelliteTLE]
    "fetch_spacetrack_active",  # → list[SatelliteOMM] or list[SatelliteTLE]
    "fetch_spacetrack_satcat",  # → list[dict]
    "spacetrack_logout",
    # --- Data Ingestion: Spacebook (unauthenticated) ---
    "fetch_xp_tle_catalog",  # → list[SatelliteTLE]
    "fetch_historical_tle",  # → list[SatelliteTLE]
    "fetch_tle_catalog",  # → list[SatelliteTLE]
    "fetch_synthetic_covariance_stk",  # → str
    "fetch_satcat_details",  # → dict
    "refresh_satcat_cache",  # → int
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
    "find_conjunction_windows",
    "ConjunctionWindow",
    "run_conjunction_sweep",
    "load_spacebook_covariance",
    "compute_collision_probability",
    "estimate_covariance",
    # --- CDM ---
    "parse_cdm_xml",
    "export_cdm_xml",
    "parse_cdm_kvn",
    "export_cdm_kvn",
    # --- Visualization ---
    "plot_trajectories",
    "plot_ground_track",
    # --- Visibility ---
    "visible_from_location",
    "passes_over_location",
    # --- Time & Datetime ---
    "convert_time",
    "prefetch_iers_data_async",
    "jd_utc_to_datetime",
    "datetime_utc_to_jd",
    "jd_to_datetime",
    "datetime_to_jd",
    # --- Coordinate Frame Transforms ---
    "teme_to_ecef",
    "ecef_to_geodetic_wgs84",
    "geodetic_to_ecef_wgs84",
    "get_eop_correction",
    # --- General Utilities ---
    "vincenty_distance",
    "orbital_elements",
    "orbit_period",
    # --- High-Fidelity Numerical Engine ---
    "compute_collision_probability_mc",
    "compute_collision_probability_timeseries",
    "propagate_covariance_stm",
    "rotate_covariance_rtn_to_eci",
    "propagate_cowell",
    "propagate_cowell_at_times",
    "propagate_cowell_batch",
    "SpatialIndex",
    # --- Maneuver & High-Fidelity Physics ---
    "ManeuverFrame",
    "FiniteBurn",
    "rotation_vnb_to_inertial",
    "rotation_rtn_to_inertial",
    "frame_to_inertial",
    "thrust_acceleration_inertial",
    "validate_burn",
    "plan_hohmann",
    "plan_bielliptic",
    "plan_inclination_change",
    "compute_delta_v_budget",
    "DeltaVBudget",
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
    "SGP4ErrorCode",  # Enum for SGP4 error codes
    # --- Config & Mode Control ---
    "set_strict_mode",
    "set_spacebook_enabled",
    "validate_burn_sequence",
    "NumericalState",
    "DragConfig",
    "SNCConfig",
    "projected_area_m2",
    "TrajectoryMap",
    "VelocityMap",
    "sun_position_teme",
    "moon_position_teme",
    "warmup",
    "help",
]

import sys
from . import config
from astra.config import set_strict_mode
from astra.config import set_spacebook_enabled
from typing import Any
import numpy as np
import threading  # noqa: E402


def warmup() -> None:
    """Pre-compile Numba JIT kernels to eliminate first-run latency.

    Performs a trivial 1-second propagation and conjunction check to
    trigger binary code generation for the numerical integrator, drag
    model, and spatial index. Highly recommended for production workers.
    """
    from astra.propagator import propagate_cowell, NumericalState
    from astra.conjunction import find_conjunctions
    from astra.models import DebrisObject

    # 1. Warm-up Propagator (Numba Cowell kernel)
    p0 = np.array([7000.0, 0.0, 0.0])
    v0 = np.array([0.0, 7.5, 0.0])
    state = NumericalState(t_jd=2460000.5, position_km=p0, velocity_km_s=v0)
    _ = propagate_cowell(state, duration_s=1.0, dt_out=1.0)

    # 2. Warm-up Conjunction Screening (Numba and SpatialIndex)
    from astra.models import SatelliteTLE
    tle = SatelliteTLE(
        norad_id="WARMUP",
        name="WARMUP",
        line1="1 99999U 99999A   26100.00000000  .00000000  00000-0  00000-0 0  9999",
        line2="2 99999   0.0000   0.0000 0000000   0.0000   0.0000 15.00000000    00",
        epoch_jd=2460000.5,
        object_type="PAYLOAD"
    )
    obj = DebrisObject(
        source=tle,
        altitude_km=622.0,
        inclination_deg=0.0,
        period_minutes=96.0,
        raan_deg=0.0,
        eccentricity=0.0,
        apogee_km=622.0,
        perigee_km=622.0,
        object_class="PAYLOAD",
        radius_m=5.0
    )
    # BL-06: Use two nearby but non-identical trajectories with a real close approach
    # (<threshold_km) to exercise the full conjunction pipeline including Brent
    # minimizer and Pc computation. The old code used identical positions (distance=0)
    # which tripped the `min_dist > threshold_km` gate and never warmed the hot paths.
    _r0 = np.array([7000.0, 0.0, 0.0])
    _r1 = np.array([7000.3, 0.0, 0.0])   # 300 m separation — inside threshold
    traj = {
        "WARM1": np.array([_r0, _r0 + np.array([0.1, 0.0, 0.0]), _r0 + np.array([0.2, 0.0, 0.0])]),
        "WARM2": np.array([_r1, _r1 + np.array([0.1, 0.0, 0.0]), _r1 + np.array([0.2, 0.0, 0.0])]),
    }
    times = np.array([2460000.5, 2460000.5 + 1.0/86400.0, 2460000.5 + 2.0/86400.0])
    _ = find_conjunctions(traj, times, { "WARM1": obj, "WARM2": obj }, threshold_km=1.0)


def _plotly_import_error(name: str, exc: ImportError) -> ImportError:
    return ImportError(
        f"{name} requires Plotly. Install with: "
        "pip install 'astra-core-engine[viz]' or pip install plotly>=5.18"
    )


def plot_trajectories(*args: Any, **kwargs: Any) -> Any:
    """Create an interactive Plotly 3-D trajectory figure.

    Args:
        *args: Positional arguments passed to :func:`astra.plot.plot_trajectories`.
        **kwargs: Keyword arguments passed to :func:`astra.plot.plot_trajectories`.

    Returns:
        A Plotly ``go.Figure``.

    Raises:
        ImportError: If Plotly is not installed. Install ``astra-core-engine[viz]``.

    Example::

        import astra
        fig = astra.plot_trajectories({"25544": positions_km})
    """
    try:
        from astra.plot import plot_trajectories as _plot_traj
    except ImportError as exc:
        raise _plotly_import_error("plot_trajectories", exc) from exc
    return _plot_traj(*args, **kwargs)


def plot_ground_track(*args: Any, **kwargs: Any) -> Any:
    """Create an interactive Plotly ground-track figure.

    Args:
        *args: Positional arguments passed to :func:`astra.plot.plot_ground_track`.
        **kwargs: Keyword arguments passed to :func:`astra.plot.plot_ground_track`.

    Returns:
        A Plotly ``go.Figure``.

    Raises:
        ImportError: If Plotly is not installed. Install ``astra-core-engine[viz]``.

    Example::

        import astra
        fig = astra.plot_ground_track(satellite, t_start_jd, t_end_jd)
    """
    try:
        from astra.plot import plot_ground_track as _plot_gt
    except ImportError as exc:
        raise _plotly_import_error("plot_ground_track", exc) from exc
    return _plot_gt(*args, **kwargs)


def __getattr__(name: str) -> Any:
    """Return lazily provided optional attributes."""
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


import os  # noqa: E402

_BANNER_SHOWN = False
_BANNER_LOCK = threading.Lock()


def _show_banner() -> None:
    """Print ASTRA startup banner to stderr. Fires once per process.

    AUDIT-F-03 Fix: Suppressed when ``ASTRA_NO_BANNER=1`` is set, preventing
    log pollution in production worker pools where each subprocess would
    otherwise emit the banner independently.
    
    AUDIT-FIX: Thread-safe banner using Lock to prevent race conditions
    in multi-threaded environments where multiple threads might try to
    show the banner simultaneously.
    """
    global _BANNER_SHOWN
    with _BANNER_LOCK:
        if _BANNER_SHOWN:
            return
        if os.environ.get("ASTRA_NO_BANNER", "0").strip() == "1":
            _BANNER_SHOWN = True
            return
        mode = (
            "STRICT (Flight-Grade)"
            if config.ASTRA_STRICT_MODE
            else "Relaxed (Beginner-Friendly)"
        )
        print(f"[ASTRA-Core v{__version__}] Mode: {mode}", file=sys.stderr)
        if not config.ASTRA_STRICT_MODE:
            print(
                "[ASTRA-Core] -> Missing data will be estimated with warnings.",
                file=sys.stderr,
            )
            print(
                "[ASTRA-Core] -> Flight-grade: astra.config.ASTRA_STRICT_MODE = True",
                file=sys.stderr,
            )
        print(
            "[ASTRA-Core] -> Cache: ~/.astra/data | Creds: SPACETRACK_USER / SPACETRACK_PASS",
            file=sys.stderr,
        )
        _BANNER_SHOWN = True


def help(topic: str = "") -> None:
    """Print help for ASTRA-Core API discovery.
    
    Args:
        topic: Optional topic to get specific help. One of:
            - "" (empty): Show all categories and key functions
            - "propagation": Orbit propagation functions
            - "conjunction": Conjunction analysis functions  
            - "visibility": Pass prediction functions
            - "data": Data fetching functions
            - "frames": Coordinate frame transforms
            - "maneuver": Maneuver planning functions
            - "config": Configuration options
            - "errors": Exception hierarchy
            - "env": Environment variables
    
    Examples:
        >>> import astra
        >>> astra.help()               # Show all categories
        >>> astra.help("propagation")  # Show propagation functions
        >>> astra.help("env")          # Show environment variables
    """
    _HELP_TEXT = {
        "": """
================================================================================
                        ASTRA-Core v{version} Quick Reference
================================================================================

STRICT MODE IS ON BY DEFAULT (v3.6.1+)
  To disable: astra.config.ASTRA_STRICT_MODE = False

CATEGORIES:
  propagation  - Orbit propagation (SGP4, Cowell)
  conjunction  - Conjunction analysis & collision probability  
  visibility   - Pass prediction over ground stations
  data         - Fetch TLE/OMM from CelesTrak, Space-Track, Spacebook
  frames       - Coordinate frame transforms (TEME, ECEF, geodetic)
  maneuver     - Delta-V planning & finite burns
  config       - Runtime configuration
  errors       - Exception hierarchy
  env          - Environment variables

USAGE:
  astra.help("propagation")  # Get help on a specific topic
  astra.help("env")          # See all environment variables

QUICK START:
  import astra
  
  # Fetch TLEs
  catalog = astra.fetch_celestrak_active()
  
  # Propagate
  times = astra.datetime_utc_to_jd(datetime.now(timezone.utc)) + np.arange(0, 1440, 5)/1440
  traj, vel = astra.propagate_many([catalog[0]], times)
  
  # Predict passes
  obs = astra.Observer("NYC", 40.7, -74.0, 10.0)
  passes = astra.passes_over_location(catalog[0], obs, times[0], times[-1])

================================================================================
""",
        "propagation": """
PROPAGATION FUNCTIONS:

SGP4 (Fast, large catalogs):
  propagate_orbit(satellite, epoch_jd, t_since_minutes) -> OrbitalState
  propagate_many(satellites, times_jd) -> (TrajectoryMap, VelocityMap)
  propagate_trajectory(satellite, t_start_jd, t_end_jd) -> (times, pos, vel)
  ground_track(positions_teme, times_jd) -> [(lat, lon, alt), ...]

Cowell (High-fidelity, single satellite):
  propagate_cowell(state0, duration_s, dt_out, drag_config, maneuvers)
  propagate_cowell_at_times(state0, times_jd, ...) -> list[NumericalState]
  propagate_cowell_batch(states_dict, ...) -> dict[id, list[NumericalState]]

DATA TYPES:
  SatelliteTLE  - Legacy TLE format
  SatelliteOMM   - Modern OMM format (recommended)
  SatelliteState - Union[TLE, OMM] - accepted by all propagation functions

EXAMPLE:
  import astra, numpy as np
  from datetime import datetime, timezone
  
  tle = astra.parse_tle("ISS", line1, line2)
  t_jd = astra.datetime_utc_to_jd(datetime.now(timezone.utc))
  state = astra.propagate_orbit(tle, tle.epoch_jd, (t_jd - tle.epoch_jd) * 1440)
  print(state.position_km)  # [x, y, z] in TEME frame
""",
        "conjunction": """
CONJUNCTION ANALYSIS:

  find_conjunctions(trajectories, times_jd, elements_map, threshold_km, 
                    max_workers=4) -> list[ConjunctionEvent]
  run_conjunction_sweep(catalog, t_start_jd, t_end_jd, threshold_km) -> list[ConjunctionEvent]
  closest_approach(traj_a, traj_b, times_jd) -> (min_dist_km, tca_jd, idx)
  distance_3d(pos_a, pos_b) -> distances

COLLISION PROBABILITY:
  compute_collision_probability(miss_vec, rel_vel, cov_a, cov_b, 
                                radius_a, radius_b) -> float
  compute_collision_probability_mc(...) -> float  # Monte Carlo
  estimate_covariance(days_since_epoch) -> 3x3 covariance (heuristic)

COVARIANCE:
  load_spacebook_covariance(norad_id) -> 6x6 covariance from COMSPOC
  propagate_covariance_stm(cov0, Phi) -> propagated covariance

PARALLELISM:
  find_conjunctions(..., max_workers=8)  # Control thread pool size
""",
        "visibility": """
PASS PREDICTION:

  passes_over_location(satellite, observer, t_start_jd, t_end_jd, 
                       step_minutes=1.0) -> list[PassEvent]
  visible_from_location(positions_teme, times_jd, observer) -> elevations_deg

OBSERVER:
  obs = astra.Observer(
      name="Station",
      latitude_deg=40.7,      # NOT lat_deg!
      longitude_deg=-74.0,    # NOT lon_deg!
      elevation_m=10.0,
      min_elevation_deg=10.0
  )

PASS EVENT ATTRIBUTES:
  PassEvent.aos_jd              # Rise time (Julian Date)
  PassEvent.tca_jd              # Time of closest approach
  PassEvent.los_jd              # Set time
  PassEvent.max_elevation_deg   # Maximum elevation
  PassEvent.azimuth_at_aos_deg  # Azimuth at rise
  PassEvent.azimuth_at_tca_deg  # Azimuth at peak
  PassEvent.azimuth_at_los_deg  # Azimuth at set
  PassEvent.duration_seconds
  PassEvent.satellite_illuminated  # True if sunlit at TCA
  PassEvent.observer_in_darkness   # True if observer in Earth shadow
""",
        "data": """
DATA FETCHING:

CELESTRAK (no account):
  fetch_celestrak_active() -> list[SatelliteTLE]
  fetch_celestrak_group("starlink") -> list[SatelliteTLE]
  fetch_celestrak_active_omm() -> list[SatelliteOMM]  # Modern format
  fetch_celestrak_group_omm("starlink") -> list[SatelliteOMM]

SPACE-TRACK (requires free account):
  Set env vars: SPACETRACK_USER, SPACETRACK_PASS
  fetch_spacetrack_active() -> list[SatelliteOMM]
  fetch_spacetrack_group("starlink") -> list[SatelliteOMM]
  spacetrack_logout()  # End session

SPACEBOOK/COMSPOC (no account):
  fetch_tle_catalog() -> list[SatelliteTLE]
  fetch_xp_tle_catalog() -> list[SatelliteTLE]  # High-precision XP-TLE
  fetch_synthetic_covariance_stk(norad_id) -> str  # STK ephemeris with covariance
  get_space_weather_sb(jd) -> (f107, f107_adj, ap)
  get_eop_sb(jd) -> (xp, yp, dut1)

PARSING:
  parse_tle(name, line1, line2) -> SatelliteTLE
  validate_tle(name, line1, line2) -> bool
  parse_omm_json(text) -> list[SatelliteOMM]
  load_omm_file(path) -> list[SatelliteOMM]
""",
        "frames": """
COORDINATE FRAMES:

  TEME (SGP4 output) -> ECEF (Earth-fixed):
    teme_to_ecef(positions_teme, times_jd, use_spacebook_eop=True) -> (N,3) ECEF km

  ECEF -> Geodetic:
    ecef_to_geodetic_wgs84(x, y, z) -> (lat_deg, lon_deg, alt_km)

  Geodetic -> ECEF:
    geodetic_to_ecef_wgs84(lat_deg, lon_deg, alt_km) -> (x, y, z) km

  EOP (Earth Orientation Parameters):
    get_eop_correction(times_jd) -> (xp, yp, dut1)

FRAME CHAIN:
  TLE --SGP4--> TEME --teme_to_ecef--> ECEF --ecef_to_geodetic--> Lat/Lon/Alt

EXAMPLE:
  pos_ecef = astra.teme_to_ecef(state.position_km[np.newaxis,:], 
                                 np.array([t_jd]))
  lat, lon, alt = astra.ecef_to_geodetic_wgs84(pos_ecef[0,0], 
                                                pos_ecef[0,1], 
                                                pos_ecef[0,2])
""",
        "maneuver": """
MANEUVER PLANNING:

IMPULSIVE DELTA-V:
  plan_hohmann(r1_km, r2_km, mass_kg, thrust_N, isp_s, 
               epoch_jd, initial_true_anomaly_deg) -> list[FiniteBurn]
  plan_bielliptic(r1_km, r2_km, rb_km, ...) -> list[FiniteBurn]
  plan_inclination_change(inc1_deg, inc2_deg, r_km, ...) -> FiniteBurn

DELTA-V BUDGET:
  compute_delta_v_budget(burns, initial_mass_kg) -> DeltaVBudget
  budget.total_delta_v_m_s  # Total delta-V in m/s
  budget.propellant_kg      # Total propellant used

VALIDATION:
  validate_burn(burn: FiniteBurn) -> bool
  validate_burn_sequence(burns: list[FiniteBurn]) -> bool

FINITE BURN ATTRIBUTES:
  FiniteBurn.epoch_ignition_jd  # Start time
  FiniteBurn.duration_s
  FiniteBurn.thrust_N
  FiniteBurn.isp_s
  FiniteBurn.direction  # Unit vector in VNB or RTN frame
  FiniteBurn.frame      # ManeuverFrame.VNB or ManeuverFrame.RTN
""",
        "config": """
CONFIGURATION:

STRICT MODE (ON by default since v3.6.1):
  astra.config.ASTRA_STRICT_MODE  # Current setting
  astra.set_strict_mode(True)     # Enable strict mode
  astra.set_strict_mode(False)    # Disable for relaxed mode

SPACEBOOK:
  astra.config.SPACEBOOK_ENABLED
  astra.set_spacebook_enabled(False)  # Disable Spacebook I/O

WHAT STRICT MODE DOES:
  - Raises EphemerisError if DE421 unavailable
  - Raises SpaceWeatherError if F10.7/Ap missing
  - Raises PropagationError for NaN trajectories
  - Rejects heuristic covariance in Pc computation
  - Validates TLE staleness (>30 days)

RELAXED MODE (non-strict):
  - Falls back to analytical Sun/Moon approximations
  - Uses synthetic space weather (F10.7=150, Ap=15)
  - Estimates covariance when CDM unavailable
  - Warns instead of raising on stale TLEs
""",
        "errors": """
EXCEPTION HIERARCHY:

  AstraError (base class)
  ├── InvalidTLEError      - Malformed TLE (checksum, format)
  ├── PropagationError     - SGP4 error code or NaN trajectory
  ├── EphemerisError       - DE421 ephemeris unavailable (strict mode)
  ├── SpaceWeatherError    - F10.7/Ap data unavailable (strict mode)
  ├── FilterError          - Invalid filter configuration
  ├── CoordinateError      - Frame transformation failure
  ├── ManeuverError        - Invalid burn sequence
  └── SpacebookError       - COMSPOC API failure
      └── SpacebookLookupError - NORAD ID not found

SGP4 ERROR CODES:
  from astra.errors import SGP4ErrorCode
  
  SGP4ErrorCode.OK                    # 0 - Success
  SGP4ErrorCode.MEAN_ELEMENTS_INVALID # 1
  SGP4ErrorCode.MEAN_MOTION_TOO_SMALL # 2
  SGP4ErrorCode.SEMIMAJOR_AXIS_NEG    # 3
  SGP4ErrorCode.ECCENTRICITY_INVALID  # 4
  SGP4ErrorCode.FUTURE_POSITION_ERROR # 5 - May have decayed
  SGP4ErrorCode.SATELLITE_DECAYED     # 6 - Below 156 km

EXAMPLE:
  state = astra.propagate_orbit(tle, ...)
  if state.error_code != astra.SGP4ErrorCode.OK:
      print(f"Error: {astra.SGP4ErrorCode(state.error_code).name}")
""",
        "env": """
ENVIRONMENT VARIABLES:

CACHE DIRECTORY:
  ASTRA_DATA_DIR=~/.astra/data
  # Where DE421 ephemeris, IERS finals, space weather CSV are cached
  # Default: ~/.astra/data/

STRICT MODE (overridden by code):
  ASTRA_STRICT_MODE=1   # Enable strict mode
  ASTRA_STRICT_MODE=0   # Disable strict mode

SPACEBOOK:
  ASTRA_SPACEBOOK_ENABLED=true   # Enable COMSPOC data sources
  ASTRA_SPACEBOOK_ENABLED=false  # Disable

BANNER:
  ASTRA_NO_BANNER=1   # Suppress startup banner

SPACE-TRACK CREDENTIALS:
  SPACETRACK_USER=your@email.com
  SPACETRACK_PASS=your_password
  # Required for fetch_spacetrack_* functions

PARALLELISM:
  ASTRA_MAX_WORKERS=8  # Default thread pool size for conjunction sweep

TESTING:
  ASTRA_TEST_CACHE_DIR=/tmp/astra-test  # Test cache location
""",
    }
    
    version = __version__ if '__version__' in dir() else "3.6.1"
    print(_HELP_TEXT.get(topic.lower(), _HELP_TEXT[""]).format(version=version))


_show_banner()
