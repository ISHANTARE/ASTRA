"""ASTRA Core package.

A pure Python computation engine for orbital analysis, conjunctions, 
and spatial filtering.
"""
from astra.conjunction import (
    closest_approach,
    distance_3d,
    find_conjunctions,
)
from astra.covariance import (
    compute_collision_probability,
    estimate_covariance,
)
from astra.cdm import parse_cdm_xml, ConjunctionDataMessage
from astra.data import (
    fetch_celestrak_active,
    fetch_celestrak_comprehensive,
    fetch_celestrak_group,
)
from astra.debris import (
    apply_filters,
    catalog_statistics,
    filter_altitude,
    filter_region,
    filter_time_window,
    make_debris_object,
)
from astra.errors import (
    AstraError,
    CoordinateError,
    FilterError,
    InvalidTLEError,
    PropagationError,
)
from astra.models import (
    ConjunctionEvent,
    DebrisObject,
    FilterConfig,
    Observer,
    OrbitalState,
    PassEvent,
    SatelliteTLE,
)
from astra.orbit import (
    ground_track,
    propagate_many,
    propagate_orbit,
    propagate_trajectory,
)
from astra.plot import plot_trajectories
from astra.time import convert_time
from astra.tle import load_tle_catalog, parse_tle, validate_tle
from astra.utils import haversine_distance, orbit_period, orbital_elements
from astra.visibility import passes_over_location, visible_from_location

__all__ = [
    # TLE
    "parse_tle",
    "validate_tle",
    "load_tle_catalog",
    # Orbit
    "propagate_orbit",
    "propagate_many",
    "propagate_trajectory",
    "ground_track",
    # Debris
    "filter_altitude",
    "filter_region",
    "filter_time_window",
    "catalog_statistics",
    "make_debris_object",
    "apply_filters",
    # Conjunction
    "distance_3d",
    "closest_approach",
    "find_conjunctions",
    "compute_collision_probability",
    "estimate_covariance",
    # Data
    "fetch_celestrak_active",
    "fetch_celestrak_comprehensive",
    "fetch_celestrak_group",
    "parse_cdm_xml",
    # Plotting
    "plot_trajectories",
    # Visibility
    "visible_from_location",
    "passes_over_location",
    # Time
    "convert_time",
    # Utils
    "haversine_distance",
    "orbital_elements",
    "orbit_period",
    # Models
    "SatelliteTLE",
    "OrbitalState",
    "DebrisObject",
    "ConjunctionEvent",
    "Observer",
    "PassEvent",
    "FilterConfig",
    "ConjunctionDataMessage",
    # Errors
    "AstraError",
    "InvalidTLEError",
    "PropagationError",
    "FilterError",
    "CoordinateError",
]
