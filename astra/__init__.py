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
    compute_collision_probability_mc,
    estimate_covariance,
    propagate_covariance_stm,
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
    ManeuverError,
    PropagationError,
)
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
    projected_area_m2,
)
from astra.orbit import (
    ground_track,
    propagate_many,
    propagate_many_generator,
    propagate_orbit,
    propagate_trajectory,
)
from astra.propagator import (
    NumericalState,
    DragConfig,
    propagate_cowell,
)
from astra.maneuver import (
    rotation_vnb_to_inertial,
    rotation_rtn_to_inertial,
    frame_to_inertial,
    thrust_acceleration_inertial,
    validate_burn,
)
from astra.data_pipeline import (
    sun_position_de,
    moon_position_de,
    get_space_weather,
    load_space_weather,
    atmospheric_density_empirical,
)
from astra.plot import plot_trajectories
from astra.time import convert_time
from astra.tle import load_tle_catalog, parse_tle, validate_tle
from astra.utils import haversine_distance, orbit_period, orbital_elements
from astra.visibility import passes_over_location, visible_from_location
from astra.spatial_index import SpatialIndex

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
    # Sterling Architecture
    "compute_collision_probability_mc",
    "propagate_covariance_stm",
    "propagate_cowell",
    "NumericalState",
    "DragConfig",
    "projected_area_m2",
    "propagate_many_generator",
    "SpatialIndex",
    # Maneuver & High-Fidelity Physics
    "ManeuverFrame",
    "FiniteBurn",
    "ManeuverError",
    "rotation_vnb_to_inertial",
    "rotation_rtn_to_inertial",
    "frame_to_inertial",
    "thrust_acceleration_inertial",
    "validate_burn",
    "sun_position_de",
    "moon_position_de",
    "get_space_weather",
    "load_space_weather",
    "atmospheric_density_empirical",
]
