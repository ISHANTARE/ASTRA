# astra/models.py
"""ASTRA Core data models.

All persistent data structures used across the library are defined here as
frozen dataclasses.  This is the single source of truth for all inter-module
data types — no other module defines dataclasses.

Every dataclass uses ``frozen=True`` to enforce immutability.
"""

from __future__ import annotations
from typing import Any

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Union

import numpy as np
from astra.constants import G0_STD as _G0_STD  # [FM-9 Fix — Finding #12/22]

# ---------------------------------------------------------------------------
# Type Aliases
# ---------------------------------------------------------------------------

# A map from NORAD ID to trajectory array.
# Shape of each array: (T, 3) where T = number of timesteps.
# Units: km, frame: TEME.
TrajectoryMap = dict[str, np.ndarray]

# NORAD ID → velocity array (same keys as TrajectoryMap).
# Shape of each array: (T, 3) where T = number of timesteps.
# Units: km/s, frame: TEME.
VelocityMap = dict[str, np.ndarray]

# A list of NORAD ID pairs (candidate conjunction pairs).
CandidatePairs = list[tuple[str, str]]

# Time array: array of Julian Dates.
TimeArray = np.ndarray  # shape (T,), dtype float64


def _quat_rotate(q: tuple[float, float, float, float], v: np.ndarray) -> np.ndarray:
    """Rotate vector v by quaternion q = (w, x, y, z)."""
    w, qx, qy, qz = q
    u = np.array([qx, qy, qz])
    return v + 2.0 * np.cross(u, np.cross(u, v) + w * v)


def projected_area_m2(
    dimensions_m: tuple[float, float, float],
    attitude_quaternion: tuple[float, float, float, float],
    rel_vel_direction: np.ndarray,
) -> float:
    """Compute projected cross-sectional area of a box onto the B-plane.

    Models the satellite as a rectangular cuboid with given dimensions,
    rotated by the attitude quaternion. Projects the 3 face normals
    onto the relative velocity direction and sums visible face areas.

    This replaces the isotropic sphere assumption for collision geometry.

    Args:
        dimensions_m: (length, width, height) in meters.
        attitude_quaternion: (w, x, y, z) unit quaternion.
        rel_vel_direction: (3,) unit vector along relative velocity.

    Returns:
        Projected area in m².
    """
    length, w, h = dimensions_m
    q = attitude_quaternion

    # Body-frame face normals and their areas
    faces = [
        (np.array([1.0, 0.0, 0.0]), w * h),  # +X face
        (np.array([0.0, 1.0, 0.0]), length * h),  # +Y face
        (np.array([0.0, 0.0, 1.0]), length * w),  # +Z face
    ]

    total_area = 0.0
    for normal_body, area in faces:
        # Rotate face normal to ECI frame
        normal_eci = _quat_rotate(q, normal_body)
        # Projected area contribution = |n · v_hat| * face_area
        # Factor of 2: both +/- faces can contribute
        cos_angle = abs(float(np.dot(normal_eci, rel_vel_direction)))
        total_area += cos_angle * area

    return total_area


# ---------------------------------------------------------------------------
# Maneuver Frame Enumeration
# ---------------------------------------------------------------------------


class ManeuverFrame(Enum):
    """Reference frame for specifying maneuver thrust direction.

    **VNB (Velocity-Normal-Binormal)**

    - **V:** Along the instantaneous velocity vector.
    - **N:** Along the orbital angular momentum (R × V).
    - **B:** Completes the right-handed triad (V × N).

    Preferred for orbit-raising and lowering burns: thrust stays aligned with
    velocity even on eccentric orbits.

    **RTN (Radial-Transverse-Normal)** / **RIC (Radial-Intrack-Crosstrack)**

    - **R:** Along the geocentric radial (position) vector.
    - **T:** Perpendicular to R in the orbital plane, roughly along-track.
    - **N:** Along the orbital angular momentum (R × V).

    Preferred for station-keeping and relative navigation.
    """

    VNB = "VNB"
    RTN = "RTN"


# ---------------------------------------------------------------------------
# FiniteBurn (Maneuver Definition)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FiniteBurn:
    """Finite-duration thrust maneuver for the 7-DOF Cowell propagator.

    Models a continuous engine burn with dynamically steered thrust
    (re-computed at every integration sub-step) and Tsiolkovsky-coupled
    mass depletion.

    All epochs are Julian Dates.  Thrust direction is given as a *unit
    vector* in the chosen ``ManeuverFrame``; the engine magnitude is
    set by ``thrust_N``.
    """

    epoch_ignition_jd: float
    duration_s: float
    thrust_N: float
    isp_s: float
    direction: tuple[float, float, float]
    frame: ManeuverFrame

    @property
    def epoch_cutoff_jd(self) -> float:
        """Julian Date of engine cutoff."""
        return self.epoch_ignition_jd + self.duration_s / 86400.0

    @property
    def mass_flow_rate_kg_s(self) -> float:
        """Propellant mass flow rate dm/dt (kg/s, positive value).

        Derived from the Tsiolkovsky relation:
            dm/dt = F / (Isp * g₀)
        """
        # [FM-9 Fix — Finding #12/22] Import from constants instead of local literal.
        return self.thrust_N / (self.isp_s * _G0_STD)

    def __post_init__(self) -> None:
        """Validate burn parameters at construction time.

        Frozen dataclass: do not assign fields here except via ``object.__setattr__``
        if normalization is ever added (initialization runs before immutability locks).

        Raises:
            ValueError: If any parameter is physically invalid.
        """
        import numpy as _np

        if self.duration_s <= 0.0:
            raise ValueError(
                f"FiniteBurn.duration_s must be strictly positive, got {self.duration_s}."
            )
        if self.thrust_N <= 0.0:
            raise ValueError(
                f"FiniteBurn.thrust_N must be strictly positive, got {self.thrust_N} N."
            )
        if self.isp_s <= 0.0:
            raise ValueError(
                f"FiniteBurn.isp_s must be strictly positive, got {self.isp_s} s."
            )
        d_mag = float(_np.linalg.norm(self.direction))
        if abs(d_mag - 1.0) > 1e-6:
            raise ValueError(
                f"FiniteBurn.direction must be a unit vector (|d| = 1.0). "
                f"Got magnitude {d_mag:.8f}. Normalize before constructing FiniteBurn."
            )


# ---------------------------------------------------------------------------
# SatelliteTLE
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SatelliteTLE:
    """Parsed Two-Line Element set for a single orbital object.

    This is the entry point for all computations.  Any satellite, debris,
    or rocket body that can be analysed in ASTRA Core must first exist as
    a ``SatelliteTLE``.
    """

    norad_id: str
    name: str
    line1: str
    line2: str
    epoch_jd: float
    object_type: str
    classification_flag: str = "U"  # 'U'=Unclassified, 'C'=Classified, 'S'=Secret
    bstar: float = 0.0
    # NOTE: This is a security classification, NOT an object type.
    # object_type requires separate SATCAT enrichment.
    rcs_m2: Optional[float] = (
        None  # Radar Cross Section in **m²**. Required for high-fidelity SRP/drag.
    )
    radius_m: Optional[float] = None
    dimensions_m: Optional[tuple[float, float, float]] = None  # (length, width, height)
    attitude_quaternion: Optional[tuple[float, float, float, float]] = (
        None  # (w, x, y, z)
    )
    attitude_mode: str = "TUMBLING"  # Options: "NADIR", "TUMBLING", "INERTIAL"

    @classmethod
    def from_strings(cls, line1: str, line2: str, name: str = "") -> "SatelliteTLE":
        """Create a SatelliteTLE directly from two raw lines, auto-calculating epoch.

        If ``name`` is omitted, a synthetic name ``NORAD-{id}`` is generated from
        the NORAD catalog number embedded in line 1, matching ``load_tle_catalog()``.

        Args:
            line1: Valid TLE line 1.
            line2: Valid TLE line 2.
            name: Optional name for the satellite. If empty, auto-generated.

        Returns:
            A fully initialized SatelliteTLE object.
        """
        if not name:
            norad_id = line1[2:7].strip() if len(line1) >= 7 else ""
            name = f"NORAD-{norad_id}" if norad_id else "Unknown"
        from astra.tle import parse_tle

        return parse_tle(name, line1, line2)


# ---------------------------------------------------------------------------
# SatelliteOMM
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SatelliteOMM:
    """Parsed CCSDS Orbit Mean-Elements Message (OMM) for a single orbital object.

    Represents the modern, high-fidelity successor to the legacy TLE format.
    OMM JSON payloads are published by Space-Track.org and CelesTrak and
    contain additional physical metadata (mass, RCS) that TLEs structurally
    cannot carry.

    All angular fields are stored in **radians** (already converted by the
    OMM parser from the source's degree representation) to allow direct
    injection into ``Satrec.sgp4init()`` without further transformation.
    """

    norad_id: str
    name: str
    epoch_jd: float
    object_type: str
    inclination_rad: float
    raan_rad: float
    argpo_rad: float
    mo_rad: float
    eccentricity: float
    mean_motion_rad_min: float
    bstar: float
    mean_motion_dot: float = 0.0
    mean_motion_ddot: float = 0.0
    rcs_m2: Optional[float] = None  # Radar Cross Section in **m²**.
    mass_kg: Optional[float] = (
        None  # Spacecraft mass in **kg**. Mandatory for powered maneuvers.
    )
    cd_area_over_mass: Optional[float] = None

    @classmethod
    def from_dict(cls, record: dict[str, Any]) -> "SatelliteOMM":
        """Construct a ``SatelliteOMM`` from a raw OMM JSON dictionary.

        Convenience factory — delegates to ``astra.omm.parse_omm_record()``
        so the caller does not need to import the parser module explicitly.

        Args:
            record: A single OMM record dictionary (as from ``json.loads()``).

        Returns:
            Fully populated ``SatelliteOMM`` instance.

        Example::

            import json, astra
            records = json.loads(open("catalog.json").read())
            sats = [astra.SatelliteOMM.from_dict(r) for r in records]
        """
        from astra.omm import parse_omm_record

        return parse_omm_record(record)


# ---------------------------------------------------------------------------
# SatelliteState – polymorphic Union across both data formats
# ---------------------------------------------------------------------------

#: Type alias accepted by all ASTRA-Core physics functions.
#: Any function receiving a ``SatelliteState`` correctly handles
#: both legacy TLE and modern OMM objects without modification.
SatelliteState = Union[SatelliteTLE, SatelliteOMM]

# ---------------------------------------------------------------------------
# OrbitalState
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OrbitalState:
    """Complete kinematic state of a single object at one instant.

    Produced by SGP4 propagation for a single satellite at a single
    time step.
    """

    norad_id: str
    t_jd: float
    position_km: np.ndarray  # shape (3,), dtype float64, TEME, km
    velocity_km_s: np.ndarray  # shape (3,), dtype float64, TEME, km/s
    error_code: int


# ---------------------------------------------------------------------------
# DebrisObject
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DebrisObject:
    """Cataloged orbital object with pre-derived parameters for filtering.

    Contains the authoritative orbital source (either ``SatelliteTLE`` or
    ``SatelliteOMM``) plus derived orbital metrics that enable fast filtering
    **without propagation**.
    """

    source: SatelliteState
    altitude_km: float
    inclination_deg: float
    period_minutes: float
    raan_deg: float
    eccentricity: float
    apogee_km: float
    perigee_km: float
    object_class: str
    rcs_m2: Optional[float] = None
    radius_m: Optional[float] = None

    # ------------------------------------------------------------------
    # Backwards-compatibility shim
    # ------------------------------------------------------------------
    @property
    def tle(self) -> SatelliteTLE:
        """Legacy accessor for TLE source. Raises if source is OMM.

        .. deprecated::
            Use ``.source`` instead, which handles both TLE and OMM.
        """
        if isinstance(self.source, SatelliteTLE):
            return self.source
        raise AttributeError(
            "This DebrisObject was built from an OMM record, not a TLE. "
            "Use `.source` to access the underlying SatelliteOMM."
        )

    def __repr__(self) -> str:
        """Concise representation to avoid terminal hangs on large lists (A-06)."""
        sid = getattr(self.source, "norad_id", "UNK")
        return f"<DebrisObject NORAD={sid} Alt={self.altitude_km:.1f}km>"


# ---------------------------------------------------------------------------
# ConjunctionEvent
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConjunctionEvent:
    """Detected close-approach event between two orbital objects.

    Primary output of conjunction analysis.
    Use ``collision_probability_nan`` for threshold checks without
    ``TypeError`` when Pc is unknown (``None`` becomes ``float('nan')``).
    """

    object_a_id: str
    object_b_id: str
    tca_jd: float
    miss_distance_km: float
    relative_velocity_km_s: float
    collision_probability: Optional[float]  # None when no covariance available
    risk_level: str
    position_a_km: np.ndarray  # shape (3,), TEME, km
    position_b_km: np.ndarray  # shape (3,), TEME, km
    covariance_source: str = "SYNTHETIC"  # "CDM", "STM", "SYNTHETIC", or "UNAVAILABLE"

    @property
    def collision_probability_nan(self) -> float:
        """Collision probability as a ``float``, with ``None`` mapped to ``float('nan')``.

        Allows safe threshold comparisons without ``TypeError``::

            pc = event.collision_probability_nan
            if not math.isnan(pc) and pc > 1e-4:
                alert(event)

        Returns:
            Pc in [0.0, 1.0] when covariance data was available, or
            ``float('nan')`` when ``collision_probability`` is ``None``.
        """
        return (
            float("nan")
            if self.collision_probability is None
            else self.collision_probability
        )


# ---------------------------------------------------------------------------
# Observer
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Observer:
    """Ground-based observation station.

    Required by all visibility functions.
    """

    name: str
    latitude_deg: float
    longitude_deg: float
    elevation_m: float
    min_elevation_deg: float = 10.0

    def __repr__(self) -> str:
        return f"<Observer '{self.name}' Lat={self.latitude_deg:.2f} Lon={self.longitude_deg:.2f}>"


# ---------------------------------------------------------------------------
# PassEvent
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PassEvent:
    """Satellite pass over a ground observer.

    A time interval during which the satellite's elevation angle exceeds
    the observer's minimum elevation threshold.
    """

    norad_id: str
    observer_name: str
    aos_jd: float
    tca_jd: float
    los_jd: float
    max_elevation_deg: float
    azimuth_at_aos_deg: float
    azimuth_at_los_deg: float
    duration_seconds: float


# ---------------------------------------------------------------------------
# FilterConfig
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FilterConfig:
    """Filter parameters for the multi-stage debris filtering pipeline.

    Passed as a single configuration object to ``apply_filters()``.
    ``None`` fields are treated as "no constraint" for that dimension.
    """

    min_altitude_km: Optional[float] = None
    max_altitude_km: Optional[float] = None
    lat_min_deg: Optional[float] = None
    lat_max_deg: Optional[float] = None
    lon_min_deg: Optional[float] = None
    lon_max_deg: Optional[float] = None
    t_start_jd: Optional[float] = None
    t_end_jd: Optional[float] = None
    object_types: Optional[tuple[str, ...]] = None
    max_objects: Optional[int] = (
        None  # Applied after other filters; keep first N survivors
    )
