# astra/models.py
"""ASTRA Core data models.

All persistent data structures used across the library are defined here as
frozen dataclasses.  This is the single source of truth for all inter-module
data types — no other module defines dataclasses.

Every dataclass uses ``frozen=True`` to enforce immutability.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Type Aliases
# ---------------------------------------------------------------------------

# A map from NORAD ID to trajectory array.
# Shape of each array: (T, 3) where T = number of timesteps.
# Units: km, frame: TEME.
TrajectoryMap = dict[str, np.ndarray]

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
    l, w, h = dimensions_m
    q = attitude_quaternion

    # Body-frame face normals and their areas
    faces = [
        (np.array([1.0, 0.0, 0.0]), w * h),  # +X face
        (np.array([0.0, 1.0, 0.0]), l * h),  # +Y face
        (np.array([0.0, 0.0, 1.0]), l * w),  # +Z face
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
# SatelliteTLE
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SatelliteTLE:
    """Parsed Two-Line Element set for a single orbital object.

    This is the entry point for all computations.  Any satellite, debris,
    or rocket body that can be analysed in ASTRA Core must first exist as
    a ``SatelliteTLE``.

    Attributes:
        norad_id: NORAD Catalog Number (1–9 digits, numeric-only string).
        name: Object name (up to 24 characters, may contain spaces).
        line1: Raw TLE Line 1 (exactly 69 characters, no newline).
        line2: Raw TLE Line 2 (exactly 69 characters, no newline).
        epoch_jd: TLE epoch as Julian Date.
        object_type: One of ``"PAYLOAD"``, ``"ROCKET_BODY"``, ``"DEBRIS"``,
            ``"UNKNOWN"``.
    """

    norad_id: str
    name: str
    line1: str
    line2: str
    epoch_jd: float
    object_type: str
    rcs_m2: Optional[float] = None
    radius_m: Optional[float] = None
    dimensions_m: Optional[tuple[float, float, float]] = None  # (length, width, height)
    attitude_quaternion: Optional[tuple[float, float, float, float]] = None  # (w, x, y, z)
    attitude_mode: str = "TUMBLING"  # Options: "NADIR", "TUMBLING", "INERTIAL"


# ---------------------------------------------------------------------------
# OrbitalState
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class OrbitalState:
    """Complete kinematic state of a single object at one instant.

    Produced by SGP4 propagation for a single satellite at a single
    time step.

    Attributes:
        norad_id: Identifier linking back to the source ``SatelliteTLE``.
        t_jd: Julian Date of this state.
        position_km: Shape ``(3,)`` array ``[x, y, z]`` in TEME, km.
        velocity_km_s: Shape ``(3,)`` array ``[vx, vy, vz]`` in TEME, km/s.
        error_code: SGP4 error code (0 = success, 1–6 = failure mode).
    """

    norad_id: str
    t_jd: float
    position_km: np.ndarray   # shape (3,), dtype float64, TEME, km
    velocity_km_s: np.ndarray  # shape (3,), dtype float64, TEME, km/s
    error_code: int


# ---------------------------------------------------------------------------
# DebrisObject
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DebrisObject:
    """Cataloged orbital object with pre-derived parameters for filtering.

    Contains the authoritative ``SatelliteTLE`` plus derived orbital metrics
    that enable fast filtering **without propagation**.

    Attributes:
        tle: Authoritative TLE source.
        altitude_km: Mean orbital altitude above Earth's surface (km).
        inclination_deg: Orbital inclination in degrees (0–180).
        period_minutes: Orbital period in minutes.
        raan_deg: Right Ascension of the Ascending Node (0–360 degrees).
        eccentricity: Orbital eccentricity (0.0 ≤ e < 1.0).
        apogee_km: Apogee altitude (km above surface).
        perigee_km: Perigee altitude (km above surface).
        object_class: Same as ``tle.object_type``.
    """

    tle: SatelliteTLE
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


# ---------------------------------------------------------------------------
# ConjunctionEvent
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ConjunctionEvent:
    """Detected close-approach event between two orbital objects.

    Primary output of conjunction analysis.

    Attributes:
        object_a_id: NORAD ID of first object (lexicographically smaller).
        object_b_id: NORAD ID of second object (lexicographically larger).
        tca_jd: Time of Closest Approach as Julian Date.
        miss_distance_km: Minimum separation distance at TCA (km).
        relative_velocity_km_s: Relative speed at TCA (km/s).
        collision_probability: Chan-method probability (0.0–1.0).
        risk_level: ``"LOW"``, ``"MEDIUM"``, ``"HIGH"``, or ``"CRITICAL"``.
        position_a_km: Shape ``(3,)`` TEME position of object A at TCA (km).
        position_b_km: Shape ``(3,)`` TEME position of object B at TCA (km).
    """

    object_a_id: str
    object_b_id: str
    tca_jd: float
    miss_distance_km: float
    relative_velocity_km_s: float
    collision_probability: float
    risk_level: str
    position_a_km: np.ndarray   # shape (3,), TEME, km
    position_b_km: np.ndarray   # shape (3,), TEME, km


# ---------------------------------------------------------------------------
# Observer
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Observer:
    """Ground-based observation station.

    Required by all visibility functions.

    Attributes:
        name: Human-readable station name.
        latitude_deg: WGS84 geodetic latitude (-90 to +90).
        longitude_deg: WGS84 longitude (-180 to +180).
        elevation_m: Station elevation above MSL in metres.
        min_elevation_deg: Minimum elevation angle to consider a pass
            visible (default 10°).
    """

    name: str
    latitude_deg: float
    longitude_deg: float
    elevation_m: float
    min_elevation_deg: float = 10.0


# ---------------------------------------------------------------------------
# PassEvent
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PassEvent:
    """Satellite pass over a ground observer.

    A time interval during which the satellite's elevation angle exceeds
    the observer's minimum elevation threshold.

    Attributes:
        norad_id: NORAD ID of the satellite.
        observer_name: Name of the ground observer.
        aos_jd: Acquisition of Signal — rise above min elevation (JD).
        tca_jd: Time of max elevation (JD).
        los_jd: Loss of Signal — drops below min elevation (JD).
        max_elevation_deg: Maximum elevation reached during pass.
        azimuth_at_aos_deg: Azimuth angle at AOS.
        azimuth_at_los_deg: Azimuth angle at LOS.
        duration_seconds: Total pass duration ``(los_jd - aos_jd) * 86400``.
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

    Attributes:
        min_altitude_km: Lower altitude bound (km), inclusive.
        max_altitude_km: Upper altitude bound (km), inclusive.
        lat_min_deg: Minimum latitude of geographic region (deg).
        lat_max_deg: Maximum latitude of geographic region (deg).
        lon_min_deg: Minimum longitude of geographic region (deg).
        lon_max_deg: Maximum longitude of geographic region (deg).
        t_start_jd: Analysis window start as Julian Date.
        t_end_jd: Analysis window end as Julian Date.
        object_types: Tuple of allowed object types, e.g.
            ``("DEBRIS", "ROCKET_BODY")``.
        max_objects: Hard cap on the number of survivors.
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
    max_objects: Optional[int] = None
