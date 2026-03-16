"""
ASTRA Orbital Propagation Module
SGP4 propagation using the sgp4 library — NEVER manual orbital mechanics.
Implements trajectory precomputation (propagate once per object, not per pair).
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np
from sgp4.api import Satrec, WGS72
from sgp4.api import jday

from app.core.config import settings

logger = logging.getLogger(__name__)


def create_satellite(tle_line1: str, tle_line2: str) -> Optional[Satrec]:
    """Create an SGP4 satellite object from TLE lines."""
    try:
        satellite = Satrec.twoline2rv(tle_line1, tle_line2, WGS72)
        return satellite
    except Exception as e:
        logger.warning(f"Failed to create satellite from TLE: {e}")
        return None


def generate_simulation_times(
    start_time: Optional[datetime] = None,
    window_hours: int = None,
    resolution_minutes: int = None,
) -> tuple[list[datetime], np.ndarray, np.ndarray]:
    """
    Generate simulation time steps.

    Returns:
        - list of datetime objects
        - numpy array of Julian date integers
        - numpy array of Julian date fractions
    """
    if window_hours is None:
        window_hours = settings.PREDICTION_WINDOW_HOURS
    if resolution_minutes is None:
        resolution_minutes = settings.TIME_RESOLUTION_MINUTES

    if start_time is None:
        start_time = datetime.now(timezone.utc)

    total_steps = int((window_hours * 60) / resolution_minutes)

    times = []
    jd_array = np.empty(total_steps, dtype=np.float64)
    fr_array = np.empty(total_steps, dtype=np.float64)

    for i in range(total_steps):
        t = start_time + timedelta(minutes=i * resolution_minutes)
        times.append(t)
        jd, fr = jday(t.year, t.month, t.day, t.hour, t.minute, t.second + t.microsecond / 1e6)
        jd_array[i] = jd
        fr_array[i] = fr

    return times, jd_array, fr_array


def propagate_object(
    tle_line1: str,
    tle_line2: str,
    jd_array: np.ndarray,
    fr_array: np.ndarray,
) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """
    Propagate a single object across all simulation time steps using SGP4.

    Returns:
        Tuple of (positions, velocities) as NumPy arrays of shape (N, 3)
        in TEME reference frame (km and km/s).
        Returns None if propagation fails.
    """
    satellite = create_satellite(tle_line1, tle_line2)
    if satellite is None:
        return None

    n = len(jd_array)
    positions = np.empty((n, 3), dtype=np.float64)
    velocities = np.empty((n, 3), dtype=np.float64)

    for i in range(n):
        error_code, r, v = satellite.sgp4(jd_array[i], fr_array[i])
        if error_code != 0:
            # Mark failed propagation with NaN
            positions[i] = [np.nan, np.nan, np.nan]
            velocities[i] = [np.nan, np.nan, np.nan]
        else:
            positions[i] = r  # km in TEME
            velocities[i] = v  # km/s in TEME

    return positions, velocities


def propagate_batch(
    objects: list[dict],
    jd_array: np.ndarray,
    fr_array: np.ndarray,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """
    Precompute trajectories for ALL objects across ALL simulation time steps.

    This is the REQUIRED architecture per doc 08:
    - Propagate each object ONCE across all time steps.
    - Store trajectory arrays in memory.
    - NEVER propagate inside pairwise comparison loops.

    Args:
        objects: list of dicts with keys 'norad_id', 'tle_line1', 'tle_line2'
        jd_array: Julian date integer parts
        fr_array: Julian date fraction parts

    Returns:
        Dict mapping norad_id -> (positions_array, velocities_array)
        Each array has shape (288, 3) in TEME frame.
    """
    trajectories = {}
    success_count = 0
    fail_count = 0

    for obj in objects:
        norad_id = obj["norad_id"]
        result = propagate_object(obj["tle_line1"], obj["tle_line2"], jd_array, fr_array)

        if result is not None:
            trajectories[norad_id] = result
            success_count += 1
        else:
            fail_count += 1

    logger.info(
        f"Batch propagation complete: {success_count} succeeded, {fail_count} failed "
        f"out of {len(objects)} objects"
    )
    return trajectories


def propagate_to_time(
    objects: list[dict],
    time_step: int,
    start_time: Optional[datetime] = None,
    resolution_minutes: int = None,
) -> tuple[np.ndarray, list[int]]:
    """
    Propagate all objects to a SPECIFIC time step and prepare coordinates for visualization.
    This fulfills the requirement that rendering positions must come from SGP4.

    Args:
        objects: list of dicts with 'norad_id', 'tle_line1', 'tle_line2'
        time_step: integer step (0 to max_steps)
        start_time: base time for t=0 (defaults to now)
        resolution_minutes: minutes per step (default from settings)

    Returns:
        (positions_array, ids_list) where:
        - positions_array is (N, 3) flattened for Three.js: [x1, y1, z1, x2, y2, z2, ...]
          in appropriate visualization scale.
        - ids_list is the corresponding norad_ids to map them in the frontend.
    """
    from app.orbit.coordinates import teme_to_visualization

    if resolution_minutes is None:
        resolution_minutes = settings.TIME_RESOLUTION_MINUTES
    if start_time is None:
        start_time = datetime.now(timezone.utc)

    # Compute exact datetime and Julian date for this specific step
    t = start_time + timedelta(minutes=time_step * resolution_minutes)
    jd, fr = jday(t.year, t.month, t.day, t.hour, t.minute, t.second + t.microsecond / 1e6)

    # We need to return flat Float32Array-compatible data for Three.js InstancedMesh
    n = len(objects)
    positions_teme = np.empty((n, 3), dtype=np.float64)
    ids_list = []

    for i, obj in enumerate(objects):
        satellite = create_satellite(obj["tle_line1"], obj["tle_line2"])
        if satellite is None:
            positions_teme[i] = [np.nan, np.nan, np.nan]
            ids_list.append(obj["norad_id"])
            continue

        error_code, r, v = satellite.sgp4(jd, fr)
        if error_code != 0:
             positions_teme[i] = [np.nan, np.nan, np.nan]
        else:
             positions_teme[i] = r
        
        ids_list.append(obj["norad_id"])

    # Scale factor: Three.js Earth has radius 1. So 1 unit = EARTH_RADIUS_KM.
    # Therefore, we scale by 1 / EARTH_RADIUS_KM
    scale = 1.0 / settings.EARTH_RADIUS_KM
    
    # Convert to Three.js coordinate system (Y-up)
    viz_positions = teme_to_visualization(positions_teme, scale=scale)
    
    # Flatten the array for easy transport and consumption by InstancedBufferAttribute
    flat_positions = viz_positions.astype(np.float32).flatten()
    
    # Replace NaNs with 0 (or a far away location) so it doesn't break WebGL
    # Put them inside Earth so they are hidden.
    flat_positions = np.nan_to_num(flat_positions, nan=0.0)

    return flat_positions, ids_list
