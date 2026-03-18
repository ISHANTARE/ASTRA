# astra/orbit.py
"""ASTRA Core orbit propagation module.

Responsible for SGP4-based orbit propagation. Generates position and velocity
vectors at specified times. Uses `sgp4` for actual propagation and `skyfield`
for coordinate frame conversions. No manual orbital equations allowed.
"""
from __future__ import annotations

import numpy as np
from sgp4.api import Satrec, WGS84
from skyfield.api import load, wgs84
from skyfield.constants import AU_KM
from skyfield.positionlib import Geocentric
from skyfield.sgp4lib import TEME

from astra.errors import AstraError
from astra.models import OrbitalState, SatelliteTLE, TrajectoryMap


def propagate_orbit(
    satellite: SatelliteTLE, epoch_jd: float, t_since_minutes: float
) -> OrbitalState:
    """Propagate a single satellite to a single point in time using SGP4.

    Args:
        satellite: Parsed SatelliteTLE object to propagate.
        epoch_jd: Reference epoch as Julian Date (typically satellite.epoch_jd).
        t_since_minutes: Minutes elapsed since epoch.

    Returns:
        OrbitalState containing position (km) and velocity (km/s) in TEME frame.
    """
    satrec = Satrec.twoline2rv(satellite.line1, satellite.line2)
    
    t_jd = epoch_jd + (t_since_minutes / 1440.0)
    fraction = 0.0  # SGP4 handles jd without needing explicit fraction division
    
    error_code, position, velocity = satrec.sgp4(t_jd, fraction)
    
    return OrbitalState(
        norad_id=satellite.norad_id,
        t_jd=t_jd,
        position_km=np.array(position, dtype=np.float64),
        velocity_km_s=np.array(velocity, dtype=np.float64),
        error_code=error_code,
    )


def propagate_many(
    satellites: list[SatelliteTLE], time_steps: np.ndarray
) -> TrajectoryMap:
    """Vectorized batch propagation of multiple satellites across a time array.

    Args:
        satellites: List of SatelliteTLE objects to propagate.
        time_steps: 1D NumPy array of T time offsets in minutes since each
            satellite's own epoch. (Shape: (T,), dtype: float64)

    Returns:
        TrajectoryMap: Dictionary mapping norad_id to np.ndarray shape (T, 3) 
            in TEME frame, km. Satellites with propagation errors at any timestep 
            store np.nan at that timestep row.
    """
    results: TrajectoryMap = {}

    for sat in satellites:
        satrec = Satrec.twoline2rv(sat.line1, sat.line2)
        
        jd_array = sat.epoch_jd + (time_steps / 1440.0)
        jd_fraction_array = np.zeros_like(jd_array)
        
        e, r, v = satrec.sgp4_array(jd_array, jd_fraction_array)
        
        # r is of shape (T, 3). e is of shape (T,) error codes.
        # Set rows with error_codes > 0 to np.nan
        r[e > 0] = np.nan
        
        results[sat.norad_id] = r

    return results


def propagate_trajectory(
    satellite: SatelliteTLE, t_start_jd: float, t_end_jd: float, step_minutes: float = 5.0
) -> tuple[np.ndarray, np.ndarray]:
    """Propagate a single satellite over a defined time window at a fixed step.

    Convenience function for analysis over a time range.

    Args:
        satellite: Source SatelliteTLE to propagate.
        t_start_jd: Window start as Julian Date.
        t_end_jd: Window end as Julian Date.
        step_minutes: Step size in minutes (default 5.0).

    Returns:
        Tuple of (time_array, position_array):
            time_array: shape (T,), Julian Dates for each step.
            position_array: shape (T, 3), TEME positions in km.
    """
    start_offset_mins = (t_start_jd - satellite.epoch_jd) * 1440.0
    end_offset_mins = (t_end_jd - satellite.epoch_jd) * 1440.0
    
    # Adding a tiny epsilon to end_offset_mins to ensure inclusive bound if it aligns exactly
    time_steps = np.arange(start_offset_mins, end_offset_mins + 1e-9, step_minutes)
    
    trajectory_map = propagate_many([satellite], time_steps)
    
    positions = trajectory_map[satellite.norad_id]
    times_jd = satellite.epoch_jd + (time_steps / 1440.0)
    
    return times_jd, positions


def ground_track(
    positions_teme: np.ndarray, times_jd: np.ndarray
) -> list[tuple[float, float]]:
    """Convert TEME Cartesian positions into geodetic coordinates for ground track.

    Args:
        positions_teme: TEME position array from propagation. Shape: (T, 3)
        times_jd: Corresponding Julian Date array. Shape: (T,)

    Returns:
        List of (latitude_deg, longitude_deg) tuples, length T.
    """
    if len(times_jd) == 0:
        return []

    ts = load.timescale()
    t = ts.utc(jd=times_jd)
    
    # 1. Convert positions to AU for skyfield operations. shape (3, T)
    r_teme_au = positions_teme.T / AU_KM
    
    # R computes TEME -> GCRS (which is standard Geocentric frame)
    R_teme_to_gcrs = np.transpose(TEME.rotation_at(t), axes=(1, 0, 2)) if hasattr(t, 'shape') else np.transpose(TEME.rotation_at(t))
    
    # 2. Convert TEME to GCRS 
    try:
        if hasattr(t, 'shape'):
            r_gcrs_au = np.einsum('ij...,j...->i...', R_teme_to_gcrs, r_teme_au)
        else:
            r_gcrs_au = R_teme_to_gcrs.dot(r_teme_au)
    except Exception as e:
        raise AstraError(f"Coordinate error in ground_track: {e}") from e
    
    # 3. Use skyfield API (Geocentric + wgs84 subpoint)
    pos = Geocentric(r_gcrs_au, t=t)
    sub = wgs84.subpoint(pos)
    
    lat_deg = sub.latitude.degrees
    lon_deg = sub.longitude.degrees
    
    if np.isscalar(lat_deg):
        return [(float(lat_deg), float(lon_deg))]
    
    return list(zip(lat_deg, lon_deg))
