"""
ASTRA Coordinate Conversions
Handles TEME → ECEF and TEME → Geodetic coordinate transformations.
"""

import math
import numpy as np
from datetime import datetime, timezone

from app.core.config import settings

# Earth parameters
RE = settings.EARTH_RADIUS_KM
EARTH_ROTATION_RATE = 7.2921158553e-5  # rad/s (WGS84)


def gmst_from_datetime(dt: datetime) -> float:
    """
    Compute Greenwich Mean Sidereal Time (GMST) in radians from a UTC datetime.
    Uses simplified IAU formula sufficient for TEME→ECEF rotation.
    """
    # Julian date
    a = (14 - dt.month) // 12
    y = dt.year + 4800 - a
    m = dt.month + 12 * a - 3

    jdn = dt.day + (153 * m + 2) // 5 + 365 * y + y // 4 - y // 100 + y // 400 - 32045
    jd = jdn + (dt.hour - 12) / 24.0 + dt.minute / 1440.0 + dt.second / 86400.0

    # Julian centuries from J2000.0
    t_ut1 = (jd - 2451545.0) / 36525.0

    # GMST in seconds, then convert to radians
    gmst_sec = (
        67310.54841
        + (876600.0 * 3600.0 + 8640184.812866) * t_ut1
        + 0.093104 * t_ut1**2
        - 6.2e-6 * t_ut1**3
    )
    gmst_rad = math.fmod(gmst_sec * (2.0 * math.pi / 86400.0), 2.0 * math.pi)
    if gmst_rad < 0:
        gmst_rad += 2.0 * math.pi

    return gmst_rad


def teme_to_ecef(position_teme: np.ndarray, dt: datetime) -> np.ndarray:
    """
    Convert a single position vector from TEME to ECEF frame.

    Args:
        position_teme: (3,) array [x, y, z] in km (TEME)
        dt: UTC datetime for the rotation angle

    Returns:
        (3,) array [x, y, z] in km (ECEF)
    """
    gmst = gmst_from_datetime(dt)
    cos_g = math.cos(gmst)
    sin_g = math.sin(gmst)

    x_teme, y_teme, z_teme = position_teme
    x_ecef = cos_g * x_teme + sin_g * y_teme
    y_ecef = -sin_g * x_teme + cos_g * y_teme
    z_ecef = z_teme

    return np.array([x_ecef, y_ecef, z_ecef])


def teme_to_ecef_batch(positions_teme: np.ndarray, times: list[datetime]) -> np.ndarray:
    """
    Convert an array of position vectors from TEME to ECEF.

    Args:
        positions_teme: (N, 3) array in km (TEME)
        times: list of N datetime objects

    Returns:
        (N, 3) array in km (ECEF)
    """
    n = len(times)
    positions_ecef = np.empty((n, 3), dtype=np.float64)

    for i in range(n):
        if np.any(np.isnan(positions_teme[i])):
            positions_ecef[i] = [np.nan, np.nan, np.nan]
        else:
            positions_ecef[i] = teme_to_ecef(positions_teme[i], times[i])

    return positions_ecef


def ecef_to_geodetic(position_ecef: np.ndarray) -> tuple[float, float, float]:
    """
    Convert ECEF (x, y, z) to geodetic (latitude, longitude, altitude).

    Returns:
        (latitude_deg, longitude_deg, altitude_km)
    """
    x, y, z = position_ecef
    lon = math.degrees(math.atan2(y, x))
    r_xy = math.sqrt(x**2 + y**2)
    lat = math.degrees(math.atan2(z, r_xy))
    alt = math.sqrt(x**2 + y**2 + z**2) - RE
    return (lat, lon, alt)


def positions_to_geodetic(positions_ecef: np.ndarray) -> np.ndarray:
    """
    Convert ECEF positions to geodetic coordinates.

    Args:
        positions_ecef: (N, 3) array in km

    Returns:
        (N, 3) array of [latitude_deg, longitude_deg, altitude_km]
    """
    n = len(positions_ecef)
    geodetic = np.empty((n, 3), dtype=np.float64)

    for i in range(n):
        if np.any(np.isnan(positions_ecef[i])):
            geodetic[i] = [np.nan, np.nan, np.nan]
        else:
            geodetic[i] = ecef_to_geodetic(positions_ecef[i])

    return geodetic


def teme_to_visualization(position_teme: np.ndarray, scale: float = 1.0) -> np.ndarray:
    """
    Convert TEME coordinates directly to Three.js visualization coordinates.
    Three.js uses a right-handed Y-up system, so we map:
      x_viz = x_teme * scale
      y_viz = z_teme * scale  (Z-up → Y-up)
      z_viz = -y_teme * scale

    Args:
        position_teme: (3,) or (N, 3) array in km
        scale: scale factor (e.g., 1/6371 to normalize to Earth radii)

    Returns:
        Array of same shape with visualization coordinates.
    """
    if position_teme.ndim == 1:
        return np.array([
            position_teme[0] * scale,
            position_teme[2] * scale,
            -position_teme[1] * scale,
        ])
    else:
        result = np.empty_like(position_teme)
        result[:, 0] = position_teme[:, 0] * scale
        result[:, 1] = position_teme[:, 2] * scale
        result[:, 2] = -position_teme[:, 1] * scale
        return result
