# astra/utils.py
"""ASTRA Core utility functions.

Pure mathematical and geometric utilities. This module contains no domain
concepts or dependencies on other ASTRA modules.
"""
import math


def haversine_distance(
    lat1_deg: float, lon1_deg: float, lat2_deg: float, lon2_deg: float
) -> float:
    """Compute the great-circle distance between two points on the Earth's surface.

    Using the standard Haversine formula with a mean Earth radius of 6371.0 km.

    Args:
        lat1_deg: Latitude of the first point in degrees.
        lon1_deg: Longitude of the first point in degrees.
        lat2_deg: Latitude of the second point in degrees.
        lon2_deg: Longitude of the second point in degrees.

    Returns:
        The great-circle distance in kilometers.
    """
    re_km = 6371.0

    lat1_rad = math.radians(lat1_deg)
    lon1_rad = math.radians(lon1_deg)
    lat2_rad = math.radians(lat2_deg)
    lon2_rad = math.radians(lon2_deg)

    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    a = (math.sin(dlat / 2.0) ** 2) + math.cos(lat1_rad) * math.cos(
        lat2_rad
    ) * (math.sin(dlon / 2.0) ** 2)
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))

    return re_km * c


def orbital_elements(line2: str) -> dict[str, float]:
    """Extract Keplerian orbital elements from a TLE line 2 string.

    Args:
        line2: Raw TLE Line 2 (must be at least 63 characters).

    Returns:
        Dictionary mapping element names to their floating-point values.
    """
    return {
        "inclination_deg": float(line2[8:16].strip()),
        "raan_deg": float(line2[17:25].strip()),
        # Eccentricity has an assumed leading decimal point in the TLE
        "eccentricity": float("0." + line2[26:33].strip()),
        "arg_perigee_deg": float(line2[34:42].strip()),
        "mean_anomaly_deg": float(line2[43:51].strip()),
        "mean_motion_rev_per_day": float(line2[52:63].strip()),
    }


def orbit_period(mean_motion_rev_per_day: float) -> float:
    """Compute the orbital period from mean motion.

    Args:
        mean_motion_rev_per_day: Mean motion in revolutions per day.

    Returns:
        Orbital period in minutes.
    """
    if mean_motion_rev_per_day <= 0:
        return float("inf")
    return (24.0 * 60.0) / mean_motion_rev_per_day
