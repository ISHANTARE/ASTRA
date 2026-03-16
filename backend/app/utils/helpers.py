"""
ASTRA Utility Helpers
General helper functions used across modules.
"""

import math


def km_to_earth_radii(km: float) -> float:
    """Convert kilometers to Earth radii."""
    return km / 6371.0


def earth_radii_to_km(er: float) -> float:
    """Convert Earth radii to kilometers."""
    return er * 6371.0


def deg_to_rad(degrees: float) -> float:
    """Convert degrees to radians."""
    return math.radians(degrees)


def rad_to_deg(radians: float) -> float:
    """Convert radians to degrees."""
    return math.degrees(radians)


def format_distance(km: float) -> str:
    """Format distance for display."""
    if km < 1.0:
        return f"{km * 1000:.1f} m"
    elif km < 100.0:
        return f"{km:.2f} km"
    else:
        return f"{km:.0f} km"


def format_velocity(km_s: float) -> str:
    """Format velocity for display."""
    return f"{km_s:.2f} km/s"
