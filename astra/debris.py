# astra/debris.py
"""ASTRA Core debris catalog filtering and statistics.

Pre-propagation filtering of debris catalogs. ALL filtering in this module
operates on DebrisObject parameters (derived TLE fields, NOT propagated
positions). Filtering is O(N) and involves no SGP4 calls.
"""
from __future__ import annotations

import math
from typing import Any

from astra.models import DebrisObject, FilterConfig, SatelliteTLE
from astra.utils import orbit_period, orbital_elements

_MU_EARTH = 398600.4418


def make_debris_object(tle: SatelliteTLE) -> DebrisObject:
    """Build a DebrisObject from a SatelliteTLE with derived elements."""
    elements = orbital_elements(tle.line2)
    period_min = orbit_period(elements["mean_motion_rev_per_day"])

    # Compute semi-major axis using Kepler's Third Law
    if period_min <= 0:
        n_rad_s = float('inf')
        a = 0.0
    else:
        n_rad_s = (2.0 * math.pi) / (period_min * 60.0)
        a = (_MU_EARTH / (n_rad_s**2)) ** (1.0 / 3.0)

    e = elements["eccentricity"]
    perigee_km = a * (1.0 - e) - 6371.0
    apogee_km = a * (1.0 + e) - 6371.0
    altitude_km = a - 6371.0

    return DebrisObject(
        tle=tle,
        altitude_km=altitude_km,
        inclination_deg=elements["inclination_deg"],
        period_minutes=period_min,
        raan_deg=elements["raan_deg"],
        eccentricity=e,
        apogee_km=apogee_km,
        perigee_km=perigee_km,
        object_class=tle.object_type,
    )


def filter_altitude(
    objects: list[DebrisObject], min_km: float, max_km: float
) -> list[DebrisObject]:
    """Retain only objects whose mean orbital altitude is within bounds.

    Also filters objects whose perigee is pathologically low 
    (perigee < min_km * 0.9) to discard quickly-decaying objects.

    Args:
        objects: List of DebrisObjects.
        min_km: Lower altitude bound (inclusive).
        max_km: Upper altitude bound (inclusive).

    Returns:
        Filtered list of DebrisObjects.
    """
    results = []
    for obj in objects:
        if min_km <= obj.altitude_km <= max_km:
            if obj.perigee_km >= (min_km * 0.9):
                results.append(obj)
    return results


def filter_region(
    objects: list[DebrisObject],
    lat_min_deg: float,
    lat_max_deg: float,
    lon_min_deg: float,
    lon_max_deg: float,
) -> list[DebrisObject]:
    """Retain objects whose ground track could pass through a bounding box.

    Approximation based on orbital inclination. It is an over-inclusive filter.

    Args:
        objects: List of DebrisObjects.
        lat_min_deg: Minimum latitude bound.
        lat_max_deg: Maximum latitude bound.
        lon_min_deg: Minimum longitude bound (currently unused, kept for API).
        lon_max_deg: Maximum longitude bound (currently unused, kept for API).

    Returns:
        Filtered list of DebrisObjects.
    """
    results = []
    for obj in objects:
        # For inclination > 90 (retrograde), the maximum latitude reached
        # is 180 - inclination.
        inc = obj.inclination_deg
        max_lat_reached = inc if inc <= 90.0 else 180.0 - inc

        # Object latitude bounds during its orbit: [-max_lat_reached, +max_lat_reached]
        # Bounding box latitude: [lat_min_deg, lat_max_deg]
        
        # Check if the two intervals overlap
        if lat_min_deg <= max_lat_reached and lat_max_deg >= -max_lat_reached:
            # Overlap exists. Longitude is skipped since the Earth rotates underneath,
            # meaning all longitudes are eventually covered in the operational band.
            results.append(obj)
    return results


def filter_time_window(
    objects: list[DebrisObject], t_start_jd: float, t_end_jd: float
) -> list[DebrisObject]:
    """Eliminate objects whose TLE epoch is too stale for predictions.

    Args:
        objects: List of DebrisObjects.
        t_start_jd: Window start as Julian Date.
        t_end_jd: Window end as Julian Date.

    Returns:
        Filtered list of DebrisObjects.
    """
    results = []
    for obj in objects:
        age_days = t_start_jd - obj.tle.epoch_jd
        
        # Stale thresholds
        # Stricter threshold for LEO due to higher atmospheric drag
        is_stale = False
        if obj.altitude_km < 2000:
            if age_days > 7.0:
                is_stale = True
        else:
            if age_days > 14.0:
                is_stale = True
                
        if not is_stale:
            results.append(obj)
            
    return results


def catalog_statistics(objects: list[DebrisObject]) -> dict[str, Any]:
    """Compute summary statistics across a debris catalog.

    Args:
        objects: List of DebrisObjects.

    Returns:
        Dictionary of computed statistics.
    """
    total_count = len(objects)

    by_type = {"PAYLOAD": 0, "ROCKET_BODY": 0, "DEBRIS": 0, "UNKNOWN": 0}
    by_regime = {"LEO": 0, "MEO": 0, "GEO": 0, "HEO": 0}
    
    inclination_dist = {"equatorial": 0, "inclined": 0, "polar": 0, "retrograde": 0}

    altitudes = []

    for obj in objects:
        # By type
        t = obj.object_class if obj.object_class in by_type else "UNKNOWN"
        by_type[t] += 1
        
        # By regime (Basic logic)
        alt = obj.altitude_km
        altitudes.append(alt)
        
        if obj.eccentricity > 0.25:
            # Highly Elliptical
            by_regime["HEO"] += 1
        elif alt < 2000:
            by_regime["LEO"] += 1
        elif 35000 <= alt <= 36000:
            # Approx GEO ring
            by_regime["GEO"] += 1
        else:
            # Middle Earth Orbit
            by_regime["MEO"] += 1

        # Inclination distribution
        inc = obj.inclination_deg
        if inc < 10.0:
            inclination_dist["equatorial"] += 1
        elif inc < 80.0:
            inclination_dist["inclined"] += 1
        elif inc <= 90.0:
            inclination_dist["polar"] += 1
        else:
            inclination_dist["retrograde"] += 1

    if not altitudes:
        return {
            "total_count": 0,
            "by_type": by_type,
            "by_regime": by_regime,
            "altitude_mean_km": 0.0,
            "altitude_std_km": 0.0,
            "altitude_min_km": 0.0,
            "altitude_max_km": 0.0,
            "inclination_distribution": inclination_dist,
        }

    alt_mean = sum(altitudes) / total_count
    # Std dev
    variance = sum((a - alt_mean) ** 2 for a in altitudes) / total_count
    alt_std = math.sqrt(variance)

    return {
        "total_count": total_count,
        "by_type": by_type,
        "by_regime": by_regime,
        "altitude_mean_km": alt_mean,
        "altitude_std_km": alt_std,
        "altitude_min_km": min(altitudes),
        "altitude_max_km": max(altitudes),
        "inclination_distribution": inclination_dist,
    }


def apply_filters(
    catalog: list[DebrisObject], config: FilterConfig
) -> list[DebrisObject]:
    """Execute the REQUIRED PIPELINE FUNCTION for filtering.

    Args:
        catalog: List of DebrisObjects to filter.
        config: FilterConfig data class outlining constraints.

    Returns:
        Filtered list of DebrisObjects.
    """
    filtered = catalog

    # 1. Apply altitude filter
    if config.min_altitude_km is not None and config.max_altitude_km is not None:
        filtered = filter_altitude(filtered, config.min_altitude_km, config.max_altitude_km)

    # 2. Apply region filter
    has_lat = config.lat_min_deg is not None and config.lat_max_deg is not None
    has_lon = config.lon_min_deg is not None and config.lon_max_deg is not None
    if has_lat and has_lon:
        # types in mypy expect float, but we verified not None
        filtered = filter_region(
            filtered,
            lat_min_deg=config.lat_min_deg,  # type: ignore[arg-type]
            lat_max_deg=config.lat_max_deg,  # type: ignore[arg-type]
            lon_min_deg=config.lon_min_deg,  # type: ignore[arg-type]
            lon_max_deg=config.lon_max_deg,  # type: ignore[arg-type]
        )

    # 3. Apply time window filter
    if config.t_start_jd is not None and config.t_end_jd is not None:
        filtered = filter_time_window(
            filtered,
            t_start_jd=config.t_start_jd,
            t_end_jd=config.t_end_jd,
        )

    # 4. Apply object type filter
    if config.object_types is not None:
        valid_types = set(config.object_types)
        filtered = [obj for obj in filtered if obj.object_class in valid_types]

    # 5. Apply max_objects cap
    if config.max_objects is not None and len(filtered) > config.max_objects:
        # The prompt specifies "max_objects cap (if provided)"
        filtered = filtered[: config.max_objects]

    return filtered
