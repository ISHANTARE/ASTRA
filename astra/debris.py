# astra/debris.py
"""ASTRA Core debris catalog filtering and statistics.

Pre-propagation filtering of debris catalogs. ALL filtering in this module
operates on DebrisObject parameters (derived TLE fields, NOT propagated
positions). Filtering is O(N) and involves no SGP4 calls.
"""

from __future__ import annotations

import math
from typing import Any, Optional

from astra.models import (
    DebrisObject,
    FilterConfig,
    SatelliteTLE,
    SatelliteOMM,
    SatelliteState,
)
from astra.utils import orbit_period, orbital_elements
from astra.constants import (
    EARTH_EQUATORIAL_RADIUS_KM,
    EARTH_MU_KM3_S2,
    TLE_AGE_LEO_MAX_DAYS,
    TLE_AGE_DEFAULT_MAX_DAYS,
)


def make_debris_object(source: SatelliteState) -> DebrisObject:
    """Build a DebrisObject from a SatelliteTLE or SatelliteOMM with derived elements.

    For TLE sources, elements are parsed directly from the raw TLE lines.
    For OMM sources, elements are read directly from the already-converted
    dataclass fields (no string parsing required).
    """
    if isinstance(source, SatelliteTLE):
        elements = orbital_elements(source.line2)
        period_min = orbit_period(elements["mean_motion_rev_per_day"])
        # Mean motion in rad/s for semi-major axis
        if period_min <= 0:
            n_rad_s = float("inf")
            a = 0.0
        else:
            n_rad_s = (2.0 * math.pi) / (period_min * 60.0)
            a = (EARTH_MU_KM3_S2 / (n_rad_s**2)) ** (1.0 / 3.0)
        e = elements["eccentricity"]
        inclination_deg = elements["inclination_deg"]
        raan_deg = elements["raan_deg"]

    elif isinstance(source, SatelliteOMM):
        # OMM already carries all elements as clean floats — no string parsing.
        # mean_motion_rad_min → convert to rev/day for orbit_period()
        mean_motion_rev_day = source.mean_motion_rad_min * 1440.0 / (2.0 * math.pi)
        period_min = orbit_period(mean_motion_rev_day)
        if period_min <= 0:
            n_rad_s = float("inf")
            a = 0.0
        else:
            n_rad_s = (2.0 * math.pi) / (period_min * 60.0)
            a = (EARTH_MU_KM3_S2 / (n_rad_s**2)) ** (1.0 / 3.0)
        e = source.eccentricity
        inclination_deg = math.degrees(source.inclination_rad)
        raan_deg = math.degrees(source.raan_rad)

    else:
        raise TypeError(f"Unsupported source type: {type(source).__name__}")

    perigee_km = a * (1.0 - e) - EARTH_EQUATORIAL_RADIUS_KM
    apogee_km = a * (1.0 + e) - EARTH_EQUATORIAL_RADIUS_KM
    altitude_km = a - EARTH_EQUATORIAL_RADIUS_KM

    # Harvest RCS from OMM if available
    rcs_m2 = getattr(source, "rcs_m2", None)

    return DebrisObject(
        source=source,
        altitude_km=altitude_km,
        inclination_deg=inclination_deg,
        period_minutes=period_min,
        raan_deg=raan_deg,
        eccentricity=e,
        apogee_km=apogee_km,
        perigee_km=perigee_km,
        object_class=source.object_type,
        rcs_m2=rcs_m2,
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
    lon_min_deg: Optional[float] = None,
    lon_max_deg: Optional[float] = None,
) -> list[DebrisObject]:
    """Retain objects whose ground track could pass through a bounding box.

    Uses a two-stage filter:

    **Stage 1 - Latitude (inclination-based):**
    An object with inclination *i* reaches latitudes up to ``i`` (prograde)
    or ``180 - i`` (retrograde). Objects whose latitude band does not
    overlap ``[lat_min_deg, lat_max_deg]`` are excluded immediately.

    **Stage 2 - Longitude (RAAN-based, if bounds are supplied):**
    For short-period orbits (period < 1440 min / 24 h), Earth's rotation
    causes the satellite's ground track to sweep all longitudes within a day,
    so the longitude filter is vacuous and all Stage-1 survivors are kept.
    For long-period orbits (GEO / HEO / deep-space), the ascending node is
    anchored near RAAN for the duration of interest, so a window check around
    RAAN is applied with a margin equal to the longitude swept in half an
    orbital period.  This remains an **over-inclusive** pre-filter.

    Note:
        For hard longitude exclusion use ``propagate_trajectory()`` followed
        by manual geodetic post-filtering.

    Args:
        objects: List of DebrisObjects.
        lat_min_deg: Minimum latitude bound (degrees).
        lat_max_deg: Maximum latitude bound (degrees).
        lon_min_deg: Minimum longitude bound (degrees, -180 to +180).
            ``None`` disables longitude filtering entirely.
        lon_max_deg: Maximum longitude bound (degrees, -180 to +180).
            ``None`` disables longitude filtering entirely.

    Returns:
        Filtered list of DebrisObjects.

    Example::

        # Retain objects that could pass over India (lat 8-37, lon 68-97)
        filtered = filter_region(
            objects,
            lat_min_deg=8.0, lat_max_deg=37.0,
            lon_min_deg=68.0, lon_max_deg=97.0,
        )
    """
    # [FM-2 Fix - Finding #5] Use `is not None` - NOT truthiness - so that
    # lon_min_deg=0.0 correctly activates the longitude filter path.
    apply_lon_filter = (lon_min_deg is not None) and (lon_max_deg is not None)

    results = []
    for obj in objects:
        # ── Stage 1: Latitude ────────────────────────────────────────────────
        inc = obj.inclination_deg
        max_lat_reached = inc if inc <= 90.0 else 180.0 - inc

        # Object latitude band: [-max_lat_reached, +max_lat_reached]
        if not (lat_min_deg <= max_lat_reached and lat_max_deg >= -max_lat_reached):
            continue  # latitude bands do not overlap → skip

        # ── Stage 2: Longitude (if bounds supplied) ──────────────────────────
        if apply_lon_filter:
            # [FM-1 Fix - Finding #1] RAAN + inclination-based longitude pre-filter.
            #
            # For orbits with period < 24 h (LEO/MEO), Earth's rotation (~360°/day)
            # means the ascending node sweeps every longitude within <= 1 day
            # regardless of RAAN. All such objects that pass Stage 1 are kept.
            period_min = obj.period_minutes
            if period_min <= 0.0 or period_min < 1440.0:
                # Short-period orbit: ascending node covers all longitudes in 24 h.
                results.append(obj)
                continue

            # Long-period orbit (GEO / HEO): ascending node stays near RAAN.
            # Apply a window check with margin for half-period nodal drift.
            raan = obj.raan_deg % 360.0       # normalise to [0, 360)
            lmin = float(lon_min_deg) % 360.0  # type: ignore[arg-type]
            lmax = float(lon_max_deg) % 360.0  # type: ignore[arg-type]

            # Half-period longitude drift: Earth rotates 360/day, so the node
            # drifts ~360*(P_min/1440)/2 degrees in half an orbital period.
            lon_half_sweep_deg = 360.0 * (period_min / 1440.0) / 2.0
            lo = (lmin - lon_half_sweep_deg) % 360.0
            hi = (lmax + lon_half_sweep_deg) % 360.0

            if lo <= hi:
                in_band = lo <= raan <= hi
            else:
                # Band wraps through 0 degrees (e.g. lo=350, hi=10)
                in_band = raan >= lo or raan <= hi

            if not in_band:
                continue  # RAAN outside reachable longitude band → skip

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
        age_days = t_start_jd - obj.source.epoch_jd

        # Stale thresholds
        # Stricter threshold for LEO due to higher atmospheric drag
        is_stale = False
        if obj.altitude_km < 2000:
            if age_days > TLE_AGE_LEO_MAX_DAYS:
                is_stale = True
        else:
            if age_days > TLE_AGE_DEFAULT_MAX_DAYS:
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
    if not objects:
        return {
            "total_count": 0,
            "by_type": {"PAYLOAD": 0, "ROCKET_BODY": 0, "DEBRIS": 0, "UNKNOWN": 0},
            "by_regime": {"LEO": 0, "MEO": 0, "GEO": 0, "HEO": 0},
            "altitude_mean_km": 0.0,
            "altitude_std_km": 0.0,
            "altitude_min_km": 0.0,
            "altitude_max_km": 0.0,
            "inclination_distribution": {
                "equatorial": 0,
                "inclined": 0,
                "polar": 0,
                "retrograde": 0,
            },
        }

    import numpy as np

    altitudes = np.array([obj.altitude_km for obj in objects])
    eccentricities = np.array([obj.eccentricity for obj in objects])
    inclinations = np.array([obj.inclination_deg for obj in objects])
    classes = [obj.object_class for obj in objects]

    # Regime classification via vectorized boolean indexing
    is_heo = eccentricities > 0.25
    is_leo = (~is_heo) & (altitudes < 2000)
    is_geo = (~is_heo) & (altitudes >= 35000) & (altitudes <= 36000)
    is_meo = (~is_heo) & (~is_leo) & (~is_geo)

    by_regime = {
        "LEO": int(np.sum(is_leo)),
        "MEO": int(np.sum(is_meo)),
        "GEO": int(np.sum(is_geo)),
        "HEO": int(np.sum(is_heo)),
    }

    # Inclination distribution
    inclination_dist = {
        "equatorial": int(np.sum(inclinations < 10.0)),
        "inclined": int(np.sum((inclinations >= 10.0) & (inclinations < 80.0))),
        "polar": int(np.sum((inclinations >= 80.0) & (inclinations <= 90.0))),
        "retrograde": int(np.sum(inclinations > 90.0)),
    }

    # Object type counts
    from collections import Counter

    type_counts = Counter(classes)
    by_type = {"PAYLOAD": 0, "ROCKET_BODY": 0, "DEBRIS": 0, "UNKNOWN": 0}
    by_type.update({k: v for k, v in type_counts.items() if k in by_type})

    return {
        "total_count": len(objects),
        "by_type": by_type,
        "by_regime": by_regime,
        "altitude_mean_km": float(np.mean(altitudes)),
        "altitude_std_km": float(np.std(altitudes)),
        "altitude_min_km": float(np.min(altitudes)),
        "altitude_max_km": float(np.max(altitudes)),
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
        filtered = filter_altitude(
            filtered, config.min_altitude_km, config.max_altitude_km
        )

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
