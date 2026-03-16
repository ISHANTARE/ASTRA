"""
ASTRA Orbital Congestion Analysis
Computes traffic density by altitude bands.
"""

import logging
from collections import defaultdict

from app.core.config import settings
from app.models.schemas import CongestionBand, CongestionResponse

logger = logging.getLogger(__name__)


def compute_congestion(
    objects: list[dict],
    band_width_km: float = None,
    max_altitude_km: float = 2000.0,
) -> CongestionResponse:
    """
    Compute orbital traffic density by altitude bands.

    Args:
        objects: list of object dicts with 'perigee_km', 'apogee_km', 'object_type'
        band_width_km: width of each altitude band in km
        max_altitude_km: maximum altitude to analyze

    Returns:
        CongestionResponse with bands and totals
    """
    if band_width_km is None:
        band_width_km = settings.ALTITUDE_BAND_WIDTH_KM

    # Initialize altitude bands
    bands = {}
    alt = 0
    while alt < max_altitude_km:
        band_key = (alt, alt + band_width_km)
        bands[band_key] = {
            "total": 0,
            "satellites": 0,
            "debris": 0,
            "rocket_bodies": 0,
        }
        alt += band_width_km

    # Count objects per band (object belongs to a band if its avg altitude falls within)
    for obj in objects:
        perigee = obj.get("perigee_km", 0)
        apogee = obj.get("apogee_km", 0)
        avg_alt = (perigee + apogee) / 2.0
        obj_type = obj.get("object_type", "unknown")

        # Find the band this object belongs to
        band_idx = int(avg_alt / band_width_km)
        band_min = band_idx * band_width_km
        band_max = band_min + band_width_km
        band_key = (band_min, band_max)

        if band_key in bands:
            bands[band_key]["total"] += 1
            if obj_type == "satellite":
                bands[band_key]["satellites"] += 1
            elif obj_type == "debris":
                bands[band_key]["debris"] += 1
            elif obj_type == "rocket_body":
                bands[band_key]["rocket_bodies"] += 1

    # Convert to response model
    band_list = []
    for (alt_min, alt_max), counts in sorted(bands.items()):
        if counts["total"] > 0:
            band_list.append(CongestionBand(
                altitude_min_km=alt_min,
                altitude_max_km=alt_max,
                total_objects=counts["total"],
                satellites=counts["satellites"],
                debris=counts["debris"],
                rocket_bodies=counts["rocket_bodies"],
            ))

    total = sum(b.total_objects for b in band_list)

    logger.info(f"Congestion analysis: {total} objects across {len(band_list)} altitude bands")

    return CongestionResponse(
        bands=band_list,
        total_objects=total,
    )
