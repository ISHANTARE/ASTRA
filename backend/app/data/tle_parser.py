"""
ASTRA TLE Data Ingestion & Parsing
Fetches TLE data from CelesTrak and parses into OrbitalObjectDetail structures.
"""

import math
import logging
from datetime import datetime, timezone
from typing import Optional

import httpx

from app.core.config import settings
from app.models.schemas import OrbitalObjectDetail, ObjectType, OrbitRegion

logger = logging.getLogger(__name__)

# Earth radius in km
RE = settings.EARTH_RADIUS_KM


def classify_object_type(name: str, group: str) -> ObjectType:
    """Classify object type based on name and CelesTrak group."""
    name_upper = name.upper()
    if "DEB" in name_upper or "debris" in group:
        return ObjectType.DEBRIS
    elif "R/B" in name_upper or name_upper.startswith("ROCKET") or name_upper.endswith("R/B"):
        return ObjectType.ROCKET_BODY
    else:
        return ObjectType.SATELLITE


def classify_orbit_region(apogee_km: float, perigee_km: float, eccentricity: float) -> OrbitRegion:
    """Classify orbit region based on altitude parameters."""
    if eccentricity > 0.25 and apogee_km > settings.MEO_MIN_KM:
        return OrbitRegion.HEO
    avg_alt = (apogee_km + perigee_km) / 2.0
    if avg_alt <= settings.LEO_MAX_KM:
        return OrbitRegion.LEO
    elif avg_alt <= settings.MEO_MAX_KM:
        return OrbitRegion.MEO
    elif abs(avg_alt - settings.GEO_ALTITUDE_KM) < settings.GEO_TOLERANCE_KM:
        return OrbitRegion.GEO
    else:
        return OrbitRegion.MEO  # Default fallback


def derive_orbital_params(mean_motion: float, eccentricity: float) -> tuple[float, float, float]:
    """
    Derive apogee, perigee (km above Earth surface), and period (min)
    from mean motion (rev/day) and eccentricity.
    """
    if mean_motion <= 0:
        return (0.0, 0.0, 0.0)

    # Period in minutes
    period_min = 1440.0 / mean_motion

    # Semi-major axis from Kepler's third law
    # T = 2π * sqrt(a³/μ)  =>  a = (μ * (T/(2π))²)^(1/3)
    period_sec = period_min * 60.0
    a = (settings.EARTH_MU * (period_sec / (2.0 * math.pi)) ** 2) ** (1.0 / 3.0)

    # Apogee and perigee altitudes above Earth surface
    apogee_km = a * (1.0 + eccentricity) - RE
    perigee_km = a * (1.0 - eccentricity) - RE

    return (max(apogee_km, 0.0), max(perigee_km, 0.0), period_min)


def parse_tle_epoch(tle_line1: str) -> Optional[datetime]:
    """Extract epoch from TLE line 1."""
    try:
        epoch_str = tle_line1[18:32].strip()
        year_2d = int(epoch_str[:2])
        day_of_year = float(epoch_str[2:])

        year = 2000 + year_2d if year_2d < 57 else 1900 + year_2d

        epoch = datetime(year, 1, 1, tzinfo=timezone.utc)
        from datetime import timedelta
        epoch += timedelta(days=day_of_year - 1)
        return epoch
    except (ValueError, IndexError):
        return None


def parse_tle_lines(name: str, line1: str, line2: str, group: str = "active") -> Optional[OrbitalObjectDetail]:
    """
    Parse a single TLE entry (name + line1 + line2) into an OrbitalObjectDetail.
    """
    try:
        # Extract NORAD catalog number
        norad_id = int(line1[2:7].strip())

        # Extract orbital elements from TLE line 2
        inclination_deg = float(line2[8:16].strip())
        raan_deg = float(line2[17:25].strip())

        # Eccentricity: TLE stores it as implied decimal (e.g., "0012345" = 0.0012345)
        ecc_str = line2[26:33].strip()
        eccentricity = float(f"0.{ecc_str}")

        arg_perigee_deg = float(line2[34:42].strip())
        mean_anomaly_deg = float(line2[43:51].strip())
        mean_motion = float(line2[52:63].strip())

        # Derive orbital parameters
        apogee_km, perigee_km, period_min = derive_orbital_params(mean_motion, eccentricity)

        # Classify
        obj_type = classify_object_type(name, group)
        orbit_region = classify_orbit_region(apogee_km, perigee_km, eccentricity)
        epoch = parse_tle_epoch(line1)

        return OrbitalObjectDetail(
            norad_id=norad_id,
            name=name.strip(),
            object_type=obj_type,
            orbit_region=orbit_region,
            tle_line1=line1.strip(),
            tle_line2=line2.strip(),
            inclination_deg=inclination_deg,
            eccentricity=eccentricity,
            period_min=period_min,
            apogee_km=round(apogee_km, 2),
            perigee_km=round(perigee_km, 2),
            mean_motion=mean_motion,
            raan_deg=raan_deg,
            arg_perigee_deg=arg_perigee_deg,
            mean_anomaly_deg=mean_anomaly_deg,
            epoch=epoch,
        )
    except (ValueError, IndexError) as e:
        logger.warning(f"Failed to parse TLE for '{name}': {e}")
        return None


def parse_tle_text(tle_text: str, group: str = "active") -> list[OrbitalObjectDetail]:
    """
    Parse raw TLE text (3-line format) into a list of OrbitalObjectDetail.
    CelesTrak 3-line format: name, line1, line2 (repeating).
    """
    lines = [line.strip() for line in tle_text.strip().splitlines() if line.strip()]
    objects = []

    i = 0
    while i + 2 < len(lines):
        # Determine if current line is a name line or a TLE line 1
        if lines[i].startswith("1 ") and lines[i + 1].startswith("2 "):
            # 2-line format (no name), use NORAD ID as name
            name = f"OBJECT-{lines[i][2:7].strip()}"
            obj = parse_tle_lines(name, lines[i], lines[i + 1], group)
            if obj:
                objects.append(obj)
            i += 2
        elif not lines[i].startswith("1 ") and lines[i + 1].startswith("1 ") and i + 2 < len(lines) and lines[i + 2].startswith("2 "):
            # Standard 3-line format
            obj = parse_tle_lines(lines[i], lines[i + 1], lines[i + 2], group)
            if obj:
                objects.append(obj)
            i += 3
        else:
            i += 1  # Skip malformed lines

    return objects


async def fetch_tle_group(group: str) -> list[OrbitalObjectDetail]:
    """Fetch and parse TLE data for a single CelesTrak group."""
    url = f"{settings.CELESTRAK_BASE_URL}?GROUP={group}&FORMAT=tle"
    logger.info(f"Fetching TLE data from: {url}")

    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            response = await client.get(url)
            response.raise_for_status()
            objects = parse_tle_text(response.text, group)
            logger.info(f"Parsed {len(objects)} objects from group '{group}'")
            return objects
        except httpx.HTTPError as e:
            logger.error(f"Failed to fetch TLE group '{group}': {e}")
            return []


async def fetch_all_tles() -> list[OrbitalObjectDetail]:
    """Fetch TLE data from all configured CelesTrak groups."""
    all_objects = []
    seen_ids = set()

    for group in settings.CELESTRAK_GROUPS:
        objects = await fetch_tle_group(group)
        for obj in objects:
            if obj.norad_id not in seen_ids:
                seen_ids.add(obj.norad_id)
                all_objects.append(obj)

    logger.info(f"Total unique objects fetched: {len(all_objects)}")
    return all_objects
