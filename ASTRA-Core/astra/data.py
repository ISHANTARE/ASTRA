"""ASTRA Core Data Ingestion Module.

Bridges the computation engine to live orbital data providers like CelesTrak 
and Space-Track. Converts real-time API responses into ASTRA trajectory pipelines.
"""
from __future__ import annotations

import requests

from astra.errors import AstraError
from astra.models import SatelliteTLE
from astra.tle import load_tle_catalog

CELESTRAK_ACTIVE_URL = "https://celestrak.org/NORAD/elements/gp.php?GROUP=active&FORMAT=tle"


def fetch_celestrak_active() -> list[SatelliteTLE]:
    """Fetch the active satellite catalog from CelesTrak.
    
    Downloads the entire live active catalog and parses it into ASTRA data models.
    """
    try:
        response = requests.get(CELESTRAK_ACTIVE_URL, timeout=20.0)
        response.raise_for_status()
    except requests.RequestException as e:
        raise AstraError(f"Failed to fetch Celestrak active catalog: {e}")
        
    lines = response.text.splitlines()
    return load_tle_catalog(lines)


def fetch_celestrak_group(group: str) -> list[SatelliteTLE]:
    """Fetch a specific constellation/group from CelesTrak.
    
    Valid groups include: 'starlink', 'oneweb', 'iridium', 'planet', 'spire', etc.
    """
    url = f"https://celestrak.org/NORAD/elements/gp.php?GROUP={group}&FORMAT=tle"
    try:
        response = requests.get(url, timeout=15.0)
        response.raise_for_status()
    except requests.RequestException as e:
        raise AstraError(f"Failed to fetch Celestrak group '{group}': {e}")
        
    lines = response.text.splitlines()
    return load_tle_catalog(lines)
