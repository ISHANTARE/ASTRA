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
    
    Valid groups include: 'active', '1999-025', 'iridium-33-debris', etc.
    """
    url = f"https://celestrak.org/NORAD/elements/gp.php?GROUP={group}&FORMAT=tle"
    try:
        response = requests.get(url, timeout=20.0)
        response.raise_for_status()
    except requests.RequestException as e:
        raise AstraError(f"Failed to fetch Celestrak group '{group}': {e}")
        
    lines = response.text.splitlines()
    return load_tle_catalog(lines)


def fetch_celestrak_comprehensive() -> list[SatelliteTLE]:
    """Fetch active payloads plus major debris clouds for a pseudo-full catalog tracking.
    
    Since Celestrak doesn't expose a single unauthenticated 'all' endpoint roughly equal 
    to Space-Track's 35k total, we assemble the ~25,000+ most important objects.
    """
    groups = [
        "active",             # All active payloads (~15k)
        "1999-025",           # Fengyun-1C debris (~3k)
        "iridium-33-debris",  # Iridium 33 debris (~300)
        "cosmos-2251-debris", # Cosmos 2251 debris (~1k)
        "1982-092",           # Cosmos 1408 debris (~500)
        "2019-006",           # MICROSAT-R debris (~100)
        "analyst",            # Analyst objects
    ]
    
    seen_ids = set()
    unified_catalog = []
    
    for g in groups:
        try:
            tles = fetch_celestrak_group(g)
            for tle in tles:
                if tle.norad_id not in seen_ids:
                    seen_ids.add(tle.norad_id)
                    unified_catalog.append(tle)
        except AstraError:
            pass # Skip if a specific group fails
            
    return unified_catalog
