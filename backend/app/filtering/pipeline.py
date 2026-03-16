"""
ASTRA Multi-Stage Filtering Pipeline
Implements the documented 6-stage filtering pipeline to reduce O(N²) comparisons.
Pipeline: Region → Altitude → Orbital Plane → Intersection → Spatial Grid → Candidate Pairs
"""

import math
import logging
from collections import defaultdict
from typing import Optional

import numpy as np

from app.core.config import settings
from app.models.schemas import OrbitalObjectDetail, OrbitRegion

logger = logging.getLogger(__name__)

# Earth parameters
RE = settings.EARTH_RADIUS_KM


# =============================================================================
# Stage 1: Region Filter
# =============================================================================

def filter_by_region(objects: list[dict], region: OrbitRegion) -> list[dict]:
    """
    Stage 1: Filter objects by orbital region.
    If region is ALL, return all objects unchanged.
    """
    if region == OrbitRegion.ALL:
        return objects

    filtered = [obj for obj in objects if obj["orbit_region"] == region.value]
    logger.info(f"Stage 1 (Region={region.value}): {len(objects)} → {len(filtered)} objects")
    return filtered


# =============================================================================
# Stage 2: Altitude Filter
# =============================================================================

def filter_by_altitude(
    objects: list[dict],
    alt_min: float,
    alt_max: float,
    margin: float = None,
) -> list[dict]:
    """
    Stage 2: Filter objects with overlapping altitude ranges.
    Objects must have perigee/apogee that overlaps [alt_min - margin, alt_max + margin].
    """
    if margin is None:
        margin = settings.ALTITUDE_FILTER_MARGIN_KM

    range_min = alt_min - margin
    range_max = alt_max + margin

    filtered = []
    for obj in objects:
        perigee = obj.get("perigee_km", 0)
        apogee = obj.get("apogee_km", 0)
        # Check if object's altitude range overlaps with the target range
        if perigee <= range_max and apogee >= range_min:
            filtered.append(obj)

    logger.info(
        f"Stage 2 (Altitude {alt_min}-{alt_max}km ±{margin}km): "
        f"{len(objects)} → {len(filtered)} objects"
    )
    return filtered


# =============================================================================
# Stage 3: Orbital Plane Filter
# =============================================================================

def filter_by_orbital_plane(
    objects: list[dict],
    threshold_deg: float = None,
) -> list[list[dict]]:
    """
    Stage 3: Group objects by inclination similarity.
    Returns groups of objects with similar inclinations (within threshold).
    Only objects within the same group can be compared.
    """
    if threshold_deg is None:
        threshold_deg = settings.INCLINATION_THRESHOLD_DEG

    if not objects:
        return []

    # Sort by inclination
    sorted_objs = sorted(objects, key=lambda o: o.get("inclination_deg", 0))

    # Group objects with similar inclinations
    groups = []
    current_group = [sorted_objs[0]]
    current_inc = sorted_objs[0].get("inclination_deg", 0)

    for obj in sorted_objs[1:]:
        inc = obj.get("inclination_deg", 0)
        if abs(inc - current_inc) <= threshold_deg:
            current_group.append(obj)
        else:
            if len(current_group) > 1:
                groups.append(current_group)
            current_group = [obj]
            current_inc = inc

    if len(current_group) > 1:
        groups.append(current_group)

    total_in_groups = sum(len(g) for g in groups)
    logger.info(
        f"Stage 3 (Orbital Plane, threshold={threshold_deg}°): "
        f"{len(objects)} objects → {len(groups)} groups, {total_in_groups} objects in groups"
    )
    return groups


# =============================================================================
# Stage 4: Orbit Intersection Check
# =============================================================================

def can_orbits_intersect(obj_a: dict, obj_b: dict) -> bool:
    """
    Stage 4: Geometric check — can two orbits physically intersect?
    Two orbits can intersect if their altitude ranges overlap.
    This is a coarse check using perigee/apogee.
    """
    a_perigee = obj_a.get("perigee_km", 0)
    a_apogee = obj_a.get("apogee_km", 0)
    b_perigee = obj_b.get("perigee_km", 0)
    b_apogee = obj_b.get("apogee_km", 0)

    # Orbits can intersect if altitude ranges overlap
    return a_perigee <= b_apogee and b_perigee <= a_apogee


def filter_intersecting_pairs(group: list[dict]) -> list[tuple[dict, dict]]:
    """
    Stage 4: From a group of objects with similar inclinations,
    filter to only pairs whose orbits can physically intersect.
    """
    pairs = []
    for i in range(len(group)):
        for j in range(i + 1, len(group)):
            if can_orbits_intersect(group[i], group[j]):
                pairs.append((group[i], group[j]))
    return pairs


# =============================================================================
# Stage 5: Orbital Spatial Grid Indexing
# =============================================================================

def compute_grid_key(
    obj: dict,
    alt_band_width: float = None,
    sector_size_deg: float = None,
) -> tuple:
    """
    Compute spatial grid key for an object.
    Grid is defined by (altitude_band, angular_sector) using RAAN.
    """
    if alt_band_width is None:
        alt_band_width = settings.ALTITUDE_BAND_WIDTH_KM
    if sector_size_deg is None:
        sector_size_deg = settings.ANGULAR_SECTOR_SIZE_DEG

    avg_alt = (obj.get("perigee_km", 0) + obj.get("apogee_km", 0)) / 2.0
    alt_band = int(avg_alt / alt_band_width)

    raan = obj.get("raan_deg", 0)
    sector = int(raan / sector_size_deg) % int(360 / sector_size_deg)

    return (alt_band, sector)


def build_spatial_grid(objects: list[dict]) -> dict[tuple, list[dict]]:
    """
    Stage 5: Build spatial grid index mapping grid cells to objects.
    """
    grid = defaultdict(list)
    for obj in objects:
        key = compute_grid_key(obj)
        grid[key].append(obj)

    logger.info(
        f"Stage 5 (Spatial Grid): {len(objects)} objects → {len(grid)} grid cells"
    )
    return dict(grid)


# =============================================================================
# Stage 6: Candidate Pair Generation
# =============================================================================

def get_adjacent_cells(cell: tuple) -> list[tuple]:
    """Get a grid cell and its adjacent cells (including diagonals)."""
    alt_band, sector = cell
    max_sectors = int(360 / settings.ANGULAR_SECTOR_SIZE_DEG)
    adjacent = []

    for da in [-1, 0, 1]:
        for ds in [-1, 0, 1]:
            adj_alt = alt_band + da
            adj_sec = (sector + ds) % max_sectors
            if adj_alt >= 0:
                adjacent.append((adj_alt, adj_sec))

    return adjacent


def generate_candidate_pairs(grid: dict[tuple, list[dict]]) -> list[tuple[int, int]]:
    """
    Stage 6: Generate candidate pairs from the spatial grid.
    Only pairs within the same cell or adjacent cells are considered.
    Returns list of (norad_id_a, norad_id_b) tuples.
    """
    seen_pairs = set()
    candidate_pairs = []

    for cell, objects in grid.items():
        # Same-cell pairs
        for i in range(len(objects)):
            for j in range(i + 1, len(objects)):
                id_a = objects[i]["norad_id"]
                id_b = objects[j]["norad_id"]
                pair = (min(id_a, id_b), max(id_a, id_b))
                if pair not in seen_pairs:
                    seen_pairs.add(pair)
                    candidate_pairs.append(pair)

        # Adjacent-cell pairs
        for adj_cell in get_adjacent_cells(cell):
            if adj_cell != cell and adj_cell in grid:
                for obj_a in objects:
                    for obj_b in grid[adj_cell]:
                        id_a = obj_a["norad_id"]
                        id_b = obj_b["norad_id"]
                        pair = (min(id_a, id_b), max(id_a, id_b))
                        if pair not in seen_pairs:
                            seen_pairs.add(pair)
                            candidate_pairs.append(pair)

    logger.info(f"Stage 6 (Candidate Pairs): Generated {len(candidate_pairs)} candidate pairs")
    return candidate_pairs


# =============================================================================
# Full Pipeline
# =============================================================================

def run_filtering_pipeline(
    objects: list[dict],
    region: OrbitRegion = OrbitRegion.LEO,
    altitude_range: tuple[float, float] = (400, 600),
) -> tuple[list[dict], list[tuple[int, int]]]:
    """
    Execute the full multi-stage filtering pipeline.

    Args:
        objects: list of object dicts from database
        region: target orbital region
        altitude_range: (min_km, max_km) altitude range

    Returns:
        Tuple of (filtered_objects, candidate_pairs)
        where candidate_pairs is a list of (norad_id_a, norad_id_b) tuples.
    """
    logger.info(f"Starting filtering pipeline with {len(objects)} objects")

    # Stage 1: Region filter
    filtered = filter_by_region(objects, region)

    if not filtered:
        logger.info("No objects remaining after region filter")
        return [], []

    # Stage 2: Altitude filter
    filtered = filter_by_altitude(filtered, altitude_range[0], altitude_range[1])

    if not filtered:
        logger.info("No objects remaining after altitude filter")
        return [], []

    # Stage 3: Orbital plane filter → groups
    groups = filter_by_orbital_plane(filtered)

    if not groups:
        logger.info("No groups formed after orbital plane filter")
        return filtered, []

    # Stage 4: Orbit intersection check (within each group)
    all_intersecting_objects = set()
    for group in groups:
        for obj in group:
            all_intersecting_objects.add(obj["norad_id"])

    # Keep only objects that are in intersecting groups
    filtered = [obj for obj in filtered if obj["norad_id"] in all_intersecting_objects]

    # Stage 5: Spatial grid indexing
    grid = build_spatial_grid(filtered)

    # Stage 6: Candidate pair generation
    candidate_pairs = generate_candidate_pairs(grid)

    logger.info(
        f"Pipeline complete: {len(objects)} → {len(filtered)} objects, "
        f"{len(candidate_pairs)} candidate pairs"
    )
    return filtered, candidate_pairs
