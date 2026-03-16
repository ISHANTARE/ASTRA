"""
ASTRA Conjunction Analysis Engine
Distance calculations, TCA detection, relative velocity computation, and risk classification.
All distance calculations use NumPy vectorization (doc 08 requirement).
"""

import logging
from datetime import datetime

import numpy as np

from app.core.config import settings
from app.models.schemas import ConjunctionEvent, RiskClassification

logger = logging.getLogger(__name__)


def compute_distances(positions_a: np.ndarray, positions_b: np.ndarray) -> np.ndarray:
    """
    Compute Euclidean distances between two objects across all time steps.
    Uses vectorized NumPy operations per doc 08 requirement.

    Args:
        positions_a: (N, 3) array of positions for object A in km
        positions_b: (N, 3) array of positions for object B in km

    Returns:
        (N,) array of distances in km
    """
    # Vectorized: distances = √((x1-x2)² + (y1-y2)² + (z1-z2)²)
    return np.linalg.norm(positions_a - positions_b, axis=1)


def classify_risk(distance_km: float) -> RiskClassification:
    """
    Classify risk based on closest approach distance.
    Per doc 03:
      < 100m (0.1 km) → Extreme
      < 1 km           → High Risk
      < 10 km          → Conjunction
      else             → Safe
    """
    if distance_km < settings.RISK_EXTREME_KM:
        return RiskClassification.EXTREME
    elif distance_km < settings.RISK_HIGH_KM:
        return RiskClassification.HIGH_RISK
    elif distance_km < settings.RISK_CONJUNCTION_KM:
        return RiskClassification.CONJUNCTION
    else:
        return RiskClassification.SAFE


def detect_conjunction(
    positions_a: np.ndarray,
    positions_b: np.ndarray,
    velocities_a: np.ndarray,
    velocities_b: np.ndarray,
    simulation_times: list[datetime],
    name_a: str,
    name_b: str,
    id_a: int,
    id_b: int,
    distance_threshold_km: float = None,
) -> ConjunctionEvent | None:
    """
    Detect a conjunction event between two objects.

    Per doc 03, the logic is:
      1. Calculate distances across all 288 simulation time steps.
      2. Find the index of minimum distance (argmin).
      3. Extract closest_distance and TCA timestamp.
      4. Retrieve velocity vectors at TCA index.
      5. Compute relative velocity: |v1 - v2|

    Args:
        positions_a/b: (N, 3) precomputed trajectory arrays in km (TEME)
        velocities_a/b: (N, 3) precomputed velocity arrays in km/s (TEME)
        simulation_times: list of N datetime objects
        name_a/b: object names
        id_a/b: NORAD IDs
        distance_threshold_km: max distance to report (default: 10 km)

    Returns:
        ConjunctionEvent if closest approach is within threshold, else None
    """
    if distance_threshold_km is None:
        distance_threshold_km = settings.RISK_CONJUNCTION_KM

    # Check for NaN positions (failed propagation)
    valid_mask = ~(np.any(np.isnan(positions_a), axis=1) | np.any(np.isnan(positions_b), axis=1))
    if not np.any(valid_mask):
        return None

    # Calculate distances across all time steps (vectorized)
    distances = compute_distances(positions_a, positions_b)

    # Mask out invalid time steps
    distances[~valid_mask] = np.inf

    # Find TCA (Time of Closest Approach)
    tca_index = np.argmin(distances)
    closest_distance = distances[tca_index]

    # Only report if within threshold
    if closest_distance > distance_threshold_km or np.isinf(closest_distance):
        return None

    # Extract TCA timestamp
    tca_time = simulation_times[tca_index]

    # Compute relative velocity at TCA: |v1 - v2|
    v1 = velocities_a[tca_index]
    v2 = velocities_b[tca_index]
    relative_velocity = np.linalg.norm(v1 - v2)

    # Classify risk
    risk = classify_risk(closest_distance)

    return ConjunctionEvent(
        object_1=name_a,
        object_2=name_b,
        object_1_id=id_a,
        object_2_id=id_b,
        closest_distance_km=round(float(closest_distance), 4),
        time_of_closest_approach_utc=tca_time,
        relative_velocity_km_s=round(float(relative_velocity), 4),
        risk_classification=risk,
    )


def analyze_candidates(
    candidate_pairs: list[tuple[int, int]],
    trajectories: dict[int, tuple[np.ndarray, np.ndarray]],
    simulation_times: list[datetime],
    object_names: dict[int, str],
    distance_threshold_km: float = None,
) -> list[ConjunctionEvent]:
    """
    Run conjunction analysis on all candidate pairs using precomputed trajectories.

    This function operates ONLY on precomputed trajectory arrays.
    No propagation is performed here (doc 08 requirement).

    Args:
        candidate_pairs: list of (id_a, id_b) tuples from filtering pipeline
        trajectories: dict mapping norad_id → (positions, velocities) arrays
        simulation_times: list of datetime objects for each time step
        object_names: dict mapping norad_id → object name
        distance_threshold_km: max distance to report as conjunction

    Returns:
        List of detected ConjunctionEvent objects, sorted by closest distance.
    """
    events = []
    analyzed = 0
    skipped = 0

    for id_a, id_b in candidate_pairs:
        # Skip if either object's trajectory is missing
        if id_a not in trajectories or id_b not in trajectories:
            skipped += 1
            continue

        pos_a, vel_a = trajectories[id_a]
        pos_b, vel_b = trajectories[id_b]

        event = detect_conjunction(
            positions_a=pos_a,
            positions_b=pos_b,
            velocities_a=vel_a,
            velocities_b=vel_b,
            simulation_times=simulation_times,
            name_a=object_names.get(id_a, f"OBJECT-{id_a}"),
            name_b=object_names.get(id_b, f"OBJECT-{id_b}"),
            id_a=id_a,
            id_b=id_b,
            distance_threshold_km=distance_threshold_km,
        )

        if event is not None:
            events.append(event)

        analyzed += 1

    # Sort by closest distance (most dangerous first)
    events.sort(key=lambda e: e.closest_distance_km)

    logger.info(
        f"Conjunction analysis complete: {analyzed} pairs analyzed, "
        f"{skipped} skipped, {len(events)} events detected"
    )
    return events
