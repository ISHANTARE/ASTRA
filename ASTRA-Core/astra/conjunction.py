"""ASTRA Core conjunction detection module.

Detects close-approach events between pairs of orbital objects. Implements an 
advanced 3-phase optimization algorithm including:
1. Sweep-and-Prune Radial Bounding Shell filter
2. Trajectory AABB volume filter
3. Cubicpline Curvilinear Interpolation for exact sub-second TCA
4. Realistic Encounter-Plane Covariance Probability
"""
from __future__ import annotations

import concurrent.futures
import numpy as np
import scipy.interpolate

from astra.errors import AstraError
from astra.models import ConjunctionEvent, DebrisObject, TrajectoryMap
from astra.covariance import compute_collision_probability, estimate_covariance
from astra.log import get_logger

logger = get_logger(__name__)



def distance_3d(pos_a: np.ndarray, pos_b: np.ndarray) -> np.ndarray:
    """Compute Euclidean distance between two position arrays."""
    return np.linalg.norm(pos_a - pos_b, axis=-1)


def _classify_risk(P_c: float) -> str:
    """Classify risk level based on collision probability."""
    if P_c > 1e-4: return "CRITICAL"
    if P_c > 1e-5: return "HIGH"
    if P_c > 1e-6: return "MEDIUM"
    return "LOW"

def closest_approach(
    trajectory_a: np.ndarray, trajectory_b: np.ndarray, times_jd: np.ndarray
) -> tuple[float, float, int]:
    """Find the exact minimum separation using cubic splines."""
    coarse_dists = distance_3d(trajectory_a, trajectory_b)
    t_idx = int(np.argmin(coarse_dists))
    
    if len(times_jd) < 3:
        return float(coarse_dists[t_idx]), float(times_jd[t_idx]), t_idx
        
    spline_A = scipy.interpolate.CubicSpline(times_jd, trajectory_a, bc_type='natural')
    spline_B = scipy.interpolate.CubicSpline(times_jd, trajectory_b, bc_type='natural')
    
    idx_low = max(0, t_idx - 1)
    idx_high = min(len(times_jd) - 1, t_idx + 1)
    
    t_dense = np.linspace(times_jd[idx_low], times_jd[idx_high], 100)
    rA_dense = spline_A(t_dense)
    rB_dense = spline_B(t_dense)
    
    dense_dists = distance_3d(rA_dense, rB_dense)
    t_dense_idx = int(np.argmin(dense_dists))
    
    return float(dense_dists[t_dense_idx]), float(t_dense[t_dense_idx]), t_idx


def find_conjunctions(
    trajectories: TrajectoryMap,
    times_jd: np.ndarray,
    elements_map: dict[str, DebrisObject],
    threshold_km: float = 5.0,
    coarse_threshold_km: float = 50.0,
) -> list[ConjunctionEvent]:
    """Find highly precise conjunction events using cubic spline interpolation."""
    norad_ids = list(trajectories.keys())
    if not norad_ids:
        return []

    T_len = len(times_jd)
    if T_len < 3:
        raise AstraError("At least 3 timesteps required for CubicSpline interpolation.")

    logger.info(f"Initiating Conjunction Analysis for {len(norad_ids)} objects over {T_len} time steps.")

    # ---------------------------------------------------------
    # Phase 1: Radial Bounding Shells (Sweep-And-Prune)
    # ---------------------------------------------------------
    logger.debug("Running Phase 1: 1D Sweep-and-Prune Radial Bounding...")
    intervals = []
    for nid in norad_ids:
        obj = elements_map.get(nid)
        if obj is None:
            raise AstraError(f"DebrisObject for NORAD {nid} missing in elements_map.")
        
        min_r = obj.perigee_km + 6371.0
        max_r = obj.apogee_km + 6371.0
        intervals.append((min_r, max_r, nid))
        
    intervals.sort(key=lambda x: x[0])
    candidate_pairs_phase1 = []
    active = []
    margin = coarse_threshold_km
    
    for min_r, max_r, nid in intervals:
        active = [a for a in active if a[1] + margin >= min_r]
        for a_min, a_max, a_nid in active:
            candidate_pairs_phase1.append((a_nid, nid))
        active.append((min_r, max_r, nid))

    if not candidate_pairs_phase1:
        logger.info("Sweep-and-Prune complete: 0 candidate pairs found.")
        return []

    logger.debug(f"Phase 1 Complete: Pruned to {len(candidate_pairs_phase1)} radial collision candidates.")

    # ---------------------------------------------------------
    # Phase 2: Trajectory Volume Cartesian AABB
    # ---------------------------------------------------------
    logger.debug("Running Phase 2: Cartesian Axis-Aligned Bounding Box Intersection...")
    traj_bounds = {}
    for nid in norad_ids:
        traj = trajectories[nid]
        min_xyz = np.nanmin(traj, axis=0) # shape (3,)
        max_xyz = np.nanmax(traj, axis=0)
        traj_bounds[nid] = (min_xyz, max_xyz)

    candidate_pairs_phase2 = set()
    for A, B in candidate_pairs_phase1:
        minA, maxA = traj_bounds[A]
        minB, maxB = traj_bounds[B]
        
        if np.all((minA - margin <= maxB) & (maxA + margin >= minB)):
            candidate_pairs_phase2.add((min(A, B), max(A, B)))

    logger.info(f"AABB Filter Complete: Analyzing precise geometry for {len(candidate_pairs_phase2)} pairs.")

    # ---------------------------------------------------------
    # Phase 3: Exact Curvilinear TCA Interpolation
    # ---------------------------------------------------------
    logger.debug("Running Phase 3: Root-finding via Spline interpolation & Mahalanobis B-Plane Covariance Projection...")
    events = []

    def evaluate_pair(pair: tuple[str, str]) -> ConjunctionEvent | None:
        """Worker function for concurrent execution."""
        A, B = pair
        traj_A = trajectories[A]
        traj_B = trajectories[B]
        
        # 1. Coarse search to find local minimum window
        coarse_dists = distance_3d(traj_A, traj_B)
        t_idx = int(np.argmin(coarse_dists))
        coarse_min = coarse_dists[t_idx]
        
        if coarse_min > coarse_threshold_km:
            return None
            
        # 2. Build Cubic Splines for exact curvilinear motion mapping
        spline_A = scipy.interpolate.CubicSpline(times_jd, traj_A, bc_type='natural')
        spline_B = scipy.interpolate.CubicSpline(times_jd, traj_B, bc_type='natural')
        
        # 3. Dense 1-second resolution evaluation across the local min bracket
        idx_low = max(0, t_idx - 1)
        idx_high = min(T_len - 1, t_idx + 1)
        
        seconds_in_bracket = int((times_jd[idx_high] - times_jd[idx_low]) * 86400.0)
        if seconds_in_bracket < 2:
            seconds_in_bracket = 2
            
        t_dense = np.linspace(times_jd[idx_low], times_jd[idx_high], seconds_in_bracket)
        
        rA_dense = spline_A(t_dense)
        rB_dense = spline_B(t_dense)
        
        dense_dists = distance_3d(rA_dense, rB_dense)
        tca_dense_idx = int(np.argmin(dense_dists))
        
        min_dist = float(dense_dists[tca_dense_idx])
        
        if min_dist > threshold_km:
            return None
            
        tca_jd = float(t_dense[tca_dense_idx])
        
        pos_A = rA_dense[tca_dense_idx]
        pos_B = rB_dense[tca_dense_idx]
        
        vel_A = spline_A(tca_jd, nu=1) / 86400.0
        vel_B = spline_B(tca_jd, nu=1) / 86400.0
        
        rel_vel_vec = vel_A - vel_B
        rel_vel = float(np.linalg.norm(rel_vel_vec))
        
        obj_A = elements_map[A]
        obj_B = elements_map[B]
        
        days_since_epoch_A = tca_jd - obj_A.tle.epoch_jd
        days_since_epoch_B = tca_jd - obj_B.tle.epoch_jd
        
        cov_A = estimate_covariance(days_since_epoch_A)
        cov_B = estimate_covariance(days_since_epoch_B)
        
        miss_vector = pos_A - pos_B
        
        P_c = compute_collision_probability(
            miss_vector, rel_vel_vec, cov_A, cov_B, combined_radius_km=0.010
        )
        
        risk = _classify_risk(P_c)
        
        return ConjunctionEvent(
            object_a_id=A,
            object_b_id=B,
            tca_jd=tca_jd,
            miss_distance_km=min_dist,
            relative_velocity_km_s=rel_vel,
            collision_probability=P_c,
            risk_level=risk,
            position_a_km=pos_A,
            position_b_km=pos_B
        )

    # Execute geometric evaluations concurrently across all CPU threads
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {executor.submit(evaluate_pair, pair): pair for pair in candidate_pairs_phase2}
        
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                if result is not None:
                    events.append(result)
            except Exception as e:
                logger.error(f"Error evaluating conjunction pair: {e}")

    events.sort(key=lambda x: x.miss_distance_km)
    logger.info(f"Conjunction Sweep Complete: {len(events)} events detected.")
    return events
