"""ASTRA Core conjunction detection module.

Detects close-approach events between pairs of orbital objects. Implements an 
advanced 3-phase optimization algorithm including:
1. Sweep-and-Prune Radial Bounding Shell filter
2. Trajectory AABB volume filter
3. Cubicspline Curvilinear Interpolation for exact sub-second TCA
4. Realistic Encounter-Plane Covariance Probability
"""
from __future__ import annotations

import concurrent.futures
import os
from typing import Optional
import numpy as np
import scipy.interpolate

from astra.errors import AstraError, PropagationError
from astra.models import ConjunctionEvent, DebrisObject, TrajectoryMap, projected_area_m2
from astra.covariance import (
    compute_collision_probability,
    estimate_covariance,
    rotate_covariance_rtn_to_eci,
)
from astra.spacebook import fetch_synthetic_covariance_stk, SPACEBOOK_ENABLED
from astra.log import get_logger
from astra.spatial_index import SpatialIndex

logger = get_logger(__name__)



def distance_3d(pos_a: np.ndarray, pos_b: np.ndarray) -> np.ndarray:
    """Compute Euclidean distance between two position arrays."""
    return np.linalg.norm(pos_a - pos_b, axis=-1)


def _classify_risk(P_c: Optional[float]) -> str:
    """Classify risk level based on collision probability."""
    if P_c is None: return "UNKNOWN"
    if P_c > 1e-4: return "CRITICAL"
    if P_c > 1e-5: return "HIGH"
    if P_c > 1e-6: return "MEDIUM"
    return "LOW"


def _dynamic_radius_km(
    obj: DebrisObject, 
    rel_vel_hat: np.ndarray, 
    pos_eci: np.ndarray, 
    vel_eci: np.ndarray
) -> float:
    """Compute effective collision radius from dynamic attitude or fallback.

    Supports both ``SatelliteTLE`` (with dimensions_m / attitude fields) and
    ``SatelliteOMM`` (which carries authoritative ``rcs_m2``). Uses safe
    ``getattr`` fallbacks so neither format raises an ``AttributeError``.

    Priority order:
        1. TLE with explicit dimensions + attitude model   → exact projected area
        2. OMM or TLE with ``rcs_m2``                     → sqrt(RCS / π) equivalent sphere
        3. DebrisObject-level ``radius_m``                 → direct radius
        4. Hard fallback                                   → 5 m sphere
    """
    source = obj.source
    import math

    # ------------------------------------------------------------------
    # Path 1: detailed geometry from SatelliteTLE dimensions
    # ------------------------------------------------------------------
    dimensions_m = getattr(source, "dimensions_m", None)
    if dimensions_m is not None:
        l, w, h = dimensions_m
        attitude_mode = getattr(source, "attitude_mode", "TUMBLING")

        if attitude_mode == "TUMBLING":
            area_m2 = 2.0 * (l*w + w*h + h*l) / 4.0

        elif attitude_mode == "NADIR":
            pos_hat = pos_eci / max(np.linalg.norm(pos_eci), 1e-12)
            vel_hat = vel_eci / max(np.linalg.norm(vel_eci), 1e-12)
            normal_hat = np.cross(pos_hat, vel_hat)
            normal_hat /= max(np.linalg.norm(normal_hat), 1e-12)
            x_hat = np.cross(normal_hat, pos_hat)
            faces = [
                (x_hat, w * h),
                (normal_hat, l * h),
                (pos_hat, l * w),
            ]
            area_m2 = sum(abs(float(np.dot(n, rel_vel_hat))) * a for n, a in faces)

        elif attitude_mode == "INERTIAL":
            attitude_quaternion = getattr(source, "attitude_quaternion", None)
            if attitude_quaternion is not None:
                area_m2 = projected_area_m2(dimensions_m, attitude_quaternion, rel_vel_hat)
            else:
                diag = math.sqrt(l**2 + w**2 + h**2)
                return (diag / 2.0) / 1000.0
        else:
            diag = math.sqrt(l**2 + w**2 + h**2)
            return (diag / 2.0) / 1000.0

        return math.sqrt(area_m2 / math.pi) / 1000.0

    # ------------------------------------------------------------------
    # Path 2: RCS from OMM metadata (or TLE rcs_m2 if populated)
    # ------------------------------------------------------------------
    rcs_m2 = obj.rcs_m2 or getattr(source, "rcs_m2", None)
    if rcs_m2 and rcs_m2 > 0:
        # Equivalent sphere radius: A = π r²  →  r = sqrt(A / π)
        return math.sqrt(rcs_m2 / math.pi) / 1000.0

    # ------------------------------------------------------------------
    # Path 3: explicit radius on DebrisObject
    # ------------------------------------------------------------------
    if obj.radius_m:
        return obj.radius_m / 1000.0

    # ------------------------------------------------------------------
    # Path 4: hard fallback — 5 m sphere
    # ------------------------------------------------------------------
    return 0.005


def load_spacebook_covariance(norad_id: int) -> np.ndarray | None:
    """Fetch and parse Spacebook Synthetic Covariance for a given satellite.

    Extracts the first 6x6 positional/velocity covariance matrix from the
    satellite's SynCoPate STK ephemeris.

    DATA-03 Fix: Validates that the covariance units header specifies km/km/s.
    STK files can declare either m/m/s or km/km/s units; accepting the wrong
    unit silently would produce Pc values off by a factor of ~10^6.
    Undeclared units default to km per STK spec §5.3.6.

    Args:
        norad_id: NORAD Catalog ID.

    Returns:
        (6, 6) covariance matrix in TEME Of Date (km, km/s), or None if
        download fails, parsing fails, unit mismatch is detected, or Spacebook
        is disabled.
    """
    if not SPACEBOOK_ENABLED:
        return None

    try:
        stk_text = fetch_synthetic_covariance_stk(norad_id)
    except Exception as exc:
        logger.warning(f"Failed to fetch Spacebook covariance for NORAD {norad_id}: {exc}")
        return None

    # DATA-03: Track declared units (km vs m) before accepting any numeric values.
    _unit_str: str | None = None
    in_cov_block = False

    for line in stk_text.splitlines():
        line = line.strip()
        if not line:
            continue
        lower = line.lower()

        if lower.startswith("covariancetimeposvel"):
            in_cov_block = True
            # Check for inline units on the header keyword line:
            # e.g.  "CovarianceTimePosVel  Units km"
            tokens = line.split()
            for i, tok in enumerate(tokens):
                if tok.lower() == "units" and i + 1 < len(tokens):
                    _unit_str = tokens[i + 1].lower()
            continue

        if in_cov_block:
            # Separate "Units km" line inside the block
            if lower.startswith("units"):
                tokens = line.split()
                if len(tokens) >= 2:
                    _unit_str = tokens[1].lower()
                continue

            fields = line.split()
            if len(fields) == 22:
                # DATA-03: Validate units before accepting the matrix.
                # Per STK spec §5.3.6, omitted units imply km — emit a debug note.
                if _unit_str is None:
                    logger.debug(
                        "Spacebook covariance for NORAD %d: no explicit units declaration; "
                        "assuming km/km/s per STK default.",
                        norad_id,
                    )
                elif _unit_str in ("m", "meters", "metre", "metres"):
                    logger.error(
                        "Spacebook covariance for NORAD %d declares unit='%s'. "
                        "ASTRA requires km/km/s units. Rejecting to prevent "
                        "silent x10^6 Pc errors (DATA-03).",
                        norad_id, _unit_str,
                    )
                    return None
                elif _unit_str not in ("km", "kilometers", "kilometre", "kilometres"):
                    logger.warning(
                        "Spacebook covariance for NORAD %d has unrecognised unit='%s'. "
                        "Proceeding, but Pc values may be incorrect if units are not km/km/s.",
                        norad_id, _unit_str,
                    )

                # First valid row — Time + 21 lower-triangular elements
                cov = np.zeros((6, 6))
                idx = 1
                for i in range(6):
                    for j in range(i + 1):
                        val = float(fields[idx])
                        cov[i, j] = val
                        cov[j, i] = val
                        idx += 1
                return cov
            elif len(fields) > 0 and not fields[0].replace(".", "").replace("-", "").isdigit():
                # Reached the end of the data block
                break

    logger.warning(f"No valid CovarianceTimePosVel block found for NORAD {norad_id}")
    return None


def closest_approach(
    trajectory_a: np.ndarray, trajectory_b: np.ndarray, times_jd: np.ndarray,
    spline_A: Optional[Any] = None, spline_B: Optional[Any] = None
) -> tuple[float, float, int]:
    """Find the exact minimum separation using cubic splines.
    
    AUDIT-D-02 Fix: Accepts pre-computed splines to allow amortised O(1) 
    evaluation time in tight loops where re-building O(N log N) is prohibitive.
    """
    coarse_dists = distance_3d(trajectory_a, trajectory_b)
    t_idx = int(np.argmin(coarse_dists))
    
    if len(times_jd) < 3:
        return float(coarse_dists[t_idx]), float(times_jd[t_idx]), t_idx
        
    if spline_A is None:
        spline_A = scipy.interpolate.CubicSpline(times_jd, trajectory_a, bc_type='natural')
    if spline_B is None:
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
    cov_map: Optional[dict[str, np.ndarray]] = None,
    vel_map: Optional[dict[str, np.ndarray]] = None,
) -> list[ConjunctionEvent]:
    """Find highly precise conjunction events using cubic spline interpolation.

    Data formats: ✓ SatelliteTLE  ✓ SatelliteOMM

    Args:
        trajectories: TrajectoryMap of NORAD-ID → (T, 3) position array (km, TEME).
        times_jd: 1-D array of T Julian Dates matching the trajectory rows.
        elements_map: NORAD-ID → DebrisObject (used for epoch, radius, covariance).
        threshold_km: Fine-filter miss-distance threshold (km).
        coarse_threshold_km: KD-tree pre-filter radius (km).
        cov_map: Optional NORAD-ID → (3, 3) or (6, 6) ECI covariance in km².
        vel_map: Optional NORAD-ID → (T, 3) SGP4 velocity array (km/s, TEME).
            When supplied the SGP4 velocities are interpolated at TCA, which is
            significantly more accurate than the position-spline derivative for
            eccentric orbits near perigee.

    Returns:
        List of ConjunctionEvent objects, sorted by miss_distance_km.

    Note:
        TCA refinement uses a fixed-density scan bracket plus cubic splines.
        A Brent-style minimization on the spline could reduce samples for
        marginal geometries; the present scheme prioritizes robustness.
    """
    norad_ids = list(trajectories.keys())
    if not norad_ids:
        return []

    T_len = len(times_jd)
    if T_len < 3:
        raise AstraError("At least 3 timesteps required for CubicSpline interpolation.")

    logger.info(f"Initiating Conjunction Analysis for {len(norad_ids)} objects over {T_len} time steps.")

    # ---------------------------------------------------------
    # Drop NaN trajectories before spatial screening.
    # ---------------------------------------------------------
    valid_trajectories: TrajectoryMap = {}
    nan_ids: list[str] = []
    for nid, traj in trajectories.items():
        if np.any(~np.isfinite(traj)):
            nan_ids.append(nid)
        else:
            valid_trajectories[nid] = traj

    if nan_ids:
        from astra import config
        if config.ASTRA_STRICT_MODE:
            raise PropagationError(
                f"[ASTRA STRICT] {len(nan_ids)} satellites have invalid trajectories: "
                f"{nan_ids[:5]}{'...' if len(nan_ids) > 5 else ''}. "
                "Conjunction analysis cannot proceed.",
                norad_id=str(nan_ids[0])
            )
        logger.warning(f"{len(nan_ids)} satellites excluded (NaN trajectories): {nan_ids[:10]}")

    trajectories = valid_trajectories
    norad_ids = list(trajectories.keys())
    if not norad_ids:
        logger.info("All satellites had NaN trajectories. No conjunctions to evaluate.")
        return []

    # ---------------------------------------------------------
    # Phase 1 & 2: Unified Trajectory-AABB SpatialIndex screening
    # ---------------------------------------------------------
    logger.debug(f"Running Phase 1 & 2: Unified Trajectory-AABB SpatialIndex screening over {T_len} timesteps...")

    idx = SpatialIndex()
    # Build a single tree for the entire propagation window (SE-C optimization)
    idx.rebuild_for_trajectories(trajectories)
    
    # Query all candidate pairs once. The index accounts for trajectory excursions.
    candidate_pairs_phase2 = set(idx.query_pairs(threshold_km=coarse_threshold_km))

    if not candidate_pairs_phase2:
        logger.info("Macro AABB Sweep complete: 0 candidate pairs found.")
        return []

    logger.info(f"Macro AABB Filter Complete: Analyzing precise geometry for {len(candidate_pairs_phase2)} pairs.")

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

        # Velocity splines from SGP4 vel_map when available (more accurate than
        # differentiating position splines), especially near perigee on eccentric orbits.
        vel_spline_A = (
            scipy.interpolate.CubicSpline(times_jd, vel_map[A], bc_type='natural')
            if (vel_map and A in vel_map) else None
        )
        vel_spline_B = (
            scipy.interpolate.CubicSpline(times_jd, vel_map[B], bc_type='natural')
            if (vel_map and B in vel_map) else None
        )
        
        # 3. Dense 1-second resolution evaluation across the local min bracket
        is_edge = (t_idx == 0 or t_idx == T_len - 1)
        bracket_width = 2 if is_edge else 1
        idx_low = max(0, t_idx - bracket_width)
        idx_high = min(T_len - 1, t_idx + bracket_width)
        
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
        
        # Prefer SGP4 velocity spline at TCA; else spline derivative (km/JD → km/s).
        if vel_spline_A is not None:
            vel_A = vel_spline_A(tca_jd)
        else:
            vel_A = spline_A(tca_jd, nu=1) / 86400.0  # km/JD → km/s

        if vel_spline_B is not None:
            vel_B = vel_spline_B(tca_jd)
        else:
            vel_B = spline_B(tca_jd, nu=1) / 86400.0  # km/JD → km/s
        
        rel_vel_vec = vel_A - vel_B
        rel_vel = float(np.linalg.norm(rel_vel_vec))
        
        obj_A = elements_map[A]
        obj_B = elements_map[B]
        
        days_since_epoch_A = tca_jd - obj_A.source.epoch_jd
        days_since_epoch_B = tca_jd - obj_B.source.epoch_jd
        
        _spacebook_cov_a = False
        if cov_map and A in cov_map:
            cov_A = cov_map[A]
        else:
            try:
                cov_A = load_spacebook_covariance(int(A))
                if cov_A is not None:
                    _spacebook_cov_a = True
            except Exception:
                cov_A = None
            if cov_A is None:
                try:
                    cov_rtn_A = estimate_covariance(days_since_epoch_A)
                    cov_A = rotate_covariance_rtn_to_eci(cov_rtn_A, pos_A, vel_A)
                except Exception:
                    cov_A = None

        _spacebook_cov_b = False
        if cov_map and B in cov_map:
            cov_B = cov_map[B]
        else:
            try:
                cov_B = load_spacebook_covariance(int(B))
                if cov_B is not None:
                    _spacebook_cov_b = True
            except Exception:
                cov_B = None
            if cov_B is None:
                try:
                    cov_rtn_B = estimate_covariance(days_since_epoch_B)
                    cov_B = rotate_covariance_rtn_to_eci(cov_rtn_B, pos_B, vel_B)
                except Exception:
                    cov_B = None
        
        miss_vector = pos_A - pos_B
        
        rel_vel_hat = rel_vel_vec / max(rel_vel, 1e-12)
        
        # Attitude-aware dynamic collision radius
        rad_A_km = _dynamic_radius_km(obj_A, rel_vel_hat, pos_A, vel_A)
        rad_B_km = _dynamic_radius_km(obj_B, rel_vel_hat, pos_B, vel_B)
        
        if cov_A is not None and cov_B is not None:
            if cov_A.shape != cov_B.shape:
                if cov_A.shape == (6, 6) and cov_B.shape == (3, 3):
                    cov_A = cov_A[:3, :3]
                elif cov_B.shape == (6, 6) and cov_A.shape == (3, 3):
                    cov_B = cov_B[:3, :3]
                    
            P_c = compute_collision_probability(
                miss_vector, rel_vel_vec, cov_A, cov_B, radius_a_km=rad_A_km, radius_b_km=rad_B_km
            )
        else:
            P_c = None
            logger.warning(
                f"Pair ({A},{B}): No covariance available — Pc set to None. "
                "Supply cov_map with CDM covariances or use Relaxed mode."
            )
        
        risk = _classify_risk(P_c) if P_c is not None else "UNKNOWN"
        
        # SE-DEF-001 Fix: Removed dead _cov_src() function (was defined here but never called).
        # Determine covariance source per object using sentinel flags set during loading:
        if cov_map and A in cov_map:
            src_A = "CDM"
        elif _spacebook_cov_a:
            src_A = "COMSPOC_SYNTHETIC"
        elif cov_A is not None:
            src_A = "SYNTHETIC"
        else:
            src_A = "UNAVAILABLE"

        if cov_map and B in cov_map:
            src_B = "CDM"
        elif _spacebook_cov_b:
            src_B = "COMSPOC_SYNTHETIC"
        elif cov_B is not None:
            src_B = "SYNTHETIC"
        else:
            src_B = "UNAVAILABLE"


        if src_A == src_B:
            covariance_src = src_A
        elif "UNAVAILABLE" in (src_A, src_B):
            covariance_src = "UNAVAILABLE"
        else:
            covariance_src = f"MIXED({src_A}+{src_B})"
            
        return ConjunctionEvent(
            object_a_id=A,
            object_b_id=B,
            tca_jd=tca_jd,
            miss_distance_km=min_dist,
            relative_velocity_km_s=rel_vel,
            collision_probability=P_c,
            risk_level=risk,
            position_a_km=pos_A,
            position_b_km=pos_B,
            covariance_source=covariance_src
        )

    # AUDIT-B-07 Fix: Use ThreadPoolExecutor as a context manager so that
    # executor.shutdown() is guaranteed on all code paths, including exceptions
    # raised before the old try/finally block was entered (e.g. during future
    # submission for very large catalogs).
    skipped = 0
    strict_error = None
    with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = {executor.submit(evaluate_pair, pair): pair for pair in candidate_pairs_phase2}
        for future in concurrent.futures.as_completed(futures):
            pair = futures[future]
            try:
                result = future.result()
                if result is not None:
                    events.append(result)
            except Exception as e:
                skipped += 1
                from astra import config
                if config.ASTRA_STRICT_MODE:
                    strict_error = AstraError(
                        f"[ASTRA STRICT] Conjunction pair {pair} evaluation failed: {e!r}. "
                        "In strict mode, all pairs must evaluate cleanly."
                    )
                    for f in futures:
                        f.cancel()
                    break
                logger.warning(f"Conjunction pair {pair} skipped: {e!r}")
    # __exit__ of the context manager calls shutdown(wait=True) automatically

    if strict_error:
        raise strict_error from None

    if skipped > 0:
        logger.warning(f"{skipped} conjunction pairs skipped due to errors. Results may be incomplete.")

    events.sort(key=lambda x: x.miss_distance_km)
    logger.info(f"Conjunction Sweep Complete: {len(events)} events detected.")
    return events
