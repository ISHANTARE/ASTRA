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
import math
from typing import Optional, Any
import numpy as np
import scipy.interpolate
from astra.errors import AstraError, PropagationError
from astra.models import (
    ConjunctionEvent,
    DebrisObject,
    TrajectoryMap,
    projected_area_m2,
)
from astra.covariance import (
    compute_collision_probability,
    estimate_covariance,
    rotate_covariance_rtn_to_eci,
)
from astra.spacebook import fetch_synthetic_covariance_stk
from astra.log import get_logger
from astra.spatial_index import SpatialIndex
logger = get_logger(__name__)
def distance_3d(pos_a: np.ndarray, pos_b: np.ndarray) -> np.ndarray:
    """Compute Euclidean distance between two position arrays."""
    return np.linalg.norm(pos_a - pos_b, axis=-1)  # type: ignore[no-any-return]
def _classify_risk(P_c: Optional[float]) -> str:
    """Classify risk level based on collision probability."""
    if P_c is None:
        return "UNKNOWN"  # type: ignore[no-any-return]
    if P_c > 1e-4:
        return "CRITICAL"  # type: ignore[no-any-return]
    if P_c > 1e-5:
        return "HIGH"  # type: ignore[no-any-return]
    if P_c > 1e-6:
        return "MEDIUM"  # type: ignore[no-any-return]
    return "LOW"  # type: ignore[no-any-return]
def _dynamic_radius_km(
    obj: DebrisObject, rel_vel_hat: np.ndarray, pos_eci: np.ndarray, vel_eci: np.ndarray
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
        length, w, h = dimensions_m
        attitude_mode = getattr(source, "attitude_mode", "TUMBLING")
        if attitude_mode == "TUMBLING":
            area_m2 = 2.0 * (length * w + w * h + h * length) / 4.0
        elif attitude_mode == "NADIR":
            pos_hat = pos_eci / max(np.linalg.norm(pos_eci), 1e-12)
            vel_hat = vel_eci / max(np.linalg.norm(vel_eci), 1e-12)
            normal_hat = np.cross(pos_hat, vel_hat)
            normal_hat /= max(np.linalg.norm(normal_hat), 1e-12)
            x_hat = np.cross(normal_hat, pos_hat)
            faces = [
                (x_hat, w * h),
                (normal_hat, length * h),
                (pos_hat, length * w),
            ]
            area_m2 = sum(abs(float(np.dot(n, rel_vel_hat))) * a for n, a in faces)
        elif attitude_mode == "INERTIAL":
            attitude_quaternion = getattr(source, "attitude_quaternion", None)
            if attitude_quaternion is not None:
                area_m2 = projected_area_m2(
                    dimensions_m, attitude_quaternion, rel_vel_hat
                )
            else:
                diag = math.sqrt(length**2 + w**2 + h**2)
                return (diag / 2.0) / 1000.0  # type: ignore[no-any-return]
        else:
            diag = math.sqrt(length**2 + w**2 + h**2)
            return (diag / 2.0) / 1000.0  # type: ignore[no-any-return]
        return math.sqrt(area_m2 / math.pi) / 1000.0  # type: ignore[no-any-return]
    # ------------------------------------------------------------------
    # Path 2: RCS from OMM metadata (or TLE rcs_m2 if populated)
    # ------------------------------------------------------------------
    rcs_m2 = obj.rcs_m2 or getattr(source, "rcs_m2", None)
    if rcs_m2 and rcs_m2 > 0:
        # Equivalent sphere radius: A = π r²  →  r = sqrt(A / π)
        return math.sqrt(rcs_m2 / math.pi) / 1000.0  # type: ignore[no-any-return]
    # ------------------------------------------------------------------
    # Path 3: explicit radius on DebrisObject
    # ------------------------------------------------------------------
    if obj.radius_m:
        return obj.radius_m / 1000.0  # type: ignore[no-any-return]
    # ------------------------------------------------------------------
    # Path 4: hard fallback — 5 m sphere
    # ------------------------------------------------------------------
    return 0.005  # type: ignore[no-any-return]
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
    from astra.config import SPACEBOOK_ENABLED
    if not SPACEBOOK_ENABLED:
        return None  # type: ignore[no-any-return]
    try:
        stk_text = fetch_synthetic_covariance_stk(norad_id)
    except AstraError as exc:
        logger.warning(
            f"Failed to fetch Spacebook covariance for NORAD {norad_id}: {exc}"
        )
        return None  # type: ignore[no-any-return]
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
                        norad_id,
                        _unit_str,
                    )
                    return None  # type: ignore[no-any-return]
                elif _unit_str not in ("km", "kilometers", "kilometre", "kilometres"):
                    logger.warning(
                        "Spacebook covariance for NORAD %d has unrecognised unit='%s'. "
                        "Proceeding, but Pc values may be incorrect if units are not km/km/s.",
                        norad_id,
                        _unit_str,
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
                return cov  # type: ignore[no-any-return]
            elif (
                len(fields) > 0
                and not fields[0].replace(".", "").replace("-", "").isdigit()
            ):
                # Reached the end of the data block
                break
    logger.warning(f"No valid CovarianceTimePosVel block found for NORAD {norad_id}")
    return None  # type: ignore[no-any-return]
def closest_approach(
    trajectory_a: np.ndarray,
    trajectory_b: np.ndarray,
    times_jd: np.ndarray,
    spline_A: Optional[Any] = None,
    spline_B: Optional[Any] = None,
) -> tuple[float, float, int]:
    """Find the exact minimum separation using cubic splines.
    Accepts pre-computed splines to allow amortised O(1)
    evaluation time in tight loops where re-building O(N log N) is prohibitive.
    """
    times_jd = np.asarray(times_jd, dtype=float)
    if times_jd.ndim != 1:
        raise AstraError("times_jd must be a one-dimensional array.")
    if len(times_jd) != len(trajectory_a) or len(times_jd) != len(trajectory_b):
        raise AstraError(
            "trajectory_a, trajectory_b, and times_jd must have matching lengths."
        )
    if np.any(np.diff(times_jd) <= 0.0):
        raise AstraError("times_jd must be strictly increasing for spline TCA refinement.")
    coarse_dists = distance_3d(trajectory_a, trajectory_b)
    t_idx = int(np.argmin(coarse_dists))
    if len(times_jd) < 3:
        return float(coarse_dists[t_idx]), float(times_jd[t_idx]), t_idx  # type: ignore[no-any-return]
    use_local_seconds = spline_A is None and spline_B is None
    spline_times = (times_jd - times_jd[0]) * 86400.0 if use_local_seconds else times_jd
    if spline_A is None:
        spline_A = scipy.interpolate.CubicSpline(
            spline_times, trajectory_a, bc_type="natural"
        )
    if spline_B is None:
        spline_B = scipy.interpolate.CubicSpline(
            spline_times, trajectory_b, bc_type="natural"
        )
    idx_low = max(0, t_idx - 1)
    idx_high = min(len(times_jd) - 1, t_idx + 1)
    # Brent minimization on the CubicSpline distance.
    # This replaces the fixed 100-point dense scan with a bounded scalar
    # optimization, improving TCA precision from ~0.5 s to sub-millisecond
    # with typically 8-15 function evaluations instead of 100.
    try:
        from scipy.optimize import minimize_scalar
        def _dist_fn(t_jd: float) -> float:
            return float(np.linalg.norm(spline_A(t_jd) - spline_B(t_jd)))
        _lo = float(spline_times[idx_low])
        _hi = float(spline_times[idx_high])
        _res = minimize_scalar(
            _dist_fn,
            bounds=(_lo, _hi),
            method="bounded",
            options={"xatol": 1e-3 if use_local_seconds else 1.16e-8},
        )
        if _res.success and _lo <= _res.x <= _hi:
            if use_local_seconds:
                tca_jd_fine = float(times_jd[0] + float(_res.x) / 86400.0)
            else:
                tca_jd_fine = float(_res.x)
            min_dist_fine = float(_res.fun)
            return min_dist_fine, tca_jd_fine, t_idx  # type: ignore[no-any-return]
    except Exception as exc:
        logger.debug(f"Brent TCA refinement failed ({exc}). Falling back to coarse 100-point scan.")
        pass  # fall back to coarse scan
    # Coarse fallback: dense 100-point bracket scan
    t_dense = np.linspace(spline_times[idx_low], spline_times[idx_high], 100)
    rA_dense = spline_A(t_dense)
    rB_dense = spline_B(t_dense)
    dense_dists = distance_3d(rA_dense, rB_dense)
    t_dense_idx = int(np.argmin(dense_dists))
    if use_local_seconds:
        tca_jd_dense = float(times_jd[0] + float(t_dense[t_dense_idx]) / 86400.0)
    else:
        tca_jd_dense = float(t_dense[t_dense_idx])
    return float(dense_dists[t_dense_idx]), tca_jd_dense, t_idx  # type: ignore[no-any-return]
def find_conjunctions(
    trajectories: TrajectoryMap,
    times_jd: np.ndarray,
    elements_map: dict[str, DebrisObject],
    threshold_km: float = 5.0,
    coarse_threshold_km: float = 50.0,
    cov_map: Optional[dict[str, np.ndarray]] = None,
    vel_map: Optional[dict[str, np.ndarray]] = None,
    max_workers: Optional[int] = None,
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
        TCA refinement uses a CubicSpline + Brent scalar minimization
        (scipy.optimize.minimize_scalar, method='bounded') for sub-millisecond
        TCA precision. Falls back to 100-point dense scan if Brent fails.
        ``max_workers`` caps the thread-pool size.  Defaults to
        ``min(cpu_count, 16)`` to prevent thread storms on large catalogs
        (>10,000 NORAD IDs generate O(N²) candidate pairs).  Set explicitly
        when running in a containerised or resource-constrained environment.
    """
    norad_ids = list(trajectories.keys())
    if not norad_ids:
        return []  # type: ignore[no-any-return]
    T_len = len(times_jd)
    if T_len < 3:
        raise AstraError("At least 3 timesteps required for CubicSpline interpolation.")
    logger.info(
        f"Initiating Conjunction Analysis for {len(norad_ids)} objects over {T_len} time steps."
    )
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
                norad_id=str(nan_ids[0]),
            )
        logger.warning(
            f"{len(nan_ids)} satellites excluded (NaN trajectories): {nan_ids[:10]}"
        )
    trajectories = valid_trajectories
    norad_ids = list(trajectories.keys())
    if not norad_ids:
        logger.info("All satellites had NaN trajectories. No conjunctions to evaluate.")
        return []  # type: ignore[no-any-return]
    # ---------------------------------------------------------
    # Phase 1 & 2: Unified Trajectory-AABB SpatialIndex screening
    # ---------------------------------------------------------
    logger.debug(
        f"Running Phase 1 & 2: Unified Trajectory-AABB SpatialIndex screening over {T_len} timesteps..."
    )
    idx = SpatialIndex()
    # Build a single tree for the entire propagation window (SE-C optimization)
    idx.rebuild_for_trajectories(trajectories)
    # Query candidate pairs once. Keep as a sequence and process with bounded
    # in-flight tasks to avoid all-at-once future submission memory spikes.
    candidate_pairs_phase2 = list(idx.query_pairs(threshold_km=coarse_threshold_km))
    if not candidate_pairs_phase2:
        logger.info("Macro AABB Sweep complete: 0 candidate pairs found.")
        return []  # type: ignore[no-any-return]
    logger.info(
        f"Macro AABB Filter Complete: Analyzing precise geometry for {len(candidate_pairs_phase2)} pairs."
    )
    # ---------------------------------------------------------
    # Phase 3: Exact Curvilinear TCA Interpolation
    # ---------------------------------------------------------
    logger.debug(
        "Running Phase 3: Root-finding via Spline interpolation & Mahalanobis B-Plane Covariance Projection..."
    )
    events = []
    def evaluate_pair(pair: tuple[str, str]) -> ConjunctionEvent | None:
        """Worker function for concurrent execution."""
        A, B = pair
        traj_A = trajectories[A]
        traj_B = trajectories[B]
        # 1. Coarse search to find a spline bracket. Do not reject solely on
        # sampled-point distance; fast crossings can occur between samples.
        coarse_dists = distance_3d(traj_A, traj_B)
        t_idx = int(np.argmin(coarse_dists))
        # 2. Build Cubic Splines for exact curvilinear motion mapping. Use a
        # local seconds scale rather than absolute Julian Dates to avoid
        # conditioning loss over short windows near JD ~2.46e6.
        times_s = (times_jd - times_jd[0]) * 86400.0
        spline_A = scipy.interpolate.CubicSpline(times_s, traj_A, bc_type="natural")
        spline_B = scipy.interpolate.CubicSpline(times_s, traj_B, bc_type="natural")
        # Velocity splines from SGP4 vel_map when available (more accurate than
        # differentiating position splines), especially near perigee on eccentric orbits.
        vel_spline_A = (
            scipy.interpolate.CubicSpline(times_s, vel_map[A], bc_type="natural")
            if (vel_map and A in vel_map)
            else None
        )
        vel_spline_B = (
            scipy.interpolate.CubicSpline(times_s, vel_map[B], bc_type="natural")
            if (vel_map and B in vel_map)
            else None
        )
        # 3. Refine TCA over the full local sample bracket before thresholding.
        # A dense 1-second pre-scan can miss sub-second brackets, so use Brent
        # directly on the spline distance and reserve the dense scan as fallback.
        is_edge = t_idx == 0 or t_idx == T_len - 1
        bracket_width = 2 if is_edge else 1
        idx_low = max(0, t_idx - bracket_width)
        idx_high = min(T_len - 1, t_idx + bracket_width)
        _lo_t = float(times_s[idx_low])
        _hi_t = float(times_s[idx_high])
        min_dist = float(coarse_dists[t_idx])
        tca_s = float(times_s[t_idx])
        tca_jd = float(times_jd[0] + tca_s / 86400.0)
        try:
            from scipy.optimize import minimize_scalar
            _spline_dist = lambda t: float(np.linalg.norm(spline_A(t) - spline_B(t)))  # noqa: E731
            _brent = minimize_scalar(
                _spline_dist,
                bounds=(_lo_t, _hi_t),
                method="bounded",
                options={"xatol": 1e-3},
            )
            if _brent.success and _lo_t <= _brent.x <= _hi_t:
                tca_s = float(_brent.x)
                tca_jd = float(times_jd[0] + tca_s / 86400.0)
                min_dist = float(_brent.fun)
                pos_A = spline_A(tca_s)
                pos_B = spline_B(tca_s)
        except Exception as exc:
            logger.debug(f"Brent TCA refinement failed ({exc}). Falling back to dense scan.")
            samples_in_bracket = max(100, int(math.ceil(_hi_t - _lo_t)) + 1)
            t_dense = np.linspace(_lo_t, _hi_t, samples_in_bracket)
            rA_dense = spline_A(t_dense)
            rB_dense = spline_B(t_dense)
            dense_dists = distance_3d(rA_dense, rB_dense)
            tca_dense_idx = int(np.argmin(dense_dists))
            min_dist = float(dense_dists[tca_dense_idx])
            tca_s = float(t_dense[tca_dense_idx])
            tca_jd = float(times_jd[0] + tca_s / 86400.0)
        if min_dist > threshold_km:
            return None  # type: ignore[no-any-return]
        # Unconditionally compute pos_A and pos_B from the final tca_jd.
        # was retained, but pos_A and pos_B were never assigned, causing UnboundLocalError
        # or inconsistent vector states further down.
        pos_A = spline_A(tca_s)
        pos_B = spline_B(tca_s)
        # Prefer SGP4 velocity spline at TCA; else spline derivative (km/JD → km/s).
        if vel_spline_A is not None:
            vel_A = vel_spline_A(tca_s)
        else:
            vel_A = spline_A(tca_s, nu=1)
        if vel_spline_B is not None:
            vel_B = vel_spline_B(tca_s)
        else:
            vel_B = spline_B(tca_s, nu=1)
        rel_vel_vec = vel_A - vel_B
        rel_vel = float(np.linalg.norm(rel_vel_vec))
        obj_A = elements_map[A]
        obj_B = elements_map[B]
        days_since_epoch_A = tca_jd - obj_A.source.epoch_jd
        days_since_epoch_B = tca_jd - obj_B.source.epoch_jd
        _spacebook_cov_a = False
        cov_A: Optional[np.ndarray] = None
        if cov_map and A in cov_map:
            cov_A = cov_map[A]
        else:
            try:
                cov_A = load_spacebook_covariance(int(A))
                if cov_A is not None:
                    _spacebook_cov_a = True
            except (ValueError, TypeError, AstraError):
                cov_A = None
            if cov_A is None:
                try:
                    cov_rtn_A = estimate_covariance(days_since_epoch_A)
                    cov_A = rotate_covariance_rtn_to_eci(cov_rtn_A, pos_A, vel_A)
                except (ValueError, ArithmeticError, AstraError):
                    cov_A = None
        _spacebook_cov_b = False
        cov_B: Optional[np.ndarray] = None
        if cov_map and B in cov_map:
            cov_B = cov_map[B]
        else:
            try:
                cov_B = load_spacebook_covariance(int(B))
                if cov_B is not None:
                    _spacebook_cov_b = True
            except (ValueError, TypeError, AstraError):
                cov_B = None
            if cov_B is None:
                try:
                    cov_rtn_B = estimate_covariance(days_since_epoch_B)
                    cov_B = rotate_covariance_rtn_to_eci(cov_rtn_B, pos_B, vel_B)
                except (ValueError, ArithmeticError, AstraError):
                    cov_B = None
        miss_vector = pos_A - pos_B
        rel_vel_hat = rel_vel_vec / max(rel_vel, 1e-12)
        # Attitude-aware dynamic collision radius
        rad_A_km = _dynamic_radius_km(obj_A, rel_vel_hat, pos_A, vel_A)
        rad_B_km = _dynamic_radius_km(obj_B, rel_vel_hat, pos_B, vel_B)
        if cov_A is not None and cov_B is not None:
            # Reject mixed-dimension covariance (e.g. 6x6 vs 3x3) instead of
            # silently truncating the 6x6 down to 3x3. The velocity uncertainty bounds
            # are fundamentally required for the dynamic-radius expansion to be valid;
            # mixing a velocity-aware covariance with a positional-only one breaks
            # the statistical symmetry of the Mahalanobis projection.
            if cov_A.shape != cov_B.shape:
                logger.warning(
                    f"Pair ({A},{B}): Covariance dimension mismatch "
                    f"({cov_A.shape} vs {cov_B.shape}). Pc set to None."
                )
                P_c = None
            else:
                P_c = compute_collision_probability(
                    miss_vector,
                    rel_vel_vec,
                    cov_A,
                    cov_B,
                    radius_a_km=rad_A_km,
                    radius_b_km=rad_B_km,
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
        return ConjunctionEvent(  # type: ignore[no-any-return]
            object_a_id=A,
            object_b_id=B,
            tca_jd=tca_jd,
            miss_distance_km=min_dist,
            relative_velocity_km_s=rel_vel,
            collision_probability=P_c,
            risk_level=risk,
            position_a_km=pos_A,
            position_b_km=pos_B,
            covariance_source=covariance_src,
        )
    # Throttle max_workers to prevent thread storms on large catalogs.
    # Avoid defaulting to host CPU count on HPC nodes, where very high core
    # counts can degrade performance through thread scheduling overhead.
    # Resolution priority (highest → lowest):
    #   1. Explicit ``max_workers`` argument passed by the caller.
    #   2. ``ASTRA_MAX_WORKERS`` environment variable (for HPC / container deployments
    #      where the operator needs fleet-wide scaling control without code changes).
    #   3. ``16`` safe conservative default.
    from astra.config import get_max_workers
    _default_workers = get_max_workers(16)
    _workers = max_workers if max_workers is not None else _default_workers
    _max_inflight = max(8, _workers * 4)
    # Initialise counters BEFORE the thread loop so they are always bound.
    # causing NameError (skipped) / UnboundLocalError (strict_error) when no pairs failed.
    skipped = 0
    strict_error: AstraError | None = None
    with concurrent.futures.ThreadPoolExecutor(max_workers=_workers) as executor:
        futures: dict[
            concurrent.futures.Future[ConjunctionEvent | None], tuple[str, str]
        ] = {}
        def _consume_done(
            done_futures: set[concurrent.futures.Future[ConjunctionEvent | None]],
        ) -> bool:
            """Consume completed futures. Returns True when strict error occurs."""
            nonlocal skipped, strict_error
            for future in done_futures:
                pair = futures.pop(future)
                try:
                    result = future.result()
                    if result is not None:
                        events.append(result)
                except (ValueError, TypeError, KeyError, IndexError, RuntimeError, AstraError, ArithmeticError) as e:
                    skipped += 1
                    from astra import config
                    if config.ASTRA_STRICT_MODE:
                        strict_error = AstraError(
                            f"[ASTRA STRICT] Conjunction pair {pair} evaluation failed: {e!r}. "
                            "In strict mode, all pairs must evaluate cleanly."
                        )
                        for f in futures:
                            f.cancel()
                        return True  # type: ignore[no-any-return]
                    logger.warning(f"Conjunction pair {pair} skipped: {e!r}")
            return False  # type: ignore[no-any-return]
        for pair in candidate_pairs_phase2:
            future = executor.submit(evaluate_pair, pair)
            futures[future] = pair
            if len(futures) >= _max_inflight:
                done, _ = concurrent.futures.wait(
                    set(futures),
                    return_when=concurrent.futures.FIRST_COMPLETED,
                )
                if _consume_done(done):
                    break
        if strict_error is None and futures:
            for future in concurrent.futures.as_completed(futures):
                if _consume_done({future}):
                    break
    # __exit__ of the context manager calls shutdown(wait=True) automatically
    if strict_error is not None:
        raise strict_error from None
    if skipped > 0:
        logger.warning(
            f"{skipped} conjunction pairs skipped due to errors. Results may be incomplete."
        )
    events.sort(key=lambda x: x.miss_distance_km)
    logger.info(f"Conjunction Sweep Complete: {len(events)} events detected.")
    return events  # type: ignore[no-any-return]
def run_conjunction_sweep(
    catalog: "list[Any]",
    t_start_jd: float,
    t_end_jd: float,
    step_minutes: float = 5.0,
    threshold_km: float = 5.0,
    coarse_threshold_km: float = 50.0,
    cov_map: "Optional[dict[str, np.ndarray]]" = None,
    max_workers: "Optional[int]" = None,
) -> "list[ConjunctionEvent]":
    """High-level conjunction sweep pipeline: catalog → ConjunctionEvent list.
    **Replaces the 5-step manual orchestration** (fetch → map → propagate →
    index → screen) with a single call.  Internally handles:
    1. Converting ``catalog`` entries (SatelliteTLE or SatelliteOMM) to
       :class:`DebrisObject` instances via :func:`make_debris_object`.
    2. Building the Julian-Date time grid between ``t_start_jd`` and
       ``t_end_jd`` at ``step_minutes`` resolution.
    3. Batch-propagating all satellites over the grid via :func:`propagate_many`.
    4. Dropping any satellite whose trajectory contains NaN values (SGP4
       error; typically very decayed TLEs).
    5. Running :func:`find_conjunctions` with the 3-phase KD-tree + spline +
       Pc pipeline on the resulting trajectory map.
    Args:
        catalog: List of ``SatelliteTLE`` or ``SatelliteOMM`` objects.
            Obtain via ``fetch_celestrak_group``, ``fetch_spacetrack_group``,
            or any other ingestion function.
        t_start_jd: Window start as Julian Date (UTC).
        t_end_jd: Window end as Julian Date (UTC). Must be > t_start_jd.
        step_minutes: Propagation step size in minutes (default 5.0).
            Smaller values improve TCA precision at the cost of memory.
        threshold_km: Fine-filter conjunction threshold in km (default 5.0).
        coarse_threshold_km: KD-tree pre-filter radius in km (default 50.0).
        cov_map: Optional NORAD-ID → (3,3) or (6,6) covariance in km².
            When omitted, synthetic covariance estimation is used.
        max_workers: Maximum thread-pool size for concurrent pair evaluation.
            Defaults to ``min(cpu_count, 16)``.
    Returns:
        List of :class:`ConjunctionEvent` objects sorted by miss distance,
        ready for downstream Pc ranking or CDM generation.
    Raises:
        AstraError: If ``t_end_jd <= t_start_jd`` or ``catalog`` is empty.
        PropagationError: Propagation failure for any object in STRICT_MODE.
    Example::
        import astra
        import math
        catalog = astra.fetch_celestrak_group("starlink")
        t0 = astra.datetime_utc_to_jd(datetime(2026, 5, 1, tzinfo=timezone.utc))
        t1 = t0 + 1.0  # 24-hour window
        events = astra.run_conjunction_sweep(catalog, t0, t1, threshold_km=5.0)
        for ev in events[:10]:
            print(f"{ev.object_a_id} vs {ev.object_b_id}  "
                  f"miss={ev.miss_distance_km:.3f} km  Pc={ev.collision_probability:.2e}")
    """
    if not catalog:
        raise AstraError("run_conjunction_sweep: catalog is empty.")
    if t_end_jd <= t_start_jd:
        raise AstraError(
            f"run_conjunction_sweep: t_end_jd ({t_end_jd}) must be "
            f"strictly greater than t_start_jd ({t_start_jd})."
        )
    if step_minutes <= 0:
        raise AstraError(
            f"run_conjunction_sweep: step_minutes must be positive, got {step_minutes}."
        )
    # ── 1. Convert catalog entries to DebrisObjects ─────────────────────────
    from astra.debris import make_debris_object
    elements_map: dict[str, DebrisObject] = {}
    skipped_build = 0
    for sat in catalog:
        try:
            obj = make_debris_object(sat)
            elements_map[sat.norad_id] = obj
        except Exception as _be:
            skipped_build += 1
            logger.debug(
                "run_conjunction_sweep: skipping NORAD %s during DebrisObject "
                "construction (%r).", getattr(sat, "norad_id", "?"), _be
            )
    if not elements_map:
        raise AstraError(
            "run_conjunction_sweep: no valid DebrisObjects could be constructed "
            "from the supplied catalog."
        )
    if skipped_build:
        logger.warning(
            "run_conjunction_sweep: %d catalog entries skipped during DebrisObject "
            "construction (malformed data or missing fields).", skipped_build
        )
    satellites = [obj.source for obj in elements_map.values()]
    # ── 2. Build Julian-Date time grid ───────────────────────────────────────
    total_minutes = (t_end_jd - t_start_jd) * 1440.0
    n_steps = max(3, int(total_minutes / step_minutes) + 1)
    times_jd = np.linspace(t_start_jd, t_end_jd, n_steps)
    logger.info(
        "run_conjunction_sweep: %d objects, %.2f-hr window, %d time steps "
        "(step=%.1f min), threshold=%.1f km.",
        len(satellites),
        (t_end_jd - t_start_jd) * 24.0,
        n_steps,
        step_minutes,
        threshold_km,
    )
    # ── 3. Batch propagate all satellites ────────────────────────────────────
    from astra.orbit import propagate_many
    trajectories, vel_map = propagate_many(satellites, times_jd)
    # ── 4. Drop NaN trajectories (SGP4 failures) ─────────────────────────────
    valid_traj: "dict[str, np.ndarray]" = {}
    valid_vel: "dict[str, np.ndarray]" = {}
    nan_count = 0
    for nid, traj in trajectories.items():
        if np.any(~np.isfinite(traj)):
            nan_count += 1
        else:
            valid_traj[nid] = traj
            if nid in vel_map:
                valid_vel[nid] = vel_map[nid]
    if nan_count:
        logger.warning(
            "run_conjunction_sweep: %d satellites excluded due to SGP4 NaN "
            "trajectories (likely decayed or invalid TLEs).", nan_count
        )
    if len(valid_traj) < 2:
        logger.info(
            "run_conjunction_sweep: fewer than 2 valid trajectories remain after "
            "NaN filtering. Returning empty event list."
        )
        return []
    # Restrict elements_map to valid trajectories only
    valid_elements: dict[str, DebrisObject] = {
        nid: obj for nid, obj in elements_map.items() if nid in valid_traj
    }
    # ── 5. Run the conjunction screening pipeline ────────────────────────────
    return find_conjunctions(
        trajectories=valid_traj,
        times_jd=times_jd,
        elements_map=valid_elements,
        threshold_km=threshold_km,
        coarse_threshold_km=coarse_threshold_km,
        cov_map=cov_map,
        vel_map=valid_vel,
        max_workers=max_workers,
    )


# ---------------------------------------------------------------------------
# ConjunctionWindow (AS-03)
# ---------------------------------------------------------------------------
from dataclasses import dataclass as _cj_dataclass


@_cj_dataclass(frozen=True)
class ConjunctionWindow:
    """Time interval during which two objects are within a distance threshold.

    Unlike :class:`ConjunctionEvent` (a point-event at TCA), a
    ``ConjunctionWindow`` defines the **entry** and **exit** epochs of the
    close-approach interval. This is required for:

    - Communication blackout window analysis.
    - Maneuver planning (avoidance window computation).
    - CDM screening compliance (7-day look-ahead windows).

    Attributes:
        object_a_id: NORAD ID of object A.
        object_b_id: NORAD ID of object B.
        entry_jd: Julian Date when distance first drops below threshold.
        exit_jd: Julian Date when distance first rises above threshold.
        tca_jd: Julian Date of closest approach within the window.
        min_distance_km: Minimum separation within the window (km).
        duration_s: Window duration in seconds.
    """
    object_a_id: str
    object_b_id: str
    entry_jd: float
    exit_jd: float
    tca_jd: float
    min_distance_km: float

    @property
    def duration_s(self) -> float:
        """Duration of the conjunction window in seconds."""
        return (self.exit_jd - self.entry_jd) * 86400.0


def find_conjunction_windows(
    trajectories: TrajectoryMap,
    times_jd: np.ndarray,
    threshold_km: float = 10.0,
    objects: Optional[dict[str, Any]] = None,
) -> list[ConjunctionWindow]:
    """Find time windows during which pairs of objects are within a distance threshold.

    Unlike :func:`find_conjunctions` which returns point-events at TCA, this
    function returns **intervals** [entry_jd, exit_jd] during which two objects
    remain within ``threshold_km``. Each window also includes the TCA and
    minimum distance within that interval.

    Uses cubic spline interpolation for sub-sample-step resolution on both
    the entry/exit bracket edges and the TCA within each window.

    Args:
        trajectories: TrajectoryMap of NORAD-ID → (T, 3) position array (km, TEME).
        times_jd: 1-D array of T Julian Dates matching trajectory rows.
        threshold_km: Distance threshold in km (default 10.0). Windows are
            defined as intervals where ``distance < threshold_km``.
        objects: Optional dict for future extension (currently unused).

    Returns:
        List of :class:`ConjunctionWindow` objects, sorted by entry_jd.

    Raises:
        AstraError: If ``times_jd`` has fewer than 3 elements or trajectories
            are inconsistent.

    Example::

        import astra
        windows = astra.find_conjunction_windows(
            trajectories=traj_map,
            times_jd=times,
            threshold_km=5.0,
        )
        for w in windows:
            print(f"{w.object_a_id} vs {w.object_b_id}: "
                  f"window={w.duration_s:.1f}s, min_dist={w.min_distance_km:.3f}km")
    """
    times_jd = np.asarray(times_jd, dtype=float)
    if times_jd.ndim != 1 or len(times_jd) < 3:
        raise AstraError("times_jd must be a 1-D array with ≥ 3 elements.")

    norad_ids = list(trajectories.keys())
    if len(norad_ids) < 2:
        return []

    windows: list[ConjunctionWindow] = []
    T = len(times_jd)
    times_s = (times_jd - times_jd[0]) * 86400.0  # local seconds for conditioning

    # Iterate over all unique pairs
    for i in range(len(norad_ids)):
        for j in range(i + 1, len(norad_ids)):
            id_a, id_b = norad_ids[i], norad_ids[j]
            traj_a = trajectories[id_a]
            traj_b = trajectories[id_b]

            if len(traj_a) != T or len(traj_b) != T:
                continue

            # Compute distances at sample points
            dists = np.linalg.norm(traj_a - traj_b, axis=1)

            # Find entry/exit crossings: where distance crosses threshold
            below = dists < threshold_km

            # Build splines for sub-sample refinement
            spline_a = scipy.interpolate.CubicSpline(
                times_s, traj_a, bc_type="natural"
            )
            spline_b = scipy.interpolate.CubicSpline(
                times_s, traj_b, bc_type="natural"
            )

            def _dist_at_s(t_s: float) -> float:
                return float(np.linalg.norm(spline_a(t_s) - spline_b(t_s)))

            # Walk through samples to find contiguous windows
            k = 0
            while k < T:
                if not below[k]:
                    k += 1
                    continue

                # Found start of a window — find the end
                win_start_idx = k
                while k < T and below[k]:
                    k += 1
                win_end_idx = k - 1  # last index that was below threshold

                # ── Refine entry time ────────────────────────────────────
                if win_start_idx > 0:
                    # Bisect between (win_start_idx - 1) and win_start_idx
                    lo_s = float(times_s[win_start_idx - 1])
                    hi_s = float(times_s[win_start_idx])
                    for _ in range(30):  # binary search iterations
                        mid_s = (lo_s + hi_s) / 2.0
                        if _dist_at_s(mid_s) < threshold_km:
                            hi_s = mid_s
                        else:
                            lo_s = mid_s
                    entry_s = hi_s
                else:
                    entry_s = float(times_s[0])

                # ── Refine exit time ─────────────────────────────────────
                if win_end_idx < T - 1:
                    lo_s = float(times_s[win_end_idx])
                    hi_s = float(times_s[win_end_idx + 1])
                    for _ in range(30):
                        mid_s = (lo_s + hi_s) / 2.0
                        if _dist_at_s(mid_s) < threshold_km:
                            lo_s = mid_s
                        else:
                            hi_s = mid_s
                    exit_s = lo_s
                else:
                    exit_s = float(times_s[-1])

                # ── Find TCA within window ───────────────────────────────
                try:
                    from scipy.optimize import minimize_scalar
                    res = minimize_scalar(
                        _dist_at_s,
                        bounds=(entry_s, exit_s),
                        method="bounded",
                        options={"xatol": 1e-3},
                    )
                    if res.success:
                        tca_s = float(res.x)
                        min_dist = float(res.fun)
                    else:
                        sub_dists = dists[win_start_idx:win_end_idx + 1]
                        min_idx = int(np.argmin(sub_dists)) + win_start_idx
                        tca_s = float(times_s[min_idx])
                        min_dist = float(dists[min_idx])
                except Exception:
                    sub_dists = dists[win_start_idx:win_end_idx + 1]
                    min_idx = int(np.argmin(sub_dists)) + win_start_idx
                    tca_s = float(times_s[min_idx])
                    min_dist = float(dists[min_idx])

                entry_jd = float(times_jd[0] + entry_s / 86400.0)
                exit_jd = float(times_jd[0] + exit_s / 86400.0)
                tca_jd = float(times_jd[0] + tca_s / 86400.0)

                windows.append(ConjunctionWindow(
                    object_a_id=id_a,
                    object_b_id=id_b,
                    entry_jd=entry_jd,
                    exit_jd=exit_jd,
                    tca_jd=tca_jd,
                    min_distance_km=min_dist,
                ))

    windows.sort(key=lambda w: w.entry_jd)
    logger.info("Conjunction window search complete: %d windows found.", len(windows))
    return windows
