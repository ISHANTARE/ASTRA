"""ASTRA Core Covariance and Uncertainty Modeling.
Calculates Mahalanobis distance and Probability of Collision (Pc)
by projecting 3D positional covariances onto the 2D encounter plane.
Includes:
- Analytical Foster/Chan Pc (rectilinear short-duration encounters)
- Monte Carlo Pc (co-orbital, slow encounters, curvilinear geometry)
- STM-based covariance propagation (linearized J2 dynamics)
- Empirical covariance estimation (TLE degradation model)
"""
from __future__ import annotations
import math
from typing import Optional, Any
import numpy as np
from scipy.integrate import solve_ivp
from astra.constants import (
    EARTH_EQUATORIAL_RADIUS_KM,
    EARTH_MU_KM3_S2,
    J2,
)
# Import the Numba-compiled NRLMSISE-00 density kernel
# at module scope so it is available in Numba's global closure when
# _acceleration_njit is compiled. Numba @njit functions can only call other
# @njit functions that are in their global namespace at compile time;
# a deferred or local import is invisible to the type-inference pass.
from astra.propagator import _nrlmsise00_density_njit  # noqa: E402
def rotate_covariance_rtn_to_eci(
    cov_rtn: np.ndarray,
    r_eci_km: np.ndarray,
    v_eci_km_s: np.ndarray,
) -> np.ndarray:
    """Map a 3×3 covariance from RTN (diagonal or full) into ECI at the given state.
    RTN: R radial (position), T along-track, N cross-track (angular momentum).
    """
    r = np.asarray(r_eci_km, dtype=float).ravel()
    v = np.asarray(v_eci_km_s, dtype=float).ravel()
    r_mag = float(np.linalg.norm(r))
    h = np.cross(r, v)
    h_mag = float(np.linalg.norm(h))
    if r_mag < 1e-9 or h_mag < 1e-12:
        return np.asarray(cov_rtn, dtype=float).copy()
    r_hat = r / r_mag
    n_hat = h / h_mag
    t_hat = np.cross(n_hat, r_hat)
    t_hat /= max(float(np.linalg.norm(t_hat)), 1e-15)
    B = np.column_stack((r_hat, t_hat, n_hat))
    C = np.asarray(cov_rtn, dtype=float)
    return B @ C @ B.T  # type: ignore[no-any-return]
def _exact_pc_2d_integral(
    miss_2d: np.ndarray,
    inv_C_p: np.ndarray,
    det_C_p: float,
    combined_radius_km: float,
) -> float:
    """Exact Pc via 2-D Gaussian integration over the hard-body collision disk.
    Computes the double integral
        Pc = 1/(2π√det C) ∬_{disk} exp(-½ δᵀ C⁻¹ δ) dx dy
    where δ = (x,y) − miss_2d is the displacement from the covariance peak.
    This is exact for arbitrary (non-diagonal) encounter-plane covariances
    and is accurate where Chan's point approximation fails (Mahalanobis < 1).
    Falls back to Chan's formula on integration failure.
    Args:
        miss_2d: (2,) projected miss vector in encounter plane (km).
        inv_C_p: (2,2) inverse of 2-D projected covariance (km⁻²).
        det_C_p: Determinant of C_p (km⁴).
        combined_radius_km: Hard-body collision radius sum (km).
    Returns:
        Probability of collision in [0.0, 1.0].
    """
    import scipy.integrate
    norm = 1.0 / (2.0 * math.pi * math.sqrt(det_C_p))
    mx, my = float(miss_2d[0]), float(miss_2d[1])
    # AUDIT-D-03 Fix: Fast-path — when the miss distance is much larger than
    # the hard-body radius, Chan's approximation is indistinguishable from the
    # exact integral.  Skip the expensive dblquad and save 5-50 ms per call.
    miss_norm = math.sqrt(mx**2 + my**2)
    if miss_norm > 5.0 * combined_radius_km:
        mahal = float(miss_2d @ inv_C_p @ miss_2d)
        area = math.pi * combined_radius_km**2
        return float(np.clip(norm * math.exp(-0.5 * mahal) * area, 0.0, 1.0))
    # Extract inv-covariance elements once outside the hot loop.
    a11 = float(inv_C_p[0, 0])
    a12 = float(inv_C_p[0, 1]) + float(inv_C_p[1, 0])  # symmetric off-diagonal
    a22 = float(inv_C_p[1, 1])
    def _integrand(y: float, x: float) -> float:
        dx, dy = x - mx, y - my
        return norm * math.exp(-0.5 * (a11 * dx * dx + a12 * dx * dy + a22 * dy * dy))
    rc = combined_radius_km
    def _y_lo(x: float) -> float:
        return -math.sqrt(max(rc * rc - x * x, 0.0))
    def _y_hi(x: float) -> float:
        return math.sqrt(max(rc * rc - x * x, 0.0))
    try:
        result, _err = scipy.integrate.dblquad(  # type: ignore[call-overload]
            _integrand,
            -rc,
            rc,
            _y_lo,
            _y_hi,
            limit=200,
            epsabs=1e-10,
            epsrel=1e-8,
        )
    except Exception:
        # Numerical failure — fall through to Chan approximation.
        area = math.pi * rc**2
        result = math.exp(-0.5 * float(miss_2d @ inv_C_p @ miss_2d)) * area * norm
    return float(np.clip(result, 0.0, 1.0))
def compute_collision_probability(
    miss_vector_km: np.ndarray,
    rel_vel_km_s: np.ndarray,
    cov_a: np.ndarray,
    cov_b: np.ndarray,
    radius_a_km: float = 0.005,
    radius_b_km: float = 0.005,
) -> float:
    """Computes Probability of Collision (Pc) via 2D projection on the encounter plane.
    Uses standard Foster/Chan methodology for projecting combined 3D Cartesian
    covariance onto the B-plane (perpendicular to relative velocity vector).
    Args:
    Args:
        miss_vector_km: (3,) relative position vector at exact TCA (km).
        rel_vel_km_s: (3,) relative velocity vector at TCA (km/s).
        cov_a: (3, 3) positional covariance matrix for Object A (km^2).
        cov_b: (3, 3) positional covariance matrix for Object B (km^2).
        radius_a_km: Hard-body radius for Object A (km).
        radius_b_km: Hard-body radius for Object B (km).
    Returns:
        Probability of collision (float) bounded [0.0, 1.0], or None if covariance is singular.
    """
    combined_radius_km = radius_a_km + radius_b_km
    C = cov_a + cov_b
    if np.all(C == 0):
        # Deterministic collision boolean
        return 1.0 if np.linalg.norm(miss_vector_km) <= combined_radius_km else 0.0
    v_mag = np.linalg.norm(rel_vel_km_s)
    if v_mag == 0:
        return 0.0
    # U_y is along relative velocity direction (out of encounter plane)
    u_y = rel_vel_km_s / v_mag
    # Robust perpendicular (avoids u_y ≈ ŷ with temp = ŷ → degenerate cross)
    abs_u = np.abs(u_y)
    if abs_u[0] <= abs_u[1] and abs_u[0] <= abs_u[2]:
        temp = np.array([1.0, 0.0, 0.0])
    elif abs_u[1] <= abs_u[2]:
        temp = np.array([0.0, 1.0, 0.0])
    else:
        temp = np.array([0.0, 0.0, 1.0])
    u_z = np.cross(u_y, temp)
    u_z /= np.linalg.norm(u_z)
    u_x = np.cross(u_y, u_z)
    # Rotation matrix to 2D encounter plane
    R = np.vstack((u_x, u_z))  # Shape (2, 3)
    # Project 3D combined covariance to 2D
    C_p = R @ C @ R.T
    # Project 3D miss vector into the 2D plane
    r_p = R @ miss_vector_km
    try:
        # AUDIT-NUM-01: Fix 4 - Apply Tikhonov regularization for severe ill-conditioning
        # mirror the MC path logic to ensure robust inversion of 2D B-plane matrices.
        cond = float(np.linalg.cond(C_p))
        if not np.isfinite(cond) or cond > 1e12:
            import logging as _reg_log
            _reg_log.getLogger(__name__).debug(
                "2D B-plane covariance ill-conditioning (cond=%.2e). Applying Tikhonov regularization.",
                cond
            )
            # Add 1e-12 km^2 (1 mm positional variance) to ensure positive-definiteness
            C_p += np.eye(2) * 1e-12
            inv_C_p = np.linalg.inv(C_p)
        else:
            inv_C_p = np.linalg.inv(C_p)
        det_C_p = np.linalg.det(C_p)
    except np.linalg.LinAlgError as _lae:
        import logging
        from astra import config
        _cov_log = logging.getLogger(__name__)
        if config.ASTRA_STRICT_MODE:
            raise ValueError(
                "[ASTRA STRICT] Degenerate 2D encounter covariance matrix — Pc cannot be "
                "computed reliably. This indicates zero uncertainty in one encounter-plane "
                f"axis (singular C_p). Inspect covariance source quality. ({_lae})"
            ) from _lae
        _cov_log.warning(
            "Singular 2D encounter covariance (det ≈ 0) — Pc returned as 1.0 to prevent "
            "false alarms. This is a DATA QUALITY failure, not a verified safe miss."
        )
        return 1.0  # type: ignore[no-any-return]
    if det_C_p <= 0:
        import logging
        from astra import config
        _cov_log = logging.getLogger(__name__)
        if config.ASTRA_STRICT_MODE:
            raise ValueError(
                "[ASTRA STRICT] Non-positive encounter-plane covariance determinant "
                f"det(C_p)={det_C_p:.3e}. Pc computation aborted; covariance is invalid."
            )
        _cov_log.warning(
            f"Non-positive det(C_p)={det_C_p:.3e} in encounter plane — Pc returned as 1.0 "
            "to prevent false alarms. Verify covariance source quality (DATA QUALITY failure)."
        )
        return 1.0  # type: ignore[no-any-return]
    # Mahalanobis distance squared (u^2)
    mahalanobis_sq = float(r_p.T @ inv_C_p @ r_p)
    # Foster/Chan assumes short, nearly rectilinear encounters.
    # For co-orbital (very slow) encounters with mahalanobis_sq < 4.0
    # (within the 2-sigma error ellipse), Chan's formula under-estimates Pc.
    # Attempt MC escalation only when full 6×6 covariances are provided.
    if mahalanobis_sq < 4.0:
        import logging
        from astra import config
        _m5_logger = logging.getLogger(__name__)
        # Only escalate to MC when full 6×6 state covariances are available
        if cov_a.shape == (6, 6) and cov_b.shape == (6, 6):
            if config.ASTRA_STRICT_MODE:
                _m5_logger.info(
                    "Foster/Chan Pc: mahalanobis_sq=%.3f < 4.0 (co-orbital regime). "
                    "Escalating to Exact Monte Carlo integration for accurate Pc.",
                    mahalanobis_sq,
                )
            return compute_collision_probability_mc(  # type: ignore[no-any-return, no-untyped-call]
                miss_vector_km=miss_vector_km,
                rel_vel_km_s=rel_vel_km_s,
                cov_a=cov_a,
                cov_b=cov_b,
                radius_a_km=radius_a_km,
                radius_b_km=radius_b_km,
                n_samples=100_000,
            )
        else:
            # 3×3 positional-only matrices supplied — MC cannot run (needs 6×6).
            # Very close encounters (Mahalanobis < 1): Chan's point
            # approximation can underestimate Pc by 30-40%.  Use the exact
            # 2D Gaussian integral via scipy.integrate.dblquad.
            if mahalanobis_sq < 1.0:
                _m5_logger.warning(
                    "Near-hit encounter (mahalanobis_sq=%.3f < 1.0) with 3×3 covariances: "
                    "using exact scipy.integrate.dblquad. "
                    "Supply 6×6 CDM covariances to use Monte Carlo instead.",
                    mahalanobis_sq,
                )
                return _exact_pc_2d_integral(r_p, inv_C_p, det_C_p, combined_radius_km)
            # mahalanobis_sq in [1.0, 4.0) with 3×3 — Chan/Foster is adequate;
            # emit a warning so the caller can decide whether to provide CDM data.
            _m5_logger.warning(
                "co-orbital encounter (mahalanobis_sq=%.3f) detected but only 3×3 "
                "covariances available — Foster/Chan formula applied (may under-estimate Pc). "
                "Supply 6×6 state covariances from CDM for MC accuracy.",
                mahalanobis_sq,
            )
    # Area of collision cross-section
    area = math.pi * (combined_radius_km**2)
    # 2D Gaussian Density Integration Approximation (Chan's formulation)
    p_c = math.exp(-0.5 * mahalanobis_sq) * area / (2.0 * math.pi * math.sqrt(det_C_p))
    return float(np.clip(p_c, 0.0, 1.0))
def estimate_covariance(
    time_since_epoch_days: float, f107_flux: float = 150.0
) -> np.ndarray:
    """Generates an estimated positional covariance matrix (3x3) in RTN frame.
    Models realistic anisotropic TLE degradation based on Vallado & Alfano (2014):
    - Radial (R): grows quadratically with time (gravity model uncertainty)
    - In-track (T): grows cubically with time (drag uncertainty dominates)
    - Cross-track (N): grows linearly with time (inclination uncertainty)
    The returned matrix is diagonal in RTN. For full accuracy, this should be
    rotated to the ECI/TEME frame using the satellite's orbital state, but for
    encounter-plane projection this diagonal approximation produces physically
    meaningful Pc values.
    .. warning::
        In ``ASTRA_STRICT_MODE=True`` this function raises ``ValueError``.
        Collision probability pipelines require real Orbit Determination
        covariances, not heuristic estimates.
    Args:
        time_since_epoch_days: Days elapsed since TLE epoch.
        f107_flux: Solar F10.7 index for drag scaling.
    Returns:
        np.ndarray: (3, 3) diagonal covariance matrix in km^2.
    Raises:
        ValueError: In STRICT_MODE — real OD covariance data must be supplied.
    """
    from astra import config
    from astra.errors import AstraError
    if config.ASTRA_STRICT_MODE:
        # AUDIT-F-04 Fix: Raise typed AstraError (not plain ValueError) so
        # callers using 'except AstraError' catch this correctly.
        raise AstraError(
            "[ASTRA STRICT] estimate_covariance() is disabled in strict mode. "
            "Collision probability calculations require real Orbit Determination "
            "covariance matrices from CDM data or propagate_covariance_stm(). "
            "Set ASTRA_STRICT_MODE=False to allow heuristic fallbacks."
        )
    import logging
    _cov_logger = logging.getLogger(__name__)
    _cov_logger.warning(
        "estimate_covariance() is generating a SYNTHETIC covariance matrix. "
        "Collision Probability (Pc) outputs are NOT statistically valid — "
        "supply a real OD covariance for mission-critical analysis."
    )
    days = max(0.1, abs(time_since_epoch_days))
    drag_scale = max(0.1, f107_flux / 150.0)
    # Piecewise in-track growth model.
    # Radial (R) and Normal (N) grow quadratically (ballistic uncertainty).
    # In-track (T) grows linearly for fresh TLEs (< 7 days) then
    # quadratically afterwards, driven by unmodeled drag and SRP.
    # The old cubic `days**3` model over-estimated T uncertainty by 20-40×
    # for 1–3 day old TLEs, inflating Pc for correctly-catalogued objects.
    sigma_r_km = min(0.01 + 0.02 * days**2, 50.0)
    if days <= 7.0:
        sigma_t_km = min(0.1 + 0.3 * drag_scale * days, 5.0)
    else:
        sigma_t_km = min(5.0 + drag_scale * 0.5 * (days - 7.0) ** 2, 100.0)
    sigma_n_km = min(0.01 + 0.05 * days, 20.0)
    return np.diag([sigma_r_km**2, sigma_t_km**2, sigma_n_km**2])
def _hcw_min_distance(
    r0_rtm: Any,
    v0_rtm: Any,
    n_rad_s: float,
    t_window_s: float = 3600.0,
    n_steps: int = 200,
) -> float:
    """Find the minimum relative distance along the HCW trajectory.
    Uses the Hill-Clohessy-Wiltshire (HCW) closed-form equations to propagate
    the relative state in the RTN (co-moving) frame and find the minimum
    separation over +/- t_window_s around TCA (Time of Closest Approach).
    HCW solution in RTN coordinates (Schaub and Junkins, Analytical Mechanics
    of Space Systems, Section 9.3):
        x(t) = (4-3c)x0 + (s/n)dx0 + (2/n)(1-c)dy0
        y(t) = 6(s-nt)x0 + y0 - (2/n)(1-c)dx0 + (1/n)(4s-3nt)dy0
        z(t) = z0*c + (dz0/n)*s
    where c=cos(nt), s=sin(nt), n=mean motion [rad/s].
    Args:
        r0_rtm: (3,) relative position in RTN frame at TCA (km).
        v0_rtm: (3,) relative velocity in RTN frame at TCA (km/s).
        n_rad_s: Mean motion of the nominal orbit (rad/s).
        t_window_s: Search window +/-t_window_s around TCA (s).
        n_steps: Number of time samples in the window.
    Returns:
        Minimum relative distance in km.
    """
    import numpy as _np
    times = _np.linspace(-t_window_s, t_window_s, n_steps)
    x0, y0, z0 = r0_rtm[0], r0_rtm[1], r0_rtm[2]
    dx0, dy0, dz0 = v0_rtm[0], v0_rtm[1], v0_rtm[2]
    nt_arr = n_rad_s * times
    c_arr = _np.cos(nt_arr)
    s_arr = _np.sin(nt_arr)
    inv_n = 1.0 / n_rad_s
    x_arr = (
        (4.0 - 3.0 * c_arr) * x0
        + s_arr * dx0 * inv_n
        + 2.0 * (1.0 - c_arr) * dy0 * inv_n
    )
    y_arr = (
        6.0 * (s_arr - nt_arr) * x0
        + y0
        - 2.0 * (1.0 - c_arr) * dx0 * inv_n
        + (4.0 * s_arr - 3.0 * nt_arr) * dy0 * inv_n
    )
    z_arr = z0 * c_arr + dz0 * inv_n * s_arr
    dist = _np.sqrt(x_arr**2 + y_arr**2 + z_arr**2)
    return float(_np.min(dist))
def compute_collision_probability_mc(
    miss_vector_km: Any,
    rel_vel_km_s: Any,
    cov_a: Any,
    cov_b: Any,
    radius_a_km: float = 0.005,
    radius_b_km: float = 0.005,
    n_samples: int = 100_000,
    seed: Any = None,
    mean_motion_rad_s: Any = None,
) -> float:
    """Monte Carlo Collision Probability for long-duration and co-orbital encounters.
    Trajectory model selection (HIGH-04):
    For encounters with relative speed >= 0.1 km/s (crossing or chase geometry),
    a linear relative path is used per sample -- accurate and fast for these cases.
    For encounters with relative speed < 0.1 km/s (co-orbital / formation flying),
    the Hill-Clohessy-Wiltshire (HCW) equations propagate each sampled state along
    the correct curved orbit-relative trajectory.  The prior linear model
    systematically under-estimates Pc in this regime because it ignores the orbital
    curvature that brings co-planar objects back toward each other on timescales of
    minutes to hours (critical for Starlink/OneWeb same-plane conjunctions).
    The HCW evaluation is fully vectorised over all samples via numpy broadcasting
    (n_samples x n_time_steps) -- no per-sample Python loop.
    Args:
        miss_vector_km: (3,) relative position at TCA (km).
        rel_vel_km_s: (3,) relative velocity at TCA (km/s).
        cov_a: (6,6) Object A full covariance (km^2, km^2/s^2 etc).
        cov_b: (6,6) Object B full covariance (km^2, km^2/s^2 etc).
        radius_a_km: Object A hard-body radius (km). Default 5 m.
        radius_b_km: Object B hard-body radius (km). Default 5 m.
        n_samples: Number of Monte Carlo samples (default 100,000).
        seed: Optional RNG seed for reproducibility.
        mean_motion_rad_s: Mean orbital motion (rad/s) for HCW propagation.
            If None, defaults to LEO 90-min orbit (2*pi/5400 ~ 1.164e-3 rad/s).
            Pass the actual mean motion of the primary object for accuracy.
    Returns:
        Probability of collision (float) bounded [0.0, 1.0].
    """
    if cov_a.shape == (3, 3) or cov_b.shape == (3, 3):
        raise ValueError(
            "Monte-Carlo (6DOF) collision probability requires a 6x6 covariance matrix. "
            "Legacy 3x3 matrices are unsupported here; fall back to 2D Foster-Chan."
        )
    combined_radius_km = radius_a_km + radius_b_km
    combined_cov = cov_a + cov_b
    if np.all(combined_cov == 0):
        return 1.0 if np.linalg.norm(miss_vector_km) <= combined_radius_km else 0.0
    rng = np.random.default_rng(seed)
    sym = 0.5 * (combined_cov + combined_cov.T)
    # AUDIT-NUM-01 Fix: Apply Tikhonov regularization for severe ill-conditioning
    # Prevents cholesky NaN generation or breakdown on colinear relative velocity
    cond = float(np.linalg.cond(sym))
    if not np.isfinite(cond) or cond > 1e12:
        import logging as _reg_log
        _reg_log.getLogger(__name__).debug(
            "Covariance ill-conditioning detected (cond=%.2e). Applying Tikhonov regularization.",
            cond,
        )
        sym += np.eye(sym.shape[0]) * 1e-12
    try:
        L = np.linalg.cholesky(sym)
    except np.linalg.LinAlgError:
        eigvals, eigvecs = np.linalg.eigh(sym)
        eigvals = np.clip(eigvals, 1e-12, None)
        combined_cov = eigvecs @ np.diag(eigvals) @ eigvecs.T
        try:
            L = np.linalg.cholesky(combined_cov)
        except np.linalg.LinAlgError:
            return 1.0 if np.linalg.norm(miss_vector_km) <= combined_radius_km else 0.0
    samples = rng.standard_normal((n_samples, 6)) @ L.T
    r_samples = miss_vector_km + samples[:, :3]
    v_samples = rel_vel_km_s + samples[:, 3:]
    rel_speed = float(np.linalg.norm(rel_vel_km_s))
    # HIGH-04: HCW model for co-orbital encounters (rel. speed < 0.1 km/s).
    # Orbital curvature brings co-planar objects back within one orbital period;
    # the linear model ignores this and systematically under-estimates Pc.
    _HCW_THRESHOLD_KM_S = 0.1  # km/s = 100 m/s
    if rel_speed < _HCW_THRESHOLD_KM_S:
        import logging as _hcw_log
        _hcw_log.getLogger(__name__).info(
            "co-orbital encounter (rel_speed=%.5f km/s < %.2f km/s): "
            "switching to HCW trajectory model for MC Pc (HIGH-04).",
            rel_speed,
            _HCW_THRESHOLD_KM_S,
        )
        n_rad = (
            mean_motion_rad_s
            if mean_motion_rad_s is not None
            else (2.0 * math.pi / 5400.0)
        )
        # Build approximate RTN basis from miss vector and velocity directions.
        r_mag = float(np.linalg.norm(miss_vector_km))
        v_hat = (
            rel_vel_km_s / rel_speed if rel_speed > 1e-10 else np.array([0.0, 1.0, 0.0])
        )
        r_hat = miss_vector_km / r_mag if r_mag > 1e-10 else np.array([1.0, 0.0, 0.0])
        n_hat = np.cross(r_hat, v_hat)
        n_hat_mag = float(np.linalg.norm(n_hat))
        if n_hat_mag < 1e-10:
            # Degenerate (r parallel to v): pick any perpendicular axis
            perp = (
                np.array([0.0, 0.0, 1.0])
                if abs(r_hat[2]) < 0.9
                else np.array([0.0, 1.0, 0.0])
            )
            n_hat = np.cross(r_hat, perp)
            n_hat /= max(float(np.linalg.norm(n_hat)), 1e-15)
        else:
            n_hat /= n_hat_mag
        t_hat = np.cross(n_hat, r_hat)
        t_hat /= max(float(np.linalg.norm(t_hat)), 1e-15)
        # ECI to RTN rotation matrix (rows are basis vectors)
        R_eci_rtn = np.vstack([r_hat, t_hat, n_hat])  # (3, 3)
        r_rtm = (R_eci_rtn @ r_samples.T).T  # (n_samples, 3)
        v_rtm = (R_eci_rtn @ v_samples.T).T  # (n_samples, 3)
        # Refine the time window for the HCW propagator.
        # For fast co-orbital encounters, the 1-period window is too coarse.
        # Ensure the window captures at least several times the encounter duration.
        t_window_s = min(2.0 * math.pi / n_rad, 10.0 * combined_radius_km / max(rel_speed, 1e-6))
        nt = n_rad
        times = np.linspace(-t_window_s, t_window_s, 400)
        nt_arr = nt * times
        c_arr = np.cos(nt_arr)
        s_arr = np.sin(nt_arr)
        inv_n = 1.0 / nt
        # Vectorised HCW: (n_samples,1) broadcast with (400,) -> (n_samples, 400)
        x0 = r_rtm[:, 0:1]
        y0 = r_rtm[:, 1:2]
        z0 = r_rtm[:, 2:3]
        dx0 = v_rtm[:, 0:1]
        dy0 = v_rtm[:, 1:2]
        dz0 = v_rtm[:, 2:3]
        x_t = (
            (4.0 - 3.0 * c_arr) * x0
            + s_arr * dx0 * inv_n
            + 2.0 * (1.0 - c_arr) * dy0 * inv_n
        )
        y_t = (
            6.0 * (s_arr - nt_arr) * x0
            + y0
            - 2.0 * (1.0 - c_arr) * dx0 * inv_n
            + (4.0 * s_arr - 3.0 * nt_arr) * dy0 * inv_n
        )
        z_t = z0 * c_arr + dz0 * inv_n * s_arr
        dist_t = np.sqrt(x_t**2 + y_t**2 + z_t**2)  # (n_samples, 400)
        d_min = np.min(dist_t, axis=1)  # (n_samples,)
    else:
        # Linear trajectory -- fast and correct for crossing/chase encounters.
        v_dot_v = np.sum(v_samples * v_samples, axis=1)
        v_dot_v[v_dot_v < 1e-16] = 1e-16
        t_min = -np.sum(r_samples * v_samples, axis=1) / v_dot_v
        # AUDIT-C-04 Fix: clamp to +/-3600 s window around TCA
        t_min = np.clip(t_min, -3600.0, 3600.0)
        r_min_vec = r_samples + v_samples * t_min[:, np.newaxis]
        d_min = np.linalg.norm(r_min_vec, axis=1)
    hits = np.sum(d_min <= combined_radius_km)
    return float(hits / n_samples)
from astra._numba_compat import njit  # noqa: E402
from astra.constants import J3, J4  # noqa: E402
@njit(fastmath=True, cache=True)
def _acceleration_njit(
    r: np.ndarray,
    v: np.ndarray,
    Bc: float,
    rho_ref: float,
    H_km: float,
    rho_ref_alt_km: float,
    f107_obs: float = 150.0,
    f107_adj: float = 150.0,
    ap_daily: float = 15.0,
    use_nrlmsise: bool = False,
) -> np.ndarray:
    """Covariance-STM acceleration kernel: J2/J3/J4 + drag.
    Supports two atmosphere models controlled by ``use_nrlmsise``:
    * ``use_nrlmsise=False`` (default): simple exponential density profile
      anchored at ``rho_ref_alt_km``.  Fast; suitable when space-weather data
      is unavailable.
    * ``use_nrlmsise=True``: calls the native Numba NRLMSISE-00 implementation
      (``_nrlmsise00_density_njit``) with the supplied F10.7 / Ap indices.
      **This mode must be selected when the Cowell propagator uses the
      NRLMSISE-00 model** (``DragConfig.model == "NRLMSISE00"``) so that the
      trajectory and its covariance use the same atmospheric drag model.
      Mismatched models cause 3-5× covariance growth-rate errors at solar max.
      
    """
    x, y, z = r[0], r[1], r[2]
    r_mag = np.linalg.norm(r)
    r2 = r_mag**2
    r3 = r2 * r_mag
    # --- Two-body gravity ---
    a_total = -EARTH_MU_KM3_S2 * r / r3
    # --- Earth Oblateness (J2, J3, J4) ---
    z2 = z**2
    r5 = r3 * r2
    r7 = r5 * r2
    r9 = r7 * r2
    Re = EARTH_EQUATORIAL_RADIUS_KM
    mu = EARTH_MU_KM3_S2
    fJ2 = 1.5 * J2 * mu * Re**2 / r5
    a_total[0] += fJ2 * x * (5.0 * z2 / r2 - 1.0)
    a_total[1] += fJ2 * y * (5.0 * z2 / r2 - 1.0)
    a_total[2] += fJ2 * z * (5.0 * z2 / r2 - 3.0)
    fJ3 = 0.5 * J3 * mu * Re**3 / r7
    a_total[0] += fJ3 * x * (35.0 * z2 * z / r2 - 15.0 * z)
    a_total[1] += fJ3 * y * (35.0 * z2 * z / r2 - 15.0 * z)
    a_total[2] += fJ3 * (35.0 * z2 * z2 / r2 - 30.0 * z2 + 3.0 * r2)
    fJ4 = 0.625 * J4 * mu * Re**4 / r9
    z4 = z2 * z2
    a_total[0] += fJ4 * x * (63.0 * z4 / r2 - 42.0 * z2 + 3.0 * r2)
    a_total[1] += fJ4 * y * (63.0 * z4 / r2 - 42.0 * z2 + 3.0 * r2)
    a_total[2] += fJ4 * z * (63.0 * z4 / r2 - 70.0 * z2 + 15.0 * r2)
    # --- Atmospheric Drag (LEO Only) ---
    # When use_nrlmsise=True, use the native
    # Numba NRLMSISE-00 implementation (_nrlmsise00_density_njit) to match
    # the drag model used by the Cowell propagator.
    # always used the exponential profile regardless of the flag, causing
    # the propagated trajectory and covariance matrix to diverge physically
    # by 3-5x during solar maximum conditions.
    alt = r_mag - Re
    if alt < 1500.0 and Bc > 0.0:
        if use_nrlmsise and alt >= 100.0:
            # NRLMSISE-00 Bates-profile model — consistent with Cowell propagator.
            rho = _nrlmsise00_density_njit(alt, f107_obs, f107_adj, ap_daily)
        else:
            # Exponential fallback for altitudes < 100 km or when model is disabled.
            rho = rho_ref * math.exp(-(alt - rho_ref_alt_km) / H_km)
        # DEF-002: co-rotating atmosphere correction (CRITICAL FIX)
        # v_rel = v - omega_earth x r  subtracts the ~0.46 km/s equatorial rotation.
        # Using inertial velocity v directly introduced ~6% systematic error in
        # drag partial derivatives (identified defect DEF-002 in audit).
        EARTH_OMEGA = 7.292115146706979e-5  # rad/s — IAU/IERS 2010
        # Manual cross product for Numba compatibility: omega x r = [0,0,w] x [x,y,z]
        # = [0*z - w*y, w*x - 0*z, 0*y - 0*x] = [-w*y, w*x, 0]
        vx_rel = v[0] - (-EARTH_OMEGA * r[1])
        vy_rel = v[1] - (EARTH_OMEGA * r[0])
        vz_rel = v[2]  # no z-component from Earth rotation
        v_rel = np.array([vx_rel, vy_rel, vz_rel])
        v_rel_mag = np.linalg.norm(v_rel)
        if v_rel_mag > 1e-6:
            # 1e-6 (m^2 to km^2) * 1e9 (kg/m^3 to kg/km^3) = 1e3; Bc is m^2/kg
            a_drag = -0.5 * rho * 1e3 * Bc * v_rel_mag * v_rel
            a_total += a_drag
    return a_total
@njit(fastmath=True, cache=True)
def _stm_jacobian_njit(
    r: np.ndarray,
    v: np.ndarray,
    Bc: float,
    rho_ref: float,
    H_km: float,
    rho_ref_alt_km: float,
    f107_obs: float = 150.0,
    f107_adj: float = 150.0,
    ap_daily: float = 15.0,
    use_nrlmsise: bool = False,
) -> np.ndarray:
    # Central-difference Jacobian step (km)
    # eps_v reduced from 1e-4 (100 m/s) to 1e-6 (1 mm/s) to prevent
    # nonlinear velocity averaging in the drag Jacobian column. The large step
    # was causing the finite-difference to straddle the nonlinear drag regime,
    # producing a systematically wrong da/dv column (covariance.py fix, matching
    # the identical fix already applied in propagator._propagator_jacobian_njit).
    eps_r = 1e-4
    eps_v = 1e-6
    J = np.zeros((6, 6))
    # Upper-right: dr/dt = v -> d(dr/dt)/dv = I
    J[0, 3] = 1.0
    J[1, 4] = 1.0
    J[2, 5] = 1.0
    # Lower-left: da/dr
    for i in range(3):
        r_plus = r.copy()
        r_minus = r.copy()
        r_plus[i] += eps_r
        r_minus[i] -= eps_r
        a_plus = _acceleration_njit(r_plus, v, Bc, rho_ref, H_km, rho_ref_alt_km, f107_obs, f107_adj, ap_daily, use_nrlmsise)
        a_minus = _acceleration_njit(r_minus, v, Bc, rho_ref, H_km, rho_ref_alt_km, f107_obs, f107_adj, ap_daily, use_nrlmsise)
        J[3:, i] = (a_plus - a_minus) / (2.0 * eps_r)
    # Lower-right: da/dv (mainly for drag; captures co-rotation effect too)
    for i in range(3):
        v_plus = v.copy()
        v_minus = v.copy()
        v_plus[i] += eps_v
        v_minus[i] -= eps_v
        a_plus = _acceleration_njit(r, v_plus, Bc, rho_ref, H_km, rho_ref_alt_km, f107_obs, f107_adj, ap_daily, use_nrlmsise)
        a_minus = _acceleration_njit(r, v_minus, Bc, rho_ref, H_km, rho_ref_alt_km, f107_obs, f107_adj, ap_daily, use_nrlmsise)
        J[3:, 3 + i] = (a_plus - a_minus) / (2.0 * eps_v)
    return J
@njit(fastmath=True, cache=True)
def _stm_derivatives_njit(
    t: float,
    y: np.ndarray,
    Bc: float,
    rho_ref: float,
    H_km: float,
    rho_ref_alt_km: float,
    f107_obs: float = 150.0,
    f107_adj: float = 150.0,
    ap_daily: float = 15.0,
    use_nrlmsise: bool = False,
) -> np.ndarray:
    r = y[:3]
    v = y[3:6]
    Phi = y[6:].reshape((6, 6))
    a_total = _acceleration_njit(r, v, Bc, rho_ref, H_km, rho_ref_alt_km, f107_obs, f107_adj, ap_daily, use_nrlmsise)
    A = _stm_jacobian_njit(r, v, Bc, rho_ref, H_km, rho_ref_alt_km, f107_obs, f107_adj, ap_daily, use_nrlmsise)
    dPhi = A @ Phi
    dy = np.empty(42)
    dy[:3] = v
    dy[3:6] = a_total
    # Flatten dPhi back to the state vector
    dy[6:] = dPhi.ravel()
    return dy
def propagate_covariance_stm(
    t_jd0: float,
    r0_km: np.ndarray,
    v0_km_s: np.ndarray,
    cov0_6x6: np.ndarray,
    duration_s: float,
    drag_config: Optional[Any] = None,
) -> np.ndarray:
    """Propagate a full 6x6 covariance matrix using the State Transition Matrix.
    Uses linearized J2/J3/J4 + exponential drag dynamics to compute the 6x6 STM
    via numerical integration, mapping the full 6x6 state uncertainty:
        C(t) = Φ(t, t₀) · C₀ · Φ(t, t₀)ᵀ
    The state-transition matrix Φ is computed by integrating the variational
    equations along the nominal trajectory. The Jacobian includes:
    - Point-mass gravity (μ/r³)
    - Zonal harmonics (J2, J3, J4)
    - **Atmospheric Drag** partials (∂a/∂v) for LEO satellites.
    """
    y0 = np.zeros(42)
    y0[:3] = r0_km
    y0[3:6] = v0_km_s
    y0[6:] = np.eye(6).ravel()
    # Capture drag constants for STM Jacobian
    Bc = 0.0
    rho_ref = 0.0
    H_km = 50.0  # Default scale height (km)
    # DEF-001 (Strategy A): compute rho_ref at actual initial orbit altitude, not 400 km
    from astra.constants import (
        EARTH_EQUATORIAL_RADIUS_KM as _Re,
        DRAG_MIN_ALTITUDE_KM as _min_alt,
    )
    r0_mag = float(np.linalg.norm(r0_km))
    rho_ref_alt_km = max(r0_mag - _Re, _min_alt)
    if drag_config:
        Bc = float(drag_config.cd * drag_config.area_m2 / drag_config.mass_kg)
        from astra.constants import DRAG_REF_DENSITY_KG_M3, DRAG_SCALE_HEIGHT_KM
        # Initialize density at actual orbit altitude for physical accuracy (DEF-001)
        try:
            from astra.data_pipeline import (
                atmospheric_density_empirical,
                get_space_weather,
            )
            f107_obs, f107_adj, ap = get_space_weather(t_jd0)
            rho_ref = atmospheric_density_empirical(
                rho_ref_alt_km, f107_obs, f107_adj, ap
            )
        except Exception as _sw_exc:
            from astra import config
            if config.ASTRA_STRICT_MODE:
                from astra.errors import PropagationError
                raise PropagationError(
                    "[ASTRA STRICT] STM covariance propagation: space-weather lookup "
                    f"failed ({_sw_exc}). Falling back to the reference density "
                    f"DRAG_REF_DENSITY_KG_M3={DRAG_REF_DENSITY_KG_M3} kg/m³ would "
                    "silently degrade covariance accuracy and corrupt downstream Pc "
                    "estimates. Run astra.data_pipeline.load_space_weather() to "
                    "populate the cache, or set ASTRA_STRICT_MODE=False to allow "
                    "the empirical fallback.",
                    norad_id="COVARIANCE_STM",
                ) from _sw_exc
            import logging as _cov_log
            _cov_log.getLogger(__name__).warning(
                "STM atmosphere: space-weather lookup failed (%s). "
                "Falling back to DRAG_REF_DENSITY_KG_M3=%.3e kg/m³ at 400 km. "
                "Covariance accuracy degraded. Set ASTRA_STRICT_MODE=True to "
                "detect this silently in production.",
                _sw_exc, DRAG_REF_DENSITY_KG_M3,
            )
            rho_ref = DRAG_REF_DENSITY_KG_M3  # fallback
            rho_ref_alt_km = 400.0
        H_km = DRAG_SCALE_HEIGHT_KM
    # Pass space-weather params and use_nrlmsise flag
    # so the STM Jacobian uses the same atmosphere model as the trajectory.
    use_nrlmsise_stm = False
    f107_stm = 150.0
    f107_adj_stm = 150.0
    ap_stm = 15.0
    if drag_config is not None:
        use_nrlmsise_stm = getattr(drag_config, "model", "EXPONENTIAL") == "NRLMSISE00"
        if use_nrlmsise_stm:
            try:
                from astra.data_pipeline import get_space_weather
                f107_stm, f107_adj_stm, ap_stm = get_space_weather(t_jd0)
            except Exception:
                pass  # fall back to defaults
    sol = solve_ivp(
        _stm_derivatives_njit,
        t_span=(0.0, duration_s),
        y0=y0,
        args=(Bc, rho_ref, H_km, rho_ref_alt_km, f107_stm, f107_adj_stm, ap_stm, use_nrlmsise_stm),
        method="DOP853",
        rtol=1e-10,
        atol=1e-12,
    )
    if not sol.success:
        # ALWAYS raise PropagationError on STM
        # integration failure. Silently returning the initial covariance is
        # life-threatening: uncertainty does not grow over time, producing
        # artificially low Pc values and false-negative miss classifications.
        # This is unconditional — there is no safe fallback for a failed STM.
        from astra.errors import PropagationError
        raise PropagationError(
            f"STM covariance propagation failed: {sol.message}. "
            "Cannot return initial covariance as a fallback — this would freeze "
            "uncertainty at epoch and produce falsely low collision probabilities. "
            "Inspect drag_config, initial state, and integration tolerances.",
            norad_id="COVARIANCE_STM",
        )
    Phi_final = sol.y[6:, -1].reshape(6, 6)
    return Phi_final @ cov0_6x6 @ Phi_final.T  # type: ignore[no-any-return]
