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
    return B @ C @ B.T


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
        result, _err = scipy.integrate.dblquad(
            _integrand,
            -rc, rc,
            _y_lo, _y_hi,
            limit=200,
            epsabs=1e-10,
            epsrel=1e-8,
        )
    except Exception:
        # Numerical failure — fall through to Chan approximation.
        area = math.pi * rc ** 2
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
        miss_vector_km: (3,) relative position vector at exact TCA (km).
        rel_vel_km_s: (3,) relative velocity vector at TCA (km/s).
        cov_a: (3, 3) positional covariance matrix for Object A (km^2).
        cov_b: (3, 3) positional covariance matrix for Object B (km^2).
        combined_radius_km: Hard-body collision radius sum (km).
        
    Returns:
        Probability of collision (float) bounded [0.0, 1.0].
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
        cond = float(np.linalg.cond(C_p))
        if not np.isfinite(cond) or cond > 1e12:
            inv_C_p = np.linalg.pinv(C_p)
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
            "Singular 2D encounter covariance (det ≈ 0) — Pc returned as 0.0. "
            "This may indicate a deterministic trajectory or zero-uncertainty axis. "
            "Do NOT interpret 0.0 as a safe miss — verify covariance data quality."
        )
        return 0.0

    if det_C_p <= 0:
        import logging
        from astra import config
        _cov_log = logging.getLogger(__name__)
        _cov_log.warning(
            f"Non-positive det(C_p)={det_C_p:.3e} in encounter plane — Pc returned as 0.0. "
            "Verify that the combined covariance matrix is positive semi-definite."
        )
        return 0.0
        
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
                    mahalanobis_sq
                )
            return compute_collision_probability_mc(
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
                mahalanobis_sq
            )

    # Area of collision cross-section
    area = math.pi * (combined_radius_km ** 2)
    
    # 2D Gaussian Density Integration Approximation (Chan's formulation)
    p_c = math.exp(-0.5 * mahalanobis_sq) * area / (2.0 * math.pi * math.sqrt(det_C_p))
    
    return float(np.clip(p_c, 0.0, 1.0))


def estimate_covariance(time_since_epoch_days: float, f107_flux: float = 150.0) -> np.ndarray:
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
    if config.ASTRA_STRICT_MODE:
        raise ValueError(
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

    sigma_r_km = min(0.05 + 0.5 * days**2, 50.0)
    sigma_t_km = min(0.1 + (2.0 * drag_scale) * days**3, 100.0)
    sigma_n_km = min(0.05 + 0.1 * days, 50.0)

    return np.diag([sigma_r_km**2, sigma_t_km**2, sigma_n_km**2])


def compute_collision_probability_mc(
    miss_vector_km: np.ndarray,
    rel_vel_km_s: np.ndarray,
    cov_a: np.ndarray,
    cov_b: np.ndarray,
    radius_a_km: float = 0.005,
    radius_b_km: float = 0.005,
    n_samples: int = 100_000,
    seed: int | None = None,
) -> float:
    """Monte Carlo Collision Probability for long-duration/co-orbital encounters.

    Unlike the analytical Foster/Chan method (which assumes rectilinear motion),
    this Monte Carlo approach works for arbitrary encounter geometries including
    co-orbital (Starlink vs Starlink) and slow planetary-approach trajectories.

    The internal minimum-distance model is linear in relative velocity per
    sample. For very slow co-orbital encounters (relative speed ≲ 10 m/s) the
    sampled geometry can underestimate curvature; validate with higher-fidelity
    ephemeris or shorter screening steps.

    Draws samples from the combined 3D Gaussian covariance and counts the
    fraction falling within the combined hard-body collision sphere.

    Args:
        miss_vector_km: (3,) relative position at TCA (km).
        rel_vel_km_s: (3,) relative velocity at TCA (km/s).
        cov_a: (6,6) Object A full covariance (km², km²/s² etc).
        cov_b: (6,6) Object B full covariance (km², km²/s² etc).
        radius_a_km: Object A hard-body radius (km).
        radius_b_km: Object B hard-body radius (km).
        n_samples: Number of Monte Carlo samples (default 100,000).
        seed: Optional RNG seed for reproducibility.

    Returns:
        Probability of collision (float) bounded [0.0, 1.0].
    """
    if cov_a.shape == (3, 3) or cov_b.shape == (3, 3):
        raise ValueError("Monte-Carlo (6DOF) collision probability requires a 6x6 covariance matrix. Legacy 3x3 matrices are unsupported here; fallback to 2D Foster-Chan.")
    combined_radius_km = radius_a_km + radius_b_km
    combined_cov = cov_a + cov_b

    if np.all(combined_cov == 0):
        return 1.0 if np.linalg.norm(miss_vector_km) <= combined_radius_km else 0.0

    rng = np.random.default_rng(seed)

    sym = 0.5 * (combined_cov + combined_cov.T)
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

    # Sample 6D error states (n_samples, 6)
    samples = rng.standard_normal((n_samples, 6)) @ L.T
    
    # 6D Relative Trajectories
    r_samples = miss_vector_km + samples[:, :3]
    v_samples = rel_vel_km_s + samples[:, 3:]
    
    # Calculate geometric minimum distance along the linear relative path for EACH sample
    # t_min = -(r · v) / (v · v)
    v_dot_v = np.sum(v_samples * v_samples, axis=1)
    # Avoid division by zero for stationary samples
    v_dot_v[v_dot_v < 1e-16] = 1e-16
    
    t_min = -np.sum(r_samples * v_samples, axis=1) / v_dot_v
    
    # Minimum distance vectors
    r_min_vec = r_samples + v_samples * t_min[:, np.newaxis]
    d_min = np.linalg.norm(r_min_vec, axis=1)
    
    hits = np.sum(d_min <= combined_radius_km)

    return float(hits / n_samples)


try:
    from numba import njit
    _NUMBA_AVAILABLE = True
except ImportError:
    _NUMBA_AVAILABLE = False
    def njit(*args, **kwargs):
        """No-op decorator when Numba is unavailable."""
        def decorator(f):
            return f
        return decorator if (args and callable(args[0])) else decorator
    import logging as _logging
    _logging.getLogger(__name__).warning(
        "Numba not installed — STM covariance kernels will run in pure-Python mode."
    )
from astra.constants import J3, J4

@njit(fastmath=True, cache=True)
def _acceleration_njit(r: np.ndarray, v: np.ndarray, Bc: float, rho_ref: float, H_km: float, rho_ref_alt_km: float) -> np.ndarray:
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
    alt = r_mag - Re
    if alt < 1500.0 and Bc > 0.0:
        # DEF-001: reference density at rho_ref_alt_km (initial orbit altitude, not hardcoded 400 km)
        rho = rho_ref * math.exp(-(alt - rho_ref_alt_km) / H_km)

        # DEF-002: co-rotating atmosphere correction (CRITICAL FIX)
        # v_rel = v - omega_earth x r  subtracts the ~0.46 km/s equatorial rotation.
        # Using inertial velocity v directly introduced ~6% systematic error in
        # drag partial derivatives (identified defect DEF-002 in audit).
        EARTH_OMEGA = 7.292115146706979e-5   # rad/s — IAU/IERS 2010
        # Manual cross product for Numba compatibility: omega x r = [0,0,w] x [x,y,z]
        # = [0*z - w*y, w*x - 0*z, 0*y - 0*x] = [-w*y, w*x, 0]
        vx_rel = v[0] - (-EARTH_OMEGA * r[1])
        vy_rel = v[1] - ( EARTH_OMEGA * r[0])
        vz_rel = v[2]  # no z-component from Earth rotation
        v_rel = np.array([vx_rel, vy_rel, vz_rel])
        v_rel_mag = np.linalg.norm(v_rel)
        if v_rel_mag > 1e-6:
            # 1e-6 (m^2 to km^2) * 1e9 (kg/m^3 to kg/km^3) = 1e3; Bc is m^2/kg
            a_drag = -0.5 * rho * 1e3 * Bc * v_rel_mag * v_rel
            a_total += a_drag

    return a_total

@njit(fastmath=True, cache=True)
def _stm_jacobian_njit(r: np.ndarray, v: np.ndarray, Bc: float, rho_ref: float, H_km: float, rho_ref_alt_km: float) -> np.ndarray:
    # Central-difference Jacobian step (km)
    eps_r = 1e-4
    eps_v = 1e-4
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
        a_plus = _acceleration_njit(r_plus, v, Bc, rho_ref, H_km, rho_ref_alt_km)
        a_minus = _acceleration_njit(r_minus, v, Bc, rho_ref, H_km, rho_ref_alt_km)
        J[3:, i] = (a_plus - a_minus) / (2.0 * eps_r)
        
    # Lower-right: da/dv (mainly for drag; captures co-rotation effect too)
    for i in range(3):
        v_plus = v.copy()
        v_minus = v.copy()
        v_plus[i] += eps_v
        v_minus[i] -= eps_v
        a_plus = _acceleration_njit(r, v_plus, Bc, rho_ref, H_km, rho_ref_alt_km)
        a_minus = _acceleration_njit(r, v_minus, Bc, rho_ref, H_km, rho_ref_alt_km)
        J[3:, 3+i] = (a_plus - a_minus) / (2.0 * eps_v)
        
    return J

@njit(fastmath=True, cache=True)
def _stm_derivatives_njit(t: float, y: np.ndarray, Bc: float, rho_ref: float, H_km: float, rho_ref_alt_km: float) -> np.ndarray:
    r = y[:3]
    v = y[3:6]
    Phi = y[6:].reshape((6, 6))
    
    a_total = _acceleration_njit(r, v, Bc, rho_ref, H_km, rho_ref_alt_km)
    A = _stm_jacobian_njit(r, v, Bc, rho_ref, H_km, rho_ref_alt_km)
    
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
    from astra.constants import EARTH_EQUATORIAL_RADIUS_KM as _Re, DRAG_MIN_ALTITUDE_KM as _min_alt
    r0_mag = float(np.linalg.norm(r0_km))
    rho_ref_alt_km = max(r0_mag - _Re, _min_alt)

    if drag_config:
        Bc = float(drag_config.cd * drag_config.area_m2 / drag_config.mass_kg)
        from astra.constants import DRAG_REF_DENSITY_KG_M3, DRAG_SCALE_HEIGHT_KM
        # Initialize density at actual orbit altitude for physical accuracy (DEF-001)
        try:
            from astra.data_pipeline import atmospheric_density_empirical, get_space_weather
            f107_obs, f107_adj, ap = get_space_weather(t_jd0)
            rho_ref = atmospheric_density_empirical(rho_ref_alt_km, f107_obs, f107_adj, ap)
        except Exception:
            rho_ref = DRAG_REF_DENSITY_KG_M3  # fallback
            rho_ref_alt_km = 400.0
        H_km = DRAG_SCALE_HEIGHT_KM

    sol = solve_ivp(
        _stm_derivatives_njit,
        t_span=(0.0, duration_s),
        y0=y0,
        args=(Bc, rho_ref, H_km, rho_ref_alt_km),
        method='DOP853',
        rtol=1e-10,
        atol=1e-12,
    )

    if not sol.success:
        # STRICT_MODE: do not silently degrade Pc inputs
        from astra import config
        if config.ASTRA_STRICT_MODE:
            from astra.errors import PropagationError
            raise PropagationError(
                f"[ASTRA STRICT] STM covariance propagation failed: {sol.message}. "
                "Returning initial covariance would silently degrade accuracy. "
                "Set ASTRA_STRICT_MODE=False to allow fallback to initial covariance.",
                norad_id="COVARIANCE_STM"
            )
        import logging
        logging.getLogger(__name__).warning(
            f"STM integration failed ({sol.message}) — returning initial covariance as conservative fallback."
        )
        return cov0_6x6  # Fallback to initial

    Phi_final = sol.y[6:, -1].reshape(6, 6)
    return Phi_final @ cov0_6x6 @ Phi_final.T
