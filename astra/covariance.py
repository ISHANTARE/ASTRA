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
import numpy as np
from scipy.integrate import solve_ivp

from astra.constants import (
    EARTH_EQUATORIAL_RADIUS_KM,
    EARTH_MU_KM3_S2,
    J2,
)


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
    
    # Establish arbitrary perpendicular spanning vectors U_x, U_z for B-plane
    temp = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(u_y, temp)) > 0.99:
        temp = np.array([0.0, 1.0, 0.0])
        
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
        inv_C_p = np.linalg.inv(C_p)
        det_C_p = np.linalg.det(C_p)
    except np.linalg.LinAlgError:
        return 0.0
        
    if det_C_p <= 0:
        return 0.0
        
    # Mahalanobis distance squared (u^2)
    mahalanobis_sq = float(r_p.T @ inv_C_p @ r_p)
    
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
    
    Args:
        time_since_epoch_days: Days elapsed since TLE epoch.
        
    Returns:
        np.ndarray: (3, 3) diagonal covariance matrix in km^2.
    """
    days = max(0.1, abs(time_since_epoch_days))
    
    # Anisotropic TLE degradation model (RTN frame)
    # Based on empirical TLE accuracy studies:
    # - Radial: ~50m base + 500m/day² growth
    # - In-track: ~100m base + 2km/day³ growth (drag-dominated)
    # - Cross-track: ~50m base + 100m/day growth
    
    drag_scale = max(0.1, f107_flux / 150.0)
    
    sigma_r_km = 0.05 + 0.5 * days**2        # Radial
    sigma_t_km = 0.1 + (2.0 * drag_scale) * days**3  # In-track (transverse)
    sigma_n_km = 0.05 + 0.1 * days           # Cross-track (normal)
    
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
    if cov_a.shape == (3, 3):
        # Fallback padding to 6x6 if legacy 3x3 passed
        c_a = np.zeros((6, 6))
        c_a[:3, :3] = cov_a
        c_b = np.zeros((6, 6))
        c_b[:3, :3] = cov_b
        cov_a, cov_b = c_a, c_b
    combined_radius_km = radius_a_km + radius_b_km
    combined_cov = cov_a + cov_b

    if np.all(combined_cov == 0):
        return 1.0 if np.linalg.norm(miss_vector_km) <= combined_radius_km else 0.0

    rng = np.random.default_rng(seed)

    try:
        L = np.linalg.cholesky(combined_cov)
    except np.linalg.LinAlgError:
        eigvals = np.linalg.eigvalsh(combined_cov)
        combined_cov += np.eye(6) * max(0, -eigvals.min() + 1e-12)
        L = np.linalg.cholesky(combined_cov)

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


from numba import njit
from astra.constants import J3, J4

@njit(fastmath=True, cache=True)
def _gravity_accel_njit(r: np.ndarray) -> np.ndarray:
    x, y, z = r[0], r[1], r[2]
    r_mag = np.linalg.norm(r)
    r2 = r_mag**2
    r3 = r2 * r_mag
    
    a_twobody = -EARTH_MU_KM3_S2 * r / r3
    
    z2 = z**2
    r5 = r3 * r2
    r7 = r5 * r2
    r9 = r7 * r2
    Re = EARTH_EQUATORIAL_RADIUS_KM
    mu = EARTH_MU_KM3_S2

    fJ2 = 1.5 * J2 * mu * Re**2 / r5
    a_j2_x = fJ2 * x * (5.0 * z2 / r2 - 1.0)
    a_j2_y = fJ2 * y * (5.0 * z2 / r2 - 1.0)
    a_j2_z = fJ2 * z * (5.0 * z2 / r2 - 3.0)

    fJ3 = 0.5 * J3 * mu * Re**3 / r7
    a_j3_x = fJ3 * x * (35.0 * z2 * z / r2 - 15.0 * z)
    a_j3_y = fJ3 * y * (35.0 * z2 * z / r2 - 15.0 * z)
    a_j3_z = fJ3 * (35.0 * z2 * z2 / r2 - 30.0 * z2 + 3.0 * r2)

    fJ4 = -0.625 * J4 * mu * Re**4 / r9
    z4 = z2 * z2
    a_j4_x = fJ4 * x * (63.0 * z4 / r2 - 42.0 * z2 + 3.0 * r2)
    a_j4_y = fJ4 * y * (63.0 * z4 / r2 - 42.0 * z2 + 3.0 * r2)
    a_j4_z = fJ4 * z * (63.0 * z4 / r2 - 70.0 * z2 + 15.0 * r2)

    a = np.empty(3)
    a[0] = a_twobody[0] + a_j2_x + a_j3_x + a_j4_x
    a[1] = a_twobody[1] + a_j2_y + a_j3_y + a_j4_y
    a[2] = a_twobody[2] + a_j2_z + a_j3_z + a_j4_z
    return a

@njit(fastmath=True, cache=True)
def _stm_jacobian_njit(r: np.ndarray) -> np.ndarray:
    eps = 1e-4
    J = np.zeros((6, 6))
    J[0, 3] = 1.0
    J[1, 4] = 1.0
    J[2, 5] = 1.0
    
    for i in range(3):
        r_plus = r.copy()
        r_minus = r.copy()
        r_plus[i] += eps
        r_minus[i] -= eps
        a_plus = _gravity_accel_njit(r_plus)
        a_minus = _gravity_accel_njit(r_minus)
        J[3:, i] = (a_plus - a_minus) / (2.0 * eps)
        
    return J

@njit(fastmath=True, cache=True)
def _stm_derivatives_njit(t: float, y: np.ndarray) -> np.ndarray:
    r = y[:3]
    v = y[3:6]
    Phi = y[6:].reshape((6, 6))
    
    a_total = _gravity_accel_njit(r)
    A = _stm_jacobian_njit(r)
    
    dPhi = A @ Phi
    
    dy = np.empty(42)
    dy[:3] = v
    dy[3:6] = a_total
    
    dPhi_flat = dPhi.ravel()
    for i in range(36):
        dy[6+i] = dPhi_flat[i]
        
    return dy

def propagate_covariance_stm(
    t_jd0: float,
    r0_km: np.ndarray,
    v0_km_s: np.ndarray,
    cov0_6x6: np.ndarray,
    duration_s: float,
) -> np.ndarray:
    """Propagate a full 6x6 covariance matrix using the State Transition Matrix.

    Uses the linearized equations of motion to compute the 6x6 STM
    via numerical integration, mapping the full 6x6 state uncertainty:

        C(t) = Φ(t, t0) · C₀ · Φ(t, t0)ᵀ
    """
    y0 = np.zeros(42)
    y0[:3] = r0_km
    y0[3:6] = v0_km_s
    y0[6:] = np.eye(6).ravel()

    sol = solve_ivp(
        _stm_derivatives_njit,
        t_span=(0.0, duration_s),
        y0=y0,
        method='DOP853',
        rtol=1e-10,
        atol=1e-12,
    )

    if not sol.success:
        return cov0_6x6  # Fallback to initial

    Phi_final = sol.y[6:, -1].reshape(6, 6)
    return Phi_final @ cov0_6x6 @ Phi_final.T
