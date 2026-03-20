"""ASTRA Core Covariance and Uncertainty Modeling.

Calculates Mahalanobis distance and Probability of Collision (Pc) 
by projecting 3D positional covariances onto the 2D encounter plane.
"""
from __future__ import annotations

import math
import numpy as np


def compute_collision_probability(
    miss_vector_km: np.ndarray, 
    rel_vel_km_s: np.ndarray, 
    cov_a: np.ndarray, 
    cov_b: np.ndarray, 
    combined_radius_km: float = 0.010,
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
    # Extremely accurate for small combined radius compared to covariance volume
    p_c = math.exp(-0.5 * mahalanobis_sq) * area / (2.0 * math.pi * math.sqrt(det_C_p))
    
    return float(np.clip(p_c, 0.0, 1.0))


def estimate_covariance(time_since_epoch_days: float) -> np.ndarray:
    """Generates an estimated positional covariance matrix (3x3).
    
    In-track covariance expands quadratically/cubic over time for TLEs.
    This creates a synthetic covariance matrix approximating SGP4 degradation, 
    oriented in RIC (Radial, In-track, Cross-track) and requiring subsequent RTN rotation.
    For MVP ASTRA Core, we simply provide an isotropic estimate expanding over time.
    
    Args:
        time_since_epoch_days: Days elapsed since TLE epoch.
        
    Returns:
        np.ndarray: (3, 3) isotropic covariance matrix in km^2.
    """
    days = max(0.1, abs(time_since_epoch_days))
    
    # Heuristic TLE degradation: Radial ~1km/day, In-track ~5km/day, Cross ~1km/day
    # Simplified here to isotropic expansion for the unconstrained MVP module
    sigma_km = 1.0 + (5.0 * days)
    variance_km2 = sigma_km ** 2
    
    return np.eye(3) * variance_km2
