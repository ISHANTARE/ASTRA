# astra/frames.py
"""ASTRA Core Coordinate Frame Transformations.

High-performance Numba-accelerated kernels for orbital frame rotations (VNB, RTN).
This module acts as a base for both propagator and maneuver logic,
resolving circular dependencies.
"""
import numpy as np
from numba import njit

@njit(fastmath=True, cache=True)
def _build_vnb_matrix_njit(pos: np.ndarray, vel: np.ndarray) -> np.ndarray:
    """Build the 3x3 rotation matrix from ECI to VNB frame.

    V (Velocity): Along the velocity vector.
    N (Normal): Along the angular momentum vector (R x V).
    B (Binormal): Completes the right-handed triad (V x N).
    """
    v_unit = vel / np.linalg.norm(vel)
    h = np.cross(pos, vel)
    n_unit = h / np.linalg.norm(h)
    b_unit = np.cross(v_unit, n_unit)

    # Matrix for ECI -> VNB: rows are basis vectors
    mat = np.zeros((3, 3))
    mat[0, 0], mat[0, 1], mat[0, 2] = v_unit[0], v_unit[1], v_unit[2]
    mat[1, 0], mat[1, 1], mat[1, 2] = n_unit[0], n_unit[1], n_unit[2]
    mat[2, 0], mat[2, 1], mat[2, 2] = b_unit[0], b_unit[1], b_unit[2]
    return mat

@njit(fastmath=True, cache=True)
def _build_rtn_matrix_njit(pos: np.ndarray, vel: np.ndarray) -> np.ndarray:
    """Build the 3x3 rotation matrix from ECI to RTN (RIC) frame.

    R (Radial): Along the geocentric position vector.
    T (Transverse/In-track): In the orbital plane, orthogonal to R.
    N (Normal): Along the angular momentum vector (R x V).
    """
    r_unit = pos / np.linalg.norm(pos)
    h = np.cross(pos, vel)
    n_unit = h / np.linalg.norm(h)
    t_unit = np.cross(n_unit, r_unit)

    # Matrix for ECI -> RTN: rows are basis vectors
    mat = np.zeros((3, 3))
    mat[0, 0], mat[0, 1], mat[0, 2] = r_unit[0], r_unit[1], r_unit[2]
    mat[1, 0], mat[1, 1], mat[1, 2] = t_unit[0], t_unit[1], t_unit[2]
    mat[2, 0], mat[2, 1], mat[2, 2] = n_unit[0], n_unit[1], n_unit[2]
    return mat
