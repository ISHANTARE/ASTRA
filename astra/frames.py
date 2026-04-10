# astra/frames.py
"""ASTRA Core Coordinate Frame Transformations.

High-performance Numba-accelerated kernels for orbital frame rotations (VNB, RTN).
This module acts as a base for both propagator and maneuver logic,
resolving circular dependencies.
"""
import numpy as np

# DEF-003: Graceful Numba fallback — mirrors the pattern in propagator.py.
# Without this, importing frames.py would hard-crash in environments where
# Numba is not installed, breaking the propagator's own graceful fallback.
try:
    from numba import njit
    _NUMBA_AVAILABLE = True
except ImportError:
    _NUMBA_AVAILABLE = False
    def njit(*args, **kwargs):  # type: ignore[misc]
        """No-op Numba decorator. Functions run as plain Python."""
        def _decorator(fn):
            return fn
        # Handle both @njit and @njit(fastmath=True, ...) call forms
        return _decorator(args[0]) if (args and callable(args[0])) else _decorator
    import logging as _logging
    _logging.getLogger(__name__).warning(
        "Numba not installed — frame rotation kernels will run in pure-Python mode."
    )

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

def get_eop_correction(times_jd: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fetch Spacebook Earth Orientation Parameters for a set of epochs.

    DEF-006: Batches calls by calendar day rather than per-timestep.
    For a 24-hour propagation at 5-minute steps (288 points), this reduces
    Spacebook API calls from 288 to ~1, with intra-day values taken from
    the representative midday fetch (EOP variation within a day is sub-ms).

    Returns:
        (xp_arcsec, yp_arcsec, dut1_s) arrays matching times_jd shape.
        If Spacebook is disabled or offline, returns arrays of zeros.
    """
    import os
    times_arr = np.atleast_1d(np.asarray(times_jd, dtype=float))
    zeros = np.zeros(len(times_arr))

    if os.environ.get("ASTRA_SPACEBOOK_ENABLED", "true").lower() == "false":
        return (zeros.copy(), zeros.copy(), zeros.copy()) if times_arr.ndim else (0.0, 0.0, 0.0)

    try:
        from astra.spacebook import get_eop_sb

        # --- DEF-006: Group by calendar date, fetch once per unique day ---
        # Floor each JD to midnight of that day (integer part).
        # EOP values change by at most ~0.3 ms/day (dUT1), well below
        # the 1-second timestep of most propagations.
        day_keys = np.floor(times_arr).astype(int)
        unique_days = np.unique(day_keys)

        eop_cache: dict = {}
        for day_jd in unique_days:
            # Query at midday of that calendar date (more representative than midnight)
            eop_cache[int(day_jd)] = get_eop_sb(float(day_jd) + 0.5)

        xps   = np.array([eop_cache[int(dk)][0] for dk in day_keys])
        yps   = np.array([eop_cache[int(dk)][1] for dk in day_keys])
        dut1s = np.array([eop_cache[int(dk)][2] for dk in day_keys])

        # Return scalars when a scalar JD was passed in
        if np.isscalar(times_jd) or (hasattr(times_jd, 'ndim') and times_jd.ndim == 0):
            return float(xps[0]), float(yps[0]), float(dut1s[0])
        return xps, yps, dut1s

    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(
            "EOP fetch failed (%s); defaulting to zero correction.", e
        )
        return (zeros.copy(), zeros.copy(), zeros.copy())


def teme_to_ecef(r_teme: np.ndarray, times_jd: np.ndarray, use_spacebook_eop: bool = True) -> np.ndarray:
    """Convert TEME position vectors to ECEF (ITRS) coordinates.
    
    If `use_spacebook_eop` is True and Spacebook is enabled, precise COMSPOC EOP metrics
    are fetched and mathematically integrated into the GCRS-ITRS rotation differential.
    Otherwise, it natively uses Skyfield's fallback IERS tables.
    
    Args:
        r_teme: (N, 3) matrix of TEME Cartesian coordinates (km)
        times_jd: (N,) array of Julian Dates
        use_spacebook_eop: Boolean toggle to inject precise parameters.
        
    Returns:
        (N, 3) matrix of ECEF (ITRS) coordinates.
    """
    from astra import data_pipeline as _dp
    from skyfield.sgp4lib import TEME
    from skyfield.framelib import itrs
    
    _dp._ensure_skyfield()
    ts = _dp._skyfield_ts
    
    # 1. TEME -> GCRS (Precession / Nutation is native and identical to both)
    t = ts.tt_jd(times_jd)
    
    R_teme_gcrs = np.transpose(TEME.rotation_at(t), axes=(1, 0, 2)) if hasattr(t, 'shape') else np.transpose(TEME.rotation_at(t))
    r_gcrs = np.einsum('ij...,j...->i...', R_teme_gcrs, r_teme.T).T if hasattr(t, 'shape') else R_teme_gcrs.dot(r_teme.T).T
    
    # 2. Base GCRS -> ITRS using Skyfield cached baseline
    R_gcrs_itrs = np.transpose(itrs.rotation_at(t), axes=(1, 0, 2)) if hasattr(t, 'shape') else np.transpose(itrs.rotation_at(t))
    r_itrs = np.einsum('ij...,j...->i...', R_gcrs_itrs, r_gcrs.T).T if hasattr(t, 'shape') else R_gcrs_itrs.dot(r_gcrs.T).T
    
    import os
    spacebook_enabled = os.environ.get("ASTRA_SPACEBOOK_ENABLED", "true").lower() != "false"
    
    if not use_spacebook_eop or not spacebook_enabled:
        return r_itrs
        
    # 3. Apply Spacebook EOP Differential (Correction Matrix)
    xp_sb, yp_sb, dut1_sb = get_eop_correction(times_jd)
    
    # Only correct if we got actual data (non-zero combined norms)
    if np.sum(np.abs(xp_sb)) == 0 and np.sum(np.abs(yp_sb)) == 0 and np.sum(np.abs(dut1_sb)) == 0:
        return r_itrs
        
    # Skyfield base parameters
    dut1_skyfield = t.dut1
    xp_sky, yp_sky, _ = t.polar_motion_angles()
    
    # Deltas
    d_dut1 = dut1_sb - dut1_skyfield
    d_xp_rc = (xp_sb - xp_sky) * (np.pi / (180.0 * 3600.0))  # arcsec to rad
    d_yp_rc = (yp_sb - yp_sky) * (np.pi / (180.0 * 3600.0))
    
    # Earth rotation phase delta
    # Earth angular velocity = ~ 7.292115e-5 rad/s
    omega_earth = 7.292115146706979e-5
    d_theta = d_dut1 * omega_earth
    
    # Apply correcting rotational approximations
    # R_diff = R_y(d_xp) @ R_x(d_yp) @ R_z(d_theta)
    
    cos_t = np.cos(d_theta)
    sin_t = np.sin(d_theta)
    
    # Pre-allocate array for corrected ITRS vectors
    r_itrs_corrected = np.empty_like(r_itrs)
    
    # Vectorized application
    if r_itrs.ndim == 2:
        for i in range(len(r_itrs)):
            x, y, z = r_itrs[i]
            
            # R_z Earth rotation adjustment
            x_rot = x * cos_t[i] + y * sin_t[i]
            y_rot = -x * sin_t[i] + y * cos_t[i]
            z_rot = z
            
            # R_x, R_y Polar motion tilt adjustment (small-angle approximation)
            # R_y(dx) @ R_x(dy) = [[1, 0, dx], [0, 1, -dy], [-dx, dy, 1]]
            dx = d_xp_rc[i]
            dy = d_yp_rc[i]
            
            x_f = x_rot + dx * z_rot
            y_f = y_rot - dy * z_rot
            z_f = -dx * x_rot + dy * y_rot + z_rot
            
            r_itrs_corrected[i] = [x_f, y_f, z_f]
    else:
        x, y, z = r_itrs
        x_rot = x * cos_t + y * sin_t
        y_rot = -x * sin_t + y * cos_t
        z_rot = z
        
        dx = d_xp_rc
        dy = d_yp_rc
        
        x_f = x_rot + dx * z_rot
        y_f = y_rot - dy * z_rot
        z_f = -dx * x_rot + dy * y_rot + z_rot
        
        r_itrs_corrected = np.array([x_f, y_f, z_f])
        
    return r_itrs_corrected

@njit(fastmath=True, cache=True)
def ecef_to_geodetic_wgs84(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert ECEF Cartesian coordinates (km) to WGS84 Geodetic coordinates.
    
    Args:
        x, y, z: 1D arrays of cartesian coordinates in km.
        
    Returns:
        tuple of (latitude_deg, longitude_deg, altitude_km).
    """
    a = 6378.137  # WGS84 Semi-major axis (km)
    f = 1.0 / 298.257223563  # Flattening
    b = a * (1.0 - f)
    e2 = 1.0 - (b**2 / a**2)
    ep2 = (a**2 - b**2) / b**2
    
    p = np.sqrt(x**2 + y**2)
    th = np.arctan2(a * z, b * p)
    
    lon = np.arctan2(y, x)
    lat = np.arctan2(z + ep2 * b * np.sin(th)**3, p - e2 * a * np.cos(th)**3)
    
    N = a / np.sqrt(1.0 - e2 * np.sin(lat)**2)
    alt = p / np.cos(lat) - N
    
    # Radians to degrees
    lat_deg = lat * (180.0 / np.pi)
    lon_deg = lon * (180.0 / np.pi)
    
    return lat_deg, lon_deg, alt

