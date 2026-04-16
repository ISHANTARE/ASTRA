from typing import Any

# astra/frames.py
"""ASTRA Core Coordinate Frame Transformations.

High-performance Numba-accelerated kernels for orbital frame rotations (VNB, RTN).
This module acts as a base for both propagator and maneuver logic,
resolving circular dependencies.
"""
import numpy as np  # noqa: E402

# DEF-003: Graceful Numba fallback — mirrors the pattern in propagator.py.
# Without this, importing frames.py would hard-crash in environments where
# Numba is not installed, breaking the propagator's own graceful fallback.
from astra._numba_compat import njit  # noqa: E402


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
    return mat  # type: ignore[no-any-return]


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
    return mat  # type: ignore[no-any-return]


def get_eop_correction(
    times_jd: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | tuple[float, float, float]:
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
        return (zeros.copy(), zeros.copy(), zeros.copy()) if times_arr.ndim else (0.0, 0.0, 0.0)  # type: ignore[no-any-return]

    try:
        from astra.spacebook import get_eop_sb

        # --- DEF-006: Group by calendar date, fetch once per unique day ---
        # Floor each JD to midnight of that day (integer part).
        # EOP values change by at most ~0.3 ms/day (dUT1), well below
        # the 1-second timestep of most propagations.
        day_keys = np.floor(times_arr).astype(int)
        unique_days = np.unique(day_keys)

        eop_cache: dict[int, Any] = {}
        for day_jd in unique_days:
            # Query at midday of that calendar date (more representative than midnight)
            eop_cache[int(day_jd)] = get_eop_sb(float(day_jd) + 0.5)

        xps = np.array([eop_cache[int(dk)][0] for dk in day_keys])
        yps = np.array([eop_cache[int(dk)][1] for dk in day_keys])
        dut1s = np.array([eop_cache[int(dk)][2] for dk in day_keys])

        # Return scalars when a scalar JD was passed in
        if np.isscalar(times_jd) or (hasattr(times_jd, "ndim") and times_jd.ndim == 0):
            return float(xps[0]), float(yps[0]), float(dut1s[0])  # type: ignore[no-any-return]
        return xps, yps, dut1s  # type: ignore[no-any-return]

    except Exception as e:
        import logging

        logging.getLogger(__name__).warning(
            "EOP fetch failed (%s); defaulting to zero correction.", e
        )
        return (zeros.copy(), zeros.copy(), zeros.copy())  # type: ignore[no-any-return]


def teme_to_ecef(
    r_teme: np.ndarray, times_jd: np.ndarray, use_spacebook_eop: bool = True
) -> np.ndarray:
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
    _dp._ensure_skyfield()
    ts = _dp._skyfield_ts
    assert ts is not None, "_ensure_skyfield() failed to initialise Skyfield timescale"
    t = ts.tt_jd(times_jd)

    R_teme_gcrs = (
        np.transpose(TEME.rotation_at(t), axes=(1, 0, 2))
        if hasattr(t, "shape")
        else np.transpose(TEME.rotation_at(t))
    )
    r_gcrs = (
        np.einsum("ij...,j...->i...", R_teme_gcrs, r_teme.T).T
        if hasattr(t, "shape")
        else R_teme_gcrs.dot(r_teme.T).T
    )

    # 2. Base GCRS -> ITRS using Skyfield cached baseline
    R_gcrs_itrs = (
        np.transpose(itrs.rotation_at(t), axes=(1, 0, 2))
        if hasattr(t, "shape")
        else np.transpose(itrs.rotation_at(t))
    )
    r_itrs = (
        np.einsum("ij...,j...->i...", R_gcrs_itrs, r_gcrs.T).T
        if hasattr(t, "shape")
        else R_gcrs_itrs.dot(r_gcrs.T).T
    )

    import os

    spacebook_enabled = (
        os.environ.get("ASTRA_SPACEBOOK_ENABLED", "true").lower() != "false"
    )

    if not use_spacebook_eop or not spacebook_enabled:
        return r_itrs  # type: ignore[no-any-return]

    # 3. Apply Spacebook EOP Differential (Correction Matrix)
    xp_sb, yp_sb, dut1_sb = get_eop_correction(times_jd)

    # Only correct if we got actual data (non-zero combined norms)
    if (
        np.sum(np.abs(xp_sb)) == 0
        and np.sum(np.abs(yp_sb)) == 0
        and np.sum(np.abs(dut1_sb)) == 0
    ):
        return r_itrs  # type: ignore[no-any-return]

    # Skyfield base parameters
    # AUDIT-C-02 Verification: Skyfield's Time.polar_motion_angles() returns
    # (xp, yp, era_rad) where xp and yp are in ARCSECONDS (the IERS Bulletin A
    # storage convention).  This matches the unit of xp_sb / yp_sb from
    # Spacebook, so the delta below is correctly arc-second-valued before the
    # (pi/648000) arcsec→radians conversion factor.  ERA (the third element) is
    # in radians but is not used here.
    dut1_skyfield = t.dut1
    xp_sky, yp_sky, _ = t.polar_motion_angles()  # both xp_sky, yp_sky in arcsec

    # Deltas
    d_dut1 = dut1_sb - dut1_skyfield
    d_xp_rc = (xp_sb - xp_sky) * (np.pi / (180.0 * 3600.0))  # arcsec to rad
    d_yp_rc = (yp_sb - yp_sky) * (np.pi / (180.0 * 3600.0))

    # [HIGH-06 fix] EOP sanity clamp: reject physically implausible deltas that
    # indicate stale, corrupt, or unit-mismatched Spacebook EOP data.
    # Limits: |ΔUT1| ≤ 1.0 s (IERS spec bound), |Δxp|/|Δyp| ≤ 0.01 rad (~2000 arcsec).
    import logging as _frames_log

    _flog = _frames_log.getLogger(__name__)
    # [FIX] Define omega_earth BEFORE it is referenced in _MAX_THETA below.
    # Previously this constant was defined at line ~199 (after the clamp section),
    # causing an UnboundLocalError whenever the EOP correction branch executed.
    omega_earth = 7.292115146706979e-5  # rad/s  — IAU/IERS 2010 Earth rotation rate
    _MAX_DUT1 = 1.0  # seconds — IERS hard limit for UT1-UTC
    _MAX_PM_RAD = 0.01  # radians ≈ 2062 arcsec — well beyond any observed value
    _MAX_THETA = _MAX_DUT1 * omega_earth  # corresponding phase limit

    if np.any(np.abs(d_dut1) > _MAX_DUT1):
        _flog.warning(
            "EOP anomaly: |Δ-DUT1|=%.4f s exceeds %.1f s limit — clamping. "
            "Check Spacebook EOP data freshness.",
            float(np.max(np.abs(d_dut1))),
            _MAX_DUT1,
        )
        d_dut1 = np.clip(d_dut1, -_MAX_DUT1, _MAX_DUT1)

    if np.any(np.abs(d_xp_rc) > _MAX_PM_RAD):
        _flog.warning(
            "EOP anomaly: |Δxp|=%.4e rad exceeds %.4e rad limit — clamping.",
            float(np.max(np.abs(d_xp_rc))),
            _MAX_PM_RAD,
        )
        d_xp_rc = np.clip(d_xp_rc, -_MAX_PM_RAD, _MAX_PM_RAD)

    if np.any(np.abs(d_yp_rc) > _MAX_PM_RAD):
        _flog.warning(
            "EOP anomaly: |Δyp|=%.4e rad exceeds %.4e rad limit — clamping.",
            float(np.max(np.abs(d_yp_rc))),
            _MAX_PM_RAD,
        )
        d_yp_rc = np.clip(d_yp_rc, -_MAX_PM_RAD, _MAX_PM_RAD)

    # Earth rotation phase delta — omega_earth already defined above.
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

    return r_itrs_corrected  # type: ignore[no-any-return]


@njit(fastmath=True, cache=True)
def ecef_to_geodetic_wgs84(
    x: np.ndarray, y: np.ndarray, z: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert ECEF Cartesian coordinates (km) to WGS84 Geodetic coordinates.

    Uses the Bowring (1976) single-pass iterative formula, which achieves
    ~0.1 mm accuracy at sea level and better at orbital altitudes.

    COORD-01 Fix: Handles the polar singularity (cos(lat)→0 at |lat|≥80°) by
    using the z-based altitude formula for high latitudes:
        alt = |z|/sin(lat) - N*(1-e²)
    instead of the equatorial formula (alt = p/cos(lat) - N) which produces
    infinite values when the satellite passes directly over a polar station.

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
    lat = np.arctan2(z + ep2 * b * np.sin(th) ** 3, p - e2 * a * np.cos(th) ** 3)

    N = a / np.sqrt(1.0 - e2 * np.sin(lat) ** 2)

    # COORD-01: Use z-based altitude formula at high latitudes where cos(lat)→0.
    # Threshold |lat| > 80° (~1.396 rad) matches the standard geodetic convention
    # used by IERS, Bowring, and Vermeille altitude calculators.
    lat_abs = np.abs(lat)
    polar_mask = lat_abs > (80.0 * np.pi / 180.0)

    # AUDIT-A-08 Fix: Guard the equatorial branch against cos(lat)→0 at poles.
    # Numba with fastmath=True evaluates BOTH branches of np.where before
    # selecting, so p/cos(lat) would produce inf at the poles even though the
    # polar_mask would select the other branch.  An inf intermediate can poison
    # downstream computations under fastmath associativity relaxation.
    cos_lat = np.cos(lat)
    safe_cos = np.where(np.abs(cos_lat) < 1e-10, 1e-10, cos_lat)
    alt_equatorial = p / safe_cos - N

    # Polar formula (safe for |lat| > 0, avoids 1/cos(lat) singularity)
    sin_lat = np.sin(lat)
    # Guard against |lat|=0 in the polar branch (shouldn't happen when polar_mask is True,
    # but Numba requires scalar-safe expressions)
    safe_sin = np.where(np.abs(sin_lat) < 1e-15, 1e-15, sin_lat)
    alt_polar = np.abs(z) / np.abs(safe_sin) - N * (1.0 - e2)

    alt = np.where(polar_mask, alt_polar, alt_equatorial)

    # Radians to degrees
    lat_deg = lat * (180.0 / np.pi)
    lon_deg = lon * (180.0 / np.pi)

    return lat_deg, lon_deg, alt  # type: ignore[no-any-return]
