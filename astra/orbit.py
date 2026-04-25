# astra/orbit.py
"""ASTRA Core orbit propagation module.

Responsible for SGP4-based orbit propagation. Generates position and velocity
vectors at specified times. Uses `sgp4` for actual propagation and `skyfield`
for coordinate frame conversions. No manual orbital equations allowed.

Supports both legacy ``SatelliteTLE`` objects (via ``Satrec.twoline2rv``)
and modern ``SatelliteOMM`` objects (via ``Satrec.sgp4init``). Use the
``_build_satrec()`` helper which dispatches automatically based on type.
"""

from __future__ import annotations

import numpy as np
from typing import Generator
from sgp4.api import Satrec, WGS84, SatrecArray

from astra.errors import AstraError
from astra.models import (
    OrbitalState,
    SatelliteTLE,
    SatelliteOMM,
    SatelliteState,
    TrajectoryMap,
    VelocityMap,
)
from astra.log import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Internal Satrec Factory
# ---------------------------------------------------------------------------


def _build_satrec(satellite: SatelliteState) -> Satrec:
    """Build a ``Satrec`` SGP4 object from either a TLE or an OMM source.

    This is the polymorphic dispatch point for the physics engine.
    All propagation functions route through here so the math core
    never needs to inspect the format of incoming data directly.

    Known Limitations:
        (None. UT1-UTC is now applied automatically.)

    Args:
        satellite: Either a ``SatelliteTLE`` or a ``SatelliteOMM`` instance.

    Returns:
        A configured ``Satrec`` object ready for ``sgp4()`` calls.
    """
    if isinstance(satellite, SatelliteTLE):
        # Legacy path: fast C++ string parsing.
        return Satrec.twoline2rv(satellite.line1, satellite.line2)

    if isinstance(satellite, SatelliteOMM):
        # Modern path: explicit sgp4init with pre-converted radian fields.
        satrec = Satrec()
        # epoch days since 1949-12-31 00:00 UT (sgp4 epoch reference)
        # JD of 1949-12-31 00:00 UT = 2433281.5
        epoch_days = satellite.epoch_jd - 2433281.5
        satrec.sgp4init(
            WGS84,
            "i",  # afspc_mode
            int(satellite.norad_id) if satellite.norad_id.isdigit() else 0,
            epoch_days,
            satellite.bstar,
            satellite.mean_motion_dot,
            satellite.mean_motion_ddot,
            satellite.eccentricity,
            satellite.argpo_rad,
            satellite.inclination_rad,
            satellite.mo_rad,
            satellite.mean_motion_rad_min,
            satellite.raan_rad,
        )
        return satrec

    raise AstraError(
        f"Unsupported satellite type: {type(satellite).__name__}. "
        "Expected SatelliteTLE or SatelliteOMM."
    )


def propagate_orbit(
    satellite: SatelliteState, epoch_jd: float, t_since_minutes: float
) -> OrbitalState:
    """Propagate a single satellite to a single point in time using SGP4.

    Data formats: ✓ SatelliteTLE  ✓ SatelliteOMM

    Args:
        satellite: Parsed SatelliteTLE or SatelliteOMM object to propagate.
        epoch_jd: Reference epoch as Julian Date (typically satellite.epoch_jd).
        t_since_minutes: Minutes elapsed since epoch.

    Returns:
        OrbitalState containing position (km) and velocity (km/s).

    Frame:
        TEME (True Equator, Mean Equinox) — raw SGP4 output frame.
        Do NOT feed directly into ECEF-expecting functions without conversion.
    """
    satrec = _build_satrec(satellite)

    t_jd = epoch_jd + (t_since_minutes / 1440.0)

    # Verify epoch staleness (SE-J)
    from astra.tle import check_tle_staleness

    check_tle_staleness(satellite, t_jd)

    try:
        from astra.data_pipeline import get_ut1_utc_correction

        ut1_utc_s = get_ut1_utc_correction(t_jd)
        t_jd_ut1 = t_jd + ut1_utc_s / 86400.0
    except Exception as exc:
        from astra import config

        if config.ASTRA_STRICT_MODE:
            from astra.errors import EphemerisError

            raise EphemerisError(f"Failed to fetch UT1-UTC correction: {exc}") from exc
        logger.warning(
            "UT1-UTC correction unavailable for NORAD %s at JD %.8f — "
            "falling back to UTC as UT1. This silently ignores the UT1-UTC "
            "offset (current value ≈ ±1 s), introducing up to ~400 m of "
            "along-track position error at LEO velocities. "
            "Populate the EOP cache via astra.data_pipeline.load_eop_data() "
            "or set ASTRA_STRICT_MODE=True to detect this in production. (%r)",
            satellite.norad_id,
            t_jd,
            exc,
        )
        t_jd_ut1 = t_jd  # fallback: UTC used as UT1 (sub-second accuracy lost)

    fraction = 0.0

    error_code, position, velocity = satrec.sgp4(t_jd_ut1, fraction)

    return OrbitalState(
        norad_id=satellite.norad_id,
        t_jd=t_jd,
        position_km=np.array(position, dtype=np.float64),
        velocity_km_s=np.array(velocity, dtype=np.float64),
        error_code=error_code,
    )


def propagate_many(
    satellites: list[SatelliteState], times_jd: np.ndarray
) -> tuple[TrajectoryMap, "VelocityMap"]:
    """Vectorized batch propagation of multiple satellites across a time array.

    Data formats: ✓ SatelliteTLE  ✓ SatelliteOMM

    Args:
        satellites: List of SatelliteState objects to propagate.
        times_jd: 1D NumPy array of T absolute Julian Dates. (Shape: (T,), dtype: float64)

    Returns:
        Tuple ``(traj_map, vel_map)``. Each value is a dict keyed by NORAD ID with
        arrays of shape ``(T, 3)`` in TEME: positions in km and velocities in km/s.
        Satellites with propagation errors at any timestep store ``nan`` in that row.

    **Frame:** TEME (True Equator, Mean Equinox), i.e. raw SGP4 output.
    Do not feed positions directly into ECEF-expecting functions without conversion.

    **Note:** Unpack as ``traj_map, vel_map = propagate_many(...)``.
    """
    results: TrajectoryMap = {}
    velocities: dict[str, np.ndarray] = {}
    N = len(satellites)
    T = len(times_jd)
    if N == 0 or T == 0:
        return results, velocities

    logger.info(
        f"Vector-propagating {N} orbits across {T} discrete time steps using SatrecArray..."
    )

    satrecs = [_build_satrec(sat) for sat in satellites]
    satrec_array = SatrecArray(satrecs)

    # 0. Verify epoch staleness for all satellites in batch (SE-J)
    from astra.tle import check_tle_staleness

    for sat in satellites:
        check_tle_staleness(sat, times_jd)

    # 1. Apply UT1-UTC correction (consistent with propagate_orbit)
    try:
        from astra.data_pipeline import get_ut1_utc_correction

        ut1_utc_s = get_ut1_utc_correction(times_jd)
        jd_array = times_jd + ut1_utc_s / 86400.0
    except Exception as exc:
        from astra import config

        if config.ASTRA_STRICT_MODE:
            from astra.errors import EphemerisError

            raise EphemerisError(
                f"Failed to fetch UT1-UTC correction for batch: {exc}"
            ) from exc
        logger.warning(
            "UT1-UTC correction unavailable for batch propagation (%d objects, %d epochs) — "
            "falling back to UTC as UT1. This silently ignores the UT1-UTC offset "
            "(current value ≈ ±1 s), introducing up to ~400 m of along-track position "
            "error at LEO velocities across all propagated states. "
            "Populate the EOP cache via astra.data_pipeline.load_eop_data() "
            "or set ASTRA_STRICT_MODE=True to detect this in production. (%r)",
            N,
            T,
            exc,
        )
        jd_array = times_jd  # fallback: UTC used as UT1

    jd_fraction_array = np.zeros_like(jd_array)

    # 2. Vectorized SGP4 call
    e, r, v = satrec_array.sgp4(jd_array, jd_fraction_array)

    r[e > 0] = np.nan
    v[e > 0] = np.nan

    for i, sat in enumerate(satellites):
        results[sat.norad_id] = r[i]
        velocities[sat.norad_id] = v[i]

    return results, velocities


def propagate_many_generator(
    satellites: list[SatelliteState], times_jd: np.ndarray, chunk_size: int = 1440
) -> Generator[tuple[np.ndarray, TrajectoryMap, "VelocityMap"], None, None]:
    """Memory-efficient batch propagation yielding time-chunked results.

    Data formats: ✓ SatelliteTLE  ✓ SatelliteOMM

    Prevents Out-Of-Memory (OOM) fatal kills during massive all-vs-all STM queries
    by yielding rolling spatial windows.
    """
    N = len(satellites)
    T = len(times_jd)
    if N == 0 or T == 0:
        return

    satrecs = [_build_satrec(sat) for sat in satellites]
    satrec_array = SatrecArray(satrecs)

    for start_idx in range(0, T, chunk_size):
        end_idx = min(start_idx + chunk_size, T)

        jd_chunk = times_jd[start_idx:end_idx]

        # Verify epoch staleness for each chunk (SE-J)
        from astra.tle import check_tle_staleness

        for sat in satellites:
            check_tle_staleness(sat, jd_chunk)

        try:
            from astra.data_pipeline import get_ut1_utc_correction

            # Vectorize UT1-UTC directly across the chunk
            ut1_utc_s = get_ut1_utc_correction(jd_chunk)
            jd_chunk_ut1 = jd_chunk + ut1_utc_s / 86400.0
        except Exception as exc:
            from astra import config

            if config.ASTRA_STRICT_MODE:
                from astra.errors import EphemerisError

                raise EphemerisError(
                    f"Failed to fetch UT1-UTC correction for chunk: {exc}"
                ) from exc
            logger.warning(
                "UT1-UTC correction unavailable for chunk propagation (%d objects, %d epochs); "
                "falling back to UTC propagation in relaxed mode. (%r)",
                N,
                len(jd_chunk),
                exc,
            )
            jd_chunk_ut1 = jd_chunk

        frac_chunk = np.zeros_like(jd_chunk)

        e, r, v = satrec_array.sgp4(jd_chunk_ut1, frac_chunk)
        r[e > 0] = np.nan
        v[e > 0] = np.nan

        chunk_map = {sat.norad_id: r[i] for i, sat in enumerate(satellites)}
        vel_map = {sat.norad_id: v[i] for i, sat in enumerate(satellites)}
        yield jd_chunk, chunk_map, vel_map


def propagate_trajectory(
    satellite: SatelliteState,
    t_start_jd: float,
    t_end_jd: float,
    step_minutes: float = 5.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Propagate a single satellite over a defined time window at a fixed step.

    Data formats: ✓ SatelliteTLE  ✓ SatelliteOMM

    Args:
        satellite: Source SatelliteTLE or SatelliteOMM to propagate.
        t_start_jd: Window start as Julian Date.
        t_end_jd: Window end as Julian Date (must be strictly > t_start_jd).
        step_minutes: Step size in minutes (must be positive; default 5.0).

    Returns:
        Tuple of (time_array, position_array, velocity_array):
            time_array: shape (T,), Julian Dates for each step.
            position_array: shape (T, 3), TEME positions in km.
            velocity_array: shape (T, 3), TEME velocities in km/s.

    Frame:
        TEME (True Equator, Mean Equinox) — raw SGP4 output frame.
        Do NOT feed directly into ECEF-expecting functions without conversion.

    Raises:
        ValueError: If t_end_jd <= t_start_jd or step_minutes <= 0.
    """
    if t_end_jd <= t_start_jd:
        raise ValueError(
            f"t_end_jd ({t_end_jd}) must be strictly greater than t_start_jd ({t_start_jd}). "
            "Check argument order."
        )
    if step_minutes <= 0:
        raise ValueError(f"step_minutes must be positive, got {step_minutes}.")

    start_offset_mins = (t_start_jd - satellite.epoch_jd) * 1440.0
    end_offset_mins = (t_end_jd - satellite.epoch_jd) * 1440.0

    # Adding a tiny epsilon to end_offset_mins to ensure inclusive bound if it aligns exactly
    time_steps = np.arange(start_offset_mins, end_offset_mins + 1e-9, step_minutes)
    times_jd = satellite.epoch_jd + (time_steps / 1440.0)

    trajectory_map, vel_map = propagate_many([satellite], times_jd)

    positions = trajectory_map[satellite.norad_id]
    velocities = vel_map[satellite.norad_id]

    return times_jd, positions, velocities


def ground_track(
    positions_teme: np.ndarray, times_jd: np.ndarray
) -> list[tuple[float, float, float]]:
    """Convert TEME Cartesian positions into geodetic coordinates for ground track.

    Args:
        positions_teme: TEME position array from propagation. Shape: (T, 3)
        times_jd: Corresponding Julian Date array. Shape: (T,)

    Returns:
        List of (latitude_deg, longitude_deg, altitude_km) tuples, length T.

    Frame:
        Input must be in TEME (True Equator, Mean Equinox) frame — the native
        SGP4 output. This function internally converts TEME → GCRS → WGS84.
    """
    if len(times_jd) == 0:
        return []

    from astra.frames import teme_to_ecef, ecef_to_geodetic_wgs84

    r_itrs_km = teme_to_ecef(positions_teme, times_jd, use_spacebook_eop=True)

    x, y, z = r_itrs_km.T
    lat_deg, lon_deg, alt_km = ecef_to_geodetic_wgs84(x, y, z)

    if np.isscalar(lat_deg) or getattr(lat_deg, "ndim", 0) == 0:
        return [(float(lat_deg), float(lon_deg), float(alt_km))]

    return list(zip(lat_deg.tolist(), lon_deg.tolist(), alt_km.tolist()))
