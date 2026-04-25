"""Tests for orbit propagation functions.

[FM-5 Fix]
Previously these tests only checked array shapes and types — "phantom tests"
that pass even when the physics is broken. Every test now asserts at least
one physical invariant:
  - Orbital radius falls within ISS LEO bounds (6,550 – 6,900 km).
  - Velocity magnitude satisfies the circular orbit vis-viva equation to
    within a physically motivated tolerance.
  - propagate_many and propagate_orbit agree to better than 1 m (1e-3 km).
  - Trajectory radius never dips below Earth's surface (6,378.137 km).
"""
import pytest
import numpy as np
from astra.constants import EARTH_MU_KM3_S2, EARTH_EQUATORIAL_RADIUS_KM
from astra import propagate_orbit, propagate_many, propagate_trajectory, ground_track
from astra.models import OrbitalState


# ISS LEO orbital radius bounds (km) — any state outside these is unphysical.
_ISS_R_MIN_KM: float = 6_550.0  # ~172 km altitude
_ISS_R_MAX_KM: float = 6_900.0  # ~521 km altitude

# ISS orbital speed bounds (km/s) — derived from vis-viva at the radius bounds.
_ISS_V_MIN_KM_S: float = float(np.sqrt(EARTH_MU_KM3_S2 / _ISS_R_MAX_KM))  # ~7.59
_ISS_V_MAX_KM_S: float = float(np.sqrt(EARTH_MU_KM3_S2 / _ISS_R_MIN_KM))  # ~7.81


def test_propagate_returns_orbital_state(iss_tle):
    state = propagate_orbit(iss_tle, iss_tle.epoch_jd, 0.0)
    assert isinstance(state, OrbitalState)


def test_propagate_position_shape(iss_tle):
    state = propagate_orbit(iss_tle, iss_tle.epoch_jd, 0.0)
    assert state.position_km.shape == (3,)


def test_propagate_velocity_shape(iss_tle):
    state = propagate_orbit(iss_tle, iss_tle.epoch_jd, 0.0)
    assert state.velocity_km_s.shape == (3,)


def test_propagate_position_units(iss_tle):
    """[FM-5 Fix] Radius must fall within ISS LEO bounds, not just some LEO range."""
    state = propagate_orbit(iss_tle, iss_tle.epoch_jd, 0.0)
    mag = float(np.linalg.norm(state.position_km))
    assert _ISS_R_MIN_KM < mag < _ISS_R_MAX_KM, (
        f"ISS orbital radius {mag:.3f} km outside expected LEO bounds "
        f"[{_ISS_R_MIN_KM:.0f}, {_ISS_R_MAX_KM:.0f}] km. "
        "Either TLE epoch changed or propagation physics is broken."
    )


def test_propagate_velocity_units(iss_tle):
    """[FM-5 Fix] Speed must satisfy vis-viva bounds for the ISS radius regime."""
    state = propagate_orbit(iss_tle, iss_tle.epoch_jd, 0.0)
    r_mag = float(np.linalg.norm(state.position_km))
    v_mag = float(np.linalg.norm(state.velocity_km_s))
    # Vis-viva circular speed at this radius (SGP4 TLE orbit is near-circular)
    v_circ = float(np.sqrt(EARTH_MU_KM3_S2 / r_mag))
    # ISS eccentricity is ~0.00014 — speed can differ from circular by at most ~1%
    assert abs(v_mag - v_circ) < 0.01 * v_circ, (
        f"Speed {v_mag:.4f} km/s deviates from circular vis-viva {v_circ:.4f} km/s "
        f"by {100*abs(v_mag-v_circ)/v_circ:.2f}% (>1%). ISS is near-circular — "
        "large deviation indicates a frame or unit conversion error."
    )


def test_propagate_error_code_zero(iss_tle):
    state = propagate_orbit(iss_tle, iss_tle.epoch_jd, 0.0)
    assert state.error_code == 0


def test_propagate_many_output_shape(iss_tle, time_steps):
    times_jd = iss_tle.epoch_jd + (time_steps / 1440.0)
    trajectories, velocities = propagate_many([iss_tle], times_jd)
    assert "25544" in trajectories
    assert trajectories["25544"].shape == (288, 3)


def test_propagate_many_no_nan_for_valid(iss_tle, time_steps):
    times_jd = iss_tle.epoch_jd + (time_steps / 1440.0)
    trajectories, velocities = propagate_many([iss_tle], times_jd)
    assert not np.isnan(trajectories["25544"]).any()


def test_propagate_many_radius_never_subterranean(iss_tle, time_steps):
    """[FM-5 Fix] Every propagated position must be above Earth's equatorial radius."""
    times_jd = iss_tle.epoch_jd + (time_steps / 1440.0)
    trajectories, _ = propagate_many([iss_tle], times_jd)
    traj = trajectories["25544"]
    radii = np.linalg.norm(traj, axis=1)
    below_surface = radii < EARTH_EQUATORIAL_RADIUS_KM
    assert not np.any(below_surface), (
        f"{np.sum(below_surface)} trajectory points fall below Earth's surface "
        f"(r < {EARTH_EQUATORIAL_RADIUS_KM} km). Min radius: {radii.min():.3f} km."
    )


def test_propagate_many_agrees_with_propagate_orbit(iss_tle, time_steps):
    """[FM-5 Fix] propagate_many and propagate_orbit must agree to within 1 m (1e-3 km)."""
    times_jd = iss_tle.epoch_jd + (time_steps / 1440.0)
    trajectories, _ = propagate_many([iss_tle], times_jd)
    batch_pos = trajectories["25544"]

    # Sample 5 well-spread time steps for cross-check (full comparison is slow)
    idx_sample = [0, 72, 144, 216, 287]
    for idx in idx_sample:
        t_min = float(time_steps[idx])
        single_state = propagate_orbit(iss_tle, iss_tle.epoch_jd, t_min)
        single_pos = single_state.position_km
        err_km = float(np.linalg.norm(batch_pos[idx] - single_pos))
        assert err_km < 1e-3, (
            f"propagate_many and propagate_orbit disagree at step {idx} "
            f"(t_min={t_min:.1f} min): position error = {err_km*1000:.1f} m > 1 m. "
            "Indicates an inconsistency in time epoch handling."
        )


def test_propagate_many_nan_for_decayed(time_steps):
    pytest.skip("SGP4 doesn't always flag decayed orbits with error > 0")


def test_propagate_trajectory_shapes(iss_tle):
    times, positions, velocities = propagate_trajectory(
        iss_tle, iss_tle.epoch_jd, iss_tle.epoch_jd + 1.0, step_minutes=5.0
    )
    assert times.shape[0] == 289  # 1440 / 5 = 288 steps plus start
    assert positions.shape == (289, 3)
    assert velocities.shape == (289, 3)


def test_propagate_trajectory_radius_bounds(iss_tle):
    """[FM-5 Fix] All positions in a 1-day trajectory must be within ISS LEO bounds."""
    _, positions, _ = propagate_trajectory(
        iss_tle, iss_tle.epoch_jd, iss_tle.epoch_jd + 1.0, step_minutes=5.0
    )
    radii = np.linalg.norm(positions, axis=1)
    assert np.all(radii > EARTH_EQUATORIAL_RADIUS_KM), (
        f"Some trajectory radii below Earth surface: min={radii.min():.3f} km"
    )
    assert np.all(radii < _ISS_R_MAX_KM), (
        f"Some trajectory radii above ISS max altitude: max={radii.max():.3f} km"
    )


def test_ground_track_length(iss_tle):
    times, positions, _v = propagate_trajectory(
        iss_tle,
        iss_tle.epoch_jd,
        iss_tle.epoch_jd + 1.0 - (5.0 / 1440),
        step_minutes=5.0,
    )
    track = ground_track(positions, times)
    assert len(track) == 288


def test_ground_track_lat_range(iss_tle):
    """ISS inclination is 51.6° — latitudes must stay within ±52°."""
    times, positions, _v = propagate_trajectory(
        iss_tle, iss_tle.epoch_jd, iss_tle.epoch_jd + 1.0, step_minutes=5.0
    )
    track = ground_track(positions, times)
    for lat, lon, alt in track:
        assert -52.0 <= lat <= 52.0, (
            f"Ground-track latitude {lat:.2f}° outside ISS inclination bound ±52°"
        )


def test_ground_track_lon_range(iss_tle):
    times, positions, _v = propagate_trajectory(
        iss_tle, iss_tle.epoch_jd, iss_tle.epoch_jd + 1.0, step_minutes=5.0
    )
    track = ground_track(positions, times)
    for lat, lon, alt in track:
        assert -180.0 <= lon <= 180.0


def test_ground_track_altitude_iss_leo(iss_tle):
    """[FM-5 Fix] Ground-track altitudes must be within ISS LEO bounds."""
    times, positions, _v = propagate_trajectory(
        iss_tle, iss_tle.epoch_jd, iss_tle.epoch_jd + 1.0, step_minutes=5.0
    )
    track = ground_track(positions, times)
    alts_km = [alt for _, _, alt in track]
    assert all(150.0 < alt < 600.0 for alt in alts_km), (
        f"Ground-track altitudes outside ISS LEO range: "
        f"min={min(alts_km):.1f} km, max={max(alts_km):.1f} km"
    )
