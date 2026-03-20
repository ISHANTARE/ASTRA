import pytest
import numpy as np

from astra import propagate_orbit, propagate_many, propagate_trajectory, ground_track
from astra.models import OrbitalState

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
    state = propagate_orbit(iss_tle, iss_tle.epoch_jd, 0.0)
    mag = np.linalg.norm(state.position_km)
    assert 6000 < mag < 8000  # LEO bounds approximately

def test_propagate_velocity_units(iss_tle):
    state = propagate_orbit(iss_tle, iss_tle.epoch_jd, 0.0)
    mag = np.linalg.norm(state.velocity_km_s)
    assert 6.0 < mag < 9.0  # LEO bounds km/s

def test_propagate_error_code_zero(iss_tle):
    state = propagate_orbit(iss_tle, iss_tle.epoch_jd, 0.0)
    assert state.error_code == 0

def test_propagate_many_output_shape(iss_tle, time_steps):
    trajectories = propagate_many([iss_tle], time_steps)
    assert "25544" in trajectories
    assert trajectories["25544"].shape == (288, 3)

def test_propagate_many_no_nan_for_valid(iss_tle, time_steps):
    trajectories = propagate_many([iss_tle], time_steps)
    assert not np.isnan(trajectories["25544"]).any()

def test_propagate_many_nan_for_decayed(time_steps):
    from astra import parse_tle
    decayed = parse_tle("Decayed", "1 25544U 98067A   99001.00000000  .00001480  00000-0  34282-4 0  9998", "2 25544  51.6442 284.1199 0001364 338.5498  21.5664 15.48922536 12349")
    t_steps_future = np.arange(1000000.0, 1000000.0 + 100, 5) # massive offset where it should fail
    trajectories = propagate_many([decayed], t_steps_future)
    assert np.isnan(trajectories["25544"]).all()

def test_propagate_trajectory_shapes(iss_tle):
    times, positions = propagate_trajectory(iss_tle, iss_tle.epoch_jd, iss_tle.epoch_jd + 1.0, step_minutes=5.0)
    assert times.shape[0] == 289  # 1440 / 5 = 288 steps plus start
    assert positions.shape == (289, 3)

def test_ground_track_length(iss_tle):
    times, positions = propagate_trajectory(iss_tle, iss_tle.epoch_jd, iss_tle.epoch_jd + 1.0 - (5.0/1440), step_minutes=5.0)
    track = ground_track(positions, times)
    assert len(track) == 288

def test_ground_track_lat_range(iss_tle):
    times, positions = propagate_trajectory(iss_tle, iss_tle.epoch_jd, iss_tle.epoch_jd + 1.0, step_minutes=5.0)
    track = ground_track(positions, times)
    for lat, lon in track:
        assert -52.0 <= lat <= 52.0

def test_ground_track_lon_range(iss_tle):
    times, positions = propagate_trajectory(iss_tle, iss_tle.epoch_jd, iss_tle.epoch_jd + 1.0, step_minutes=5.0)
    track = ground_track(positions, times)
    for lat, lon in track:
        assert -180.0 <= lon <= 180.0
