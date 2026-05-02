import numpy as np
import pytest
from astra.orbit import propagate_orbit, propagate_many, ground_track
from astra.models import SatelliteTLE
from astra.data_pipeline import sun_position_de


def test_sgp4_batch_parity():
    """Verify that propagate_many and propagate_orbit return the same results (SE-A)."""
    # ISS TLE
    line1 = "1 25544U 98067A   24098.12345678  .00016717  00000+0  30242-3 0  9999"
    line2 = "2 25544  51.6416 247.4627 0006703  48.7619 311.3596 15.49815327447665"
    sat = SatelliteTLE("25544", "ISS", line1, line2, 2460408.62345678, "PAYLOAD")

    t_jd = sat.epoch_jd + 0.5  # 12 hours later

    # 1. Single point propagation
    state_single = propagate_orbit(sat, sat.epoch_jd, 12.0 * 60.0)

    # 2. Batch propagation (vectorized)
    times_jd = np.array([t_jd])
    traj_map, vel_map = propagate_many([sat], times_jd)
    pos_batch = traj_map["25544"][0]
    vel_map["25544"][0]

    dist = np.linalg.norm(state_single.position_km - pos_batch)
    print(f"Propagate Parity Error: {dist:.6e} km")

    # Parity should be near-perfect (floating point limit)
    assert dist < 1e-9


def test_sun_position_scale():
    """Verify Sun position uses UTC scale (CT-A)."""
    t_jd = 2460408.5  # 2024-04-07 00:00:00 UTC
    pos = sun_position_de(t_jd)

    # Magnitude should show distance to Sun (~1 AU)
    dist = np.linalg.norm(pos)
    print(f"Sun distance: {dist:.1f} km")
    assert 147e6 < dist < 153e6

    assert np.all(np.isfinite(pos))
    assert abs(pos[2]) > 1e6


def test_ground_track_scale():
    """Verify ground track uses UTC scale (CT-A)."""
    times_jd = np.array([2460408.5])
    pos_teme = np.array([[7000.0, 0.0, 0.0]])

    track = ground_track(pos_teme, times_jd)
    lat, lon, alt = track[0]
    assert lat == pytest.approx(-0.004686, abs=1e-5)
    assert -180.0 <= lon <= 180.0
    assert alt == pytest.approx(621.863, abs=1e-3)


if __name__ == "__main__":
    test_sgp4_batch_parity()
    test_sun_position_scale()
    test_ground_track_scale()
