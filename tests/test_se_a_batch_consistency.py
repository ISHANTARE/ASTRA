import numpy as np
from astra.orbit import propagate_orbit, propagate_many
from astra.tle import parse_tle

# ISS TLE
_ISS_L1 = "1 25544U 98067A   21001.00000000  .00001480  00000-0  34282-4 0  9990"
_ISS_L2 = "2 25544  51.6442 284.1199 0001364 338.5498  21.5664 15.48922536 12341"


def test_batch_consistency_se_a():
    """Verify that propagate_many and propagate_orbit give identical results (UT1-UTC fix)."""
    sat = parse_tle("ISS", _ISS_L1, _ISS_L2)

    # Target time: 1.5 hours after epoch
    t_since_mins = 90.0
    t_jd = sat.epoch_jd + (t_since_mins / 1440.0)

    # 1. Single-point propagation (always had UT1-UTC)
    state_single = propagate_orbit(sat, sat.epoch_jd, t_since_mins)

    # 2. Batch propagation (previously missing UT1-UTC)
    # We pass a 3-point array to ensure SatrecArray logic is exercised
    times_jd = np.array([sat.epoch_jd, t_jd, t_jd + 0.1])
    traj_map, vel_map = propagate_many([sat], times_jd)

    pos_batch = traj_map[sat.norad_id][1]
    vel_batch = vel_map[sat.norad_id][1]

    # Compare
    # Tolerance 1e-10 is used because small floating point differences might exist in
    # the time addition (scalar vs array), but it should NOT be 4km (which is ~1e0 order).
    np.testing.assert_allclose(
        state_single.position_km,
        pos_batch,
        rtol=1e-12,
        atol=1e-12,
        err_msg="Position mismatch: propagate_many and propagate_orbit diverged!",
    )
    np.testing.assert_allclose(
        state_single.velocity_km_s,
        vel_batch,
        rtol=1e-12,
        atol=1e-12,
        err_msg="Velocity mismatch: propagate_many and propagate_orbit diverged!",
    )

    print(
        "Consistency check PASSED. Batch and single-point propagation are synchronized."
    )


if __name__ == "__main__":
    test_batch_consistency_se_a()
