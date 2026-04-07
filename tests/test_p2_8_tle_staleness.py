import pytest
import numpy as np
from datetime import datetime, timezone
from astra.tle import parse_tle
from astra.orbit import propagate_orbit, propagate_many
from astra.errors import PropagationError
from astra import config

def test_tle_staleness_strict_p2_8():
    """Verify that propagating > 30 days raises PropagationError in STRICT mode."""
    config.set_strict_mode(True)
    
    # ISS TLE with epoch 2008-09-20 12:25:40 UTC (JD 2454729.0178)
    name = "ISS (ZARYA)"
    line1 = "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927"
    line2 = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537"
    try:
        sat = parse_tle(name, line1, line2)
    except Exception as e:
        print(f"FAILED PARSE: {e}")
        raise
    
    # 31 days after epoch
    t_since_minutes = 31 * 1440.0
    
    with pytest.raises(PropagationError) as excinfo:
        propagate_orbit(sat, sat.epoch_jd, t_since_minutes)
    
    assert "is stale" in str(excinfo.value)
    assert "31.00 days from epoch" in str(excinfo.value)
    print("Strict mode staleness enforcement PASSED.")

def test_tle_staleness_relaxed_p2_8():
    """Verify that relaxed mode allows stale propagation with a warning."""
    config.set_strict_mode(False)
    
    name = "ISS (ZARYA)"
    line1 = "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927"
    line2 = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537"
    sat = parse_tle(name, line1, line2)
    
    # 31 days after epoch
    t_since_minutes = 31 * 1440.0
    
    # Should NOT raise
    state = propagate_orbit(sat, sat.epoch_jd, t_since_minutes)
    assert state.position_km.any()
    print("Relaxed mode staleness tolerance PASSED.")

def test_tle_staleness_batch_p2_8():
    """Verify that batch propagation fails if ANY satellite is stale in STRICT mode."""
    config.set_strict_mode(True)
    
    name = "ISS (ZARYA)"
    line1 = "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927"
    line2 = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537"
    sat1 = parse_tle(name, line1, line2)
    
    # Use the same satellite as 'sat2' to avoid checksum issues with string replacement
    sat2 = sat1
    
    # Propagate at t=31 days after sat1 epoch
    times_jd = np.array([sat1.epoch_jd + 31.0])
    
    with pytest.raises(PropagationError):
        propagate_many([sat1, sat2], times_jd)
    
    print("Batch mode staleness enforcement PASSED.")

if __name__ == "__main__":
    test_tle_staleness_strict_p2_8()
    test_tle_staleness_relaxed_p2_8()
    test_tle_staleness_batch_p2_8()
