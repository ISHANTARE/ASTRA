import numpy as np
from astra import data_pipeline

def test_sun_position_shift_cta():
    """Verify that Sun position correctly shifts by ~8km when moving from TT to UTC scale."""
    # J2000 epoch (UTC)
    t_jd_utc = 2451545.0
    
    # 1. New implementation (UTC-aware)
    pos_new = data_pipeline.sun_position_de(t_jd_utc)
    
    # 2. Replicate old broken implementation (TT-direct)
    data_pipeline._ensure_skyfield()
    t_old = data_pipeline._skyfield_ts.tt_jd(t_jd_utc)
    earth = data_pipeline._skyfield_eph["earth"]
    sun = data_pipeline._skyfield_eph["sun"]
    pos_old_au = earth.at(t_old).observe(sun).position.au
    pos_old = np.array(pos_old_au) * 149597870.7
    
    # Difference should be ~2000 km since Earth travels at ~29.8 km/s
    # and the TT-UTC offset is ~69.2 seconds.
    diff_km = np.linalg.norm(pos_new - pos_old)
    print(f"Sun position shift (TT vs UTC): {diff_km:.3f} km")
    
    assert diff_km > 100.0, "Fix should create a measurable difference in Sun position"
    assert diff_km < 3000.0, "Sun position shift due to 69s should be ~2000 km"
    
    # Validation against Earth orbital speed (~30 km/s * 69s)
    assert 1800.0 < diff_km < 2200.0, f"Expected ~2000 km shift, got {diff_km:.2f} km"

if __name__ == "__main__":
    test_sun_position_shift_cta()
