import pytest
import math
from astra.utils import vincenty_distance, orbital_elements, orbit_period

def test_vincenty_distance_same_point():
    assert vincenty_distance(0.0, 0.0, 0.0, 0.0) == 0.0

def test_vincenty_distance_equator():
    # 1 degree on equator is approx 111.19 km
    d = vincenty_distance(0.0, 0.0, 0.0, 1.0)
    assert 111.0 < d < 112.0

def test_orbital_elements_extracts_correctly():
    line2 = "2 25544  51.6442 284.1199 0001364 338.5498  21.5664 15.48922536 12345"
    elements = orbital_elements(line2)
    assert elements["inclination_deg"] == 51.6442
    assert elements["raan_deg"] == 284.1199
    assert elements["eccentricity"] == 0.0001364
    assert elements["arg_perigee_deg"] == 338.5498
    assert elements["mean_anomaly_deg"] == 21.5664
    assert elements["mean_motion_rev_per_day"] == 15.48922536

def test_orbit_period():
    # 15.48 rev/day -> approx 93 minutes
    p = orbit_period(15.48922536)
    assert 92.0 < p < 94.0

def test_orbit_period_zero():
    assert math.isinf(orbit_period(0.0))
