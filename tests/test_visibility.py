import pytest
import numpy as np

from astra.visibility import (
    _wgs84_observer_itrs,
    _itrs_to_enu_matrix,
    visible_from_location,
    passes_over_location,
)
from astra.models import Observer
from skyfield.api import Topos, load

def test_wgs84_observer_itrs():
    # Equator, 0 deg lon: X approx 6378.137, Y=0, Z=0
    pos = _wgs84_observer_itrs(0.0, 0.0, 0.0)
    assert abs(pos[0] - 6378.137) < 0.1
    assert abs(pos[1]) < 0.1
    assert abs(pos[2]) < 0.1

    # North Pole: X=0, Y=0, Z approx 6356.752 (b)
    # 2f - f^2 = e^2. a(1-f) = b
    pos2 = _wgs84_observer_itrs(90.0, 0.0, 0.0)
    assert abs(pos2[0]) < 0.1
    assert abs(pos2[1]) < 0.1
    assert abs(pos2[2] - 6356.752) < 0.1

def test_itrs_to_enu_matrix():
    # Equator, 0 deg: 
    # East = [0, 1, 0]
    # North = [0, 0, 1]
    # Up = [1, 0, 0]
    R = _itrs_to_enu_matrix(0.0, 0.0)
    assert np.allclose(R[0], [0, 1, 0])
    assert np.allclose(R[1], [0, 0, 1])
    assert np.allclose(R[2], [1, 0, 0])

def test_visible_from_location_shapes(iss_tle, observer):
    # Vectorized check
    T = 10
    times = np.linspace(2459000.0, 2459000.1, T)
    pos_teme = np.ones((T, 3)) * 6700.0 
    
    # Needs actual skyfield time translation without crashing
    elev = visible_from_location(pos_teme, times, observer)
    assert elev.shape == (T,)

def test_passes_over_location_integration(iss_tle, observer):
    # This involves binary search and full SGP4
    start_jd = iss_tle.epoch_jd
    end_jd = start_jd + (100.0 / 1440.0) # 100 minutes
    
    passes = passes_over_location(iss_tle, observer, float(start_jd), float(end_jd), 1.0)
    
    # We don't guarantee a pass in 100 mins, but it shouldn't crash
    assert isinstance(passes, list)
    if passes:
        for p in passes:
            # Check binary search invariants
            assert p.duration_seconds > 0.0
            assert p.max_elevation_deg >= observer.min_elevation_deg
            assert p.aos_jd < p.tca_jd < p.los_jd


def test_passes_over_location_accepts_omm(iss_omm, observer):
    """``passes_over_location`` accepts OMM-backed ``SatelliteState``."""
    start_jd = iss_omm.epoch_jd
    end_jd = start_jd + (100.0 / 1440.0)
    passes = passes_over_location(iss_omm, observer, float(start_jd), float(end_jd), 1.0)
    assert isinstance(passes, list)
