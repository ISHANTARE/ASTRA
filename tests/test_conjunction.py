import pytest
import numpy as np

from astra.conjunction import distance_3d, closest_approach, find_conjunctions
from astra.models import ConjunctionEvent, DebrisObject, SatelliteTLE
from astra.debris import make_debris_object
from astra.tle import parse_tle

def crossing_trajectories():
    T = 288
    traj_a = np.zeros((T, 3))
    traj_b = np.zeros((T, 3))
    
    # 6771 km is approx orbital radius of ISS. 
    # Must match elements_map bounding shell to pass Phase 1 SAP filter!
    traj_a[:, 2] = 6771.0
    traj_b[:, 2] = 6771.0
    
    # Move along X axis to cross exactly at T/2
    traj_a[:, 0] = np.linspace(-100.0, 100.0, T)
    traj_b[:, 0] = np.linspace(100.0, -100.0, T)
    
    # B has a slight Y offset to create a miss distance
    traj_b[:, 1] = 2.5
    
    return traj_a, traj_b


def test_distance_3d_single_point():
    a = np.array([0.0, 0.0, 0.0])
    b = np.array([3.0, 4.0, 0.0])
    assert distance_3d(a, b) == 5.0


def test_closest_approach_finds_minimum():
    traj_a, traj_b = crossing_trajectories()
    times = np.linspace(0.0, 1.0, 288) # 1 day spread
    min_dist, tca, idx = closest_approach(traj_a, traj_b, times)
    assert abs(min_dist - 2.5) < 1e-5


@pytest.fixture
def conjunction_elements():
    line1 = "1 25544U 98067A   21001.00000000  .00001480  00000-0  34282-4 0  9990"
    line2 = "2 25544  51.6442 284.1199 0001364 338.5498  21.5664 15.48922536 12341"
    sat_a = parse_tle("A", line1, line2)
    
    line1b = "1 99999U 98067A   21001.00000000  .00001480  00000-0  34282-4 0  9995"
    line2b = "2 99999  51.6442 284.1199 0001364 338.5498  21.5664 15.48922536 12346"
    sat_b = parse_tle("B", line1b, line2b)
    
    # Must override the derived parameters manually to ensure the artificial 
    # Z=6771 positions fall directly inside the bounding shell.
    obj_a = make_debris_object(sat_a)
    obj_b = make_debris_object(sat_b)
    
    # Forge bounding shells to encompass Z=6771
    object.__setattr__(obj_a, 'perigee_km', 300.0)
    object.__setattr__(obj_a, 'apogee_km', 500.0)
    object.__setattr__(obj_b, 'perigee_km', 300.0)
    object.__setattr__(obj_b, 'apogee_km', 500.0)
    
    return {
        "25544": obj_a,
        "99999": obj_b
    }


def test_find_conjunctions_no_events(conjunction_elements):
    T = 288
    times = np.arange(288.0)
    # Put them far away outside bounding shell
    trajs = {
        "25544": np.ones((T, 3)) * 10000.0,
        "99999": np.ones((T, 3)) * 50000.0,
    }
    events = find_conjunctions(trajs, times, conjunction_elements, threshold_km=5.0)
    assert len(events) == 0


def test_find_conjunctions_detects_close_pass(conjunction_elements):
    traj_a, traj_b = crossing_trajectories()
    times = np.linspace(0.0, 0.2, 288) # Times spanning the crossing
    trajs = {
        "25544": traj_a,
        "99999": traj_b,
    }
    
    events = find_conjunctions(trajs, times, conjunction_elements, threshold_km=5.0)
    assert len(events) == 1
    assert abs(events[0].miss_distance_km - 2.5) < 1e-3


def test_find_conjunctions_pair_ordering(conjunction_elements):
    traj_a, traj_b = crossing_trajectories()
    times = np.linspace(0.0, 0.2, 288)
    trajs = {
        "99999": traj_a,
        "25544": traj_b,
    }
    events = find_conjunctions(trajs, times, conjunction_elements, threshold_km=5.0)
    assert len(events) == 1
    assert int(events[0].object_a_id) < int(events[0].object_b_id)
