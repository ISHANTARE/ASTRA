from dataclasses import replace

import pytest
import numpy as np

from astra.conjunction import distance_3d, closest_approach, find_conjunctions
from astra.debris import make_debris_object
from astra.spatial_index import SpatialIndex
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
    times = np.linspace(0.0, 1.0, 288)  # 1 day spread
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
    obj_a = replace(make_debris_object(sat_a), perigee_km=300.0, apogee_km=500.0)
    obj_b = replace(make_debris_object(sat_b), perigee_km=300.0, apogee_km=500.0)

    return {"25544": obj_a, "99999": obj_b}


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
    times = np.linspace(0.0, 0.2, 288)  # Times spanning the crossing
    trajs = {
        "25544": traj_a,
        "99999": traj_b,
    }

    events = find_conjunctions(trajs, times, conjunction_elements, threshold_km=5.0)
    assert len(events) == 1
    assert abs(events[0].miss_distance_km - 2.5) < 1e-3


def test_find_conjunctions_deterministic_event_contract(conjunction_elements):
    """The threaded event pipeline must produce stable, fully populated events."""
    traj_a, traj_b = crossing_trajectories()
    t0 = 2_460_000.5
    times = np.linspace(t0, t0 + (120.0 / 86400.0), len(traj_a))
    trajs = {
        "25544": traj_a,
        "99999": traj_b,
    }
    vel_map = {
        "25544": np.repeat([[1.0, 0.0, 0.0]], len(traj_a), axis=0),
        "99999": np.repeat([[-1.0, 0.0, 0.0]], len(traj_b), axis=0),
    }
    cov_map = {
        "25544": np.eye(3) * 1e-4,
        "99999": np.eye(3) * 1e-4,
    }

    summaries = []
    for workers in (1, 2, 4):
        events = find_conjunctions(
            trajs,
            times,
            conjunction_elements,
            threshold_km=5.0,
            coarse_threshold_km=25.0,
            cov_map=cov_map,
            vel_map=vel_map,
            max_workers=workers,
        )
        assert len(events) == 1
        event = events[0]
        assert (event.object_a_id, event.object_b_id) == ("25544", "99999")
        assert event.miss_distance_km == pytest.approx(2.5, abs=1e-3)
        assert event.tca_jd == pytest.approx(times[len(times) // 2], abs=1e-4)
        assert event.relative_velocity_km_s == pytest.approx(2.0, abs=1e-12)
        assert event.collision_probability is not None
        assert event.covariance_source == "CDM"
        assert np.linalg.norm(event.position_a_km - event.position_b_km) == pytest.approx(
            event.miss_distance_km, abs=1e-9
        )
        summaries.append(
            (
                event.object_a_id,
                event.object_b_id,
                round(event.tca_jd, 10),
                round(event.miss_distance_km, 9),
                round(event.relative_velocity_km_s, 12),
                event.covariance_source,
            )
        )

    assert summaries[0] == summaries[1] == summaries[2]


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


def test_spatial_index_pairs_match_bruteforce():
    """SpatialIndex pair set matches naive O(n²) distance threshold search."""
    rng = np.random.default_rng(42)
    ids = [f"id{i}" for i in range(12)]
    pts = rng.uniform(low=-500.0, high=500.0, size=(12, 3))
    thresh = 150.0

    brute: set[tuple[str, str]] = set()
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            d = float(np.linalg.norm(pts[i] - pts[j]))
            if d <= thresh:
                a, b = ids[i], ids[j]
                brute.add((min(a, b), max(a, b)))

    idx = SpatialIndex()
    for k, oid in enumerate(ids):
        idx.insert(oid, pts[k])
    tree_pairs = set(idx.query_pairs(threshold_km=thresh))

    assert tree_pairs == brute


def test_find_conjunctions_custom_threshold(conjunction_elements):
    traj_a, traj_b = crossing_trajectories()
    times = np.linspace(0.0, 0.2, 288)
    trajs = {
        "99999": traj_a,
        "25544": traj_b,
    }
    events = find_conjunctions(
        trajs, times, conjunction_elements, threshold_km=5.0, coarse_threshold_km=25.0
    )
    assert len(events) == 1
    assert abs(events[0].miss_distance_km - 2.5) < 1e-3


def test_find_conjunctions_dense_fallback_uses_seconds_not_days(
    conjunction_elements, monkeypatch
):
    import scipy.optimize

    def _raise_minimize(*_args, **_kwargs):
        raise RuntimeError("forced optimizer failure")

    monkeypatch.setattr(scipy.optimize, "minimize_scalar", _raise_minimize)

    traj_a, traj_b = crossing_trajectories()
    t0 = 2_460_000.5
    times = np.linspace(t0, t0 + (120.0 / 86400.0), len(traj_a))
    trajs = {
        "25544": traj_a,
        "99999": traj_b,
    }
    cov_map = {
        "25544": np.eye(3) * 1e-4,
        "99999": np.eye(3) * 1e-4,
    }

    events = find_conjunctions(
        trajs,
        times,
        conjunction_elements,
        threshold_km=5.0,
        coarse_threshold_km=25.0,
        cov_map=cov_map,
        max_workers=1,
    )

    assert len(events) == 1
    assert events[0].miss_distance_km == pytest.approx(2.5, abs=1e-2)


def test_closest_approach_rejects_unsorted_times():
    traj_a, traj_b = crossing_trajectories()
    times = np.array([2.0, 1.0, 3.0])

    with pytest.raises(Exception, match="strictly increasing"):
        closest_approach(traj_a[:3], traj_b[:3], times)
