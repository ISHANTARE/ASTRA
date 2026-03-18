# 11 — ASTRA Core: Testing Strategy

---

## 1. Testing Philosophy

Every function in ASTRA Core must be independently testable with:
- Explicit, fixed input data (no network calls, no file reads)
- Known, verifiable expected outputs
- Isolated from other modules

**Test Pyramid for ASTRA Core:**

```
         [Integration Tests]
        (end-to-end pipeline)
       /                     \
      [Module Tests]      [Edge Case Tests]
     (function-level)      (boundary inputs)
    /                                       \
[Unit Tests] ─────────────────── [Property Tests]
(single function, happy path)   (hypothesis-generated inputs)
```

---

## 2. Test File Structure

```
tests/
├── conftest.py              # Shared fixtures (sample TLEs, time values)
├── test_tle.py              # astra.tle tests
├── test_orbit.py            # astra.orbit tests
├── test_debris.py           # astra.debris tests
├── test_conjunction.py      # astra.conjunction tests
├── test_visibility.py       # astra.visibility tests
├── test_time.py             # astra.time tests
├── test_utils.py            # astra.utils tests
├── test_errors.py           # Error class tests
├── test_models.py           # Dataclass tests
└── test_integration.py      # End-to-end pipeline tests
```

---

## 3. Shared Fixtures (`conftest.py`)

```python
# tests/conftest.py
import pytest
import numpy as np

# Real ISS TLE (use a fixed known epoch for determinism)
ISS_NAME = "ISS (ZARYA)"
ISS_LINE1 = "1 25544U 98067A   21001.00000000  .00001480  00000-0  34282-4 0  9990"
ISS_LINE2 = "2 25544  51.6442 284.1199 0001364 338.5498  21.5664 15.48922536 12345"
ISS_NORAD = "25544"

# Known Julian Date for testing: 2025-01-01T00:00:00 UTC
TEST_JD_START = 2460676.5  # 2025-01-01T00:00:00 UTC
TEST_JD_END   = 2460677.5  # 2025-01-02T00:00:00 UTC

@pytest.fixture
def iss_tle(astra):
    return astra.parse_tle(ISS_NAME, ISS_LINE1, ISS_LINE2)

@pytest.fixture
def sample_observer():
    from astra.models import Observer
    return Observer(
        name="Bangalore",
        latitude_deg=12.97,
        longitude_deg=77.59,
        elevation_m=920.0,
        min_elevation_deg=10.0
    )

@pytest.fixture
def time_steps():
    return np.arange(0.0, 24 * 60, 5.0)  # 288 steps

@pytest.fixture
def small_catalog():
    """A catalog of 5 known debris objects for pipeline testing."""
    ...  # Returns list[SatelliteTLE] with known, real TLEs
```

---

## 4. Module-by-Module Test Specifications

### 4.1 `test_tle.py`

| Test Name | Input | Expected Behavior |
|---|---|---|
| `test_parse_valid_tle` | ISS TLE strings | Returns `SatelliteTLE` with correct NORAD ID `"25544"` |
| `test_parse_norad_id` | ISS TLE | `tle.norad_id == "25544"` |
| `test_parse_epoch_jd` | ISS TLE | Epoch JD is a valid float > 2400000 |
| `test_line1_wrong_length` | Line1 of 68 chars | Raises `InvalidTLEError(reason="line1_wrong_length")` |
| `test_line2_wrong_length` | Line2 of 70 chars | Raises `InvalidTLEError(reason="line2_wrong_length")` |
| `test_bad_checksum_line1` | Line1 with digit 9 at col 69 changed to 0 | Raises `InvalidTLEError(reason="line1_checksum_invalid")` |
| `test_bad_checksum_line2` | Corrupted line2 checksum | Raises `InvalidTLEError(reason="line2_checksum_invalid")` |
| `test_line1_bad_start` | Line1 starting with '2' | Raises `InvalidTLEError(reason="line1_bad_start")` |
| `test_validate_valid` | Valid TLE | Returns `True` |
| `test_validate_invalid` | Corrupt TLE | Returns `False` (no exception) |
| `test_load_catalog_all_valid` | 3 valid TLE triplets | Returns list of 3 `SatelliteTLE` |
| `test_load_catalog_one_invalid` | 2 valid + 1 invalid | Returns list of 2 (invalid skipped silently) |
| `test_load_catalog_all_invalid` | 3 invalid TLEs | Raises `AstraError` |
| `test_load_catalog_empty` | Empty list | Returns empty list |

---

### 4.2 `test_orbit.py`

| Test Name | Input | Expected Behavior |
|---|---|---|
| `test_propagate_returns_orbital_state` | ISS TLE, t=0 | Returns `OrbitalState` |
| `test_propagate_position_shape` | ISS TLE, t=0 | `state.position_km.shape == (3,)` |
| `test_propagate_velocity_shape` | ISS TLE, t=0 | `state.velocity_km_s.shape == (3,)` |
| `test_propagate_position_units` | ISS TLE, t=0 | Position magnitude ≈ 6700–7200 km (LEO) |
| `test_propagate_velocity_units` | ISS TLE, t=0 | Velocity magnitude ≈ 7–8 km/s (LEO) |
| `test_propagate_error_code_zero` | Valid ISS TLE | `state.error_code == 0` |
| `test_propagate_many_output_shape` | [ISS], time_steps(288,) | Returns dict with key "25544" → shape `(288, 3)` |
| `test_propagate_many_no_nan_for_valid` | Valid ISS TLE | No `np.nan` values in output array |
| `test_propagate_many_nan_for_decayed` | Decayed TLE, far-future time | All rows are NaN |
| `test_propagate_trajectory_shapes` | ISS, 24h window | Returns `(time_array(288,), positions(288,3))` |
| `test_ground_track_length` | ISS 24h trajectory | Returns list of 288 `(lat, lon)` tuples |
| `test_ground_track_lat_range` | ISS trajectory | All lats within [-51.6, 51.6] (ISS inclination) |
| `test_ground_track_lon_range` | ISS trajectory | All lons within [-180, 180] |

**Correctness Validation:**
- Use a known ISS position at a specific epoch from NASA Horizons or Celestrak as reference
- Run `propagate_orbit()` at that exact epoch and compare positions within tolerance of ±1 km

---

### 4.3 `test_debris.py`

| Test Name | Input | Expected Behavior |
|---|---|---|
| `test_filter_altitude_returns_subset` | 10 objects, min=200, max=2000 | Returns only objects with alt in range |
| `test_filter_altitude_all_pass` | Objects all in range | Returns all |
| `test_filter_altitude_none_pass` | Objects all outside range | Returns empty list |
| `test_filter_altitude_invalid_min_gt_max` | min=2000, max=200 | Raises `FilterError` |
| `test_filter_altitude_negative_min` | min=-100 | Raises `FilterError` |
| `test_filter_region_inclination_too_low` | Equatorial sat, polar region | Object excluded |
| `test_filter_region_polar_orbit_included` | Polar sat (incl=98°), any region | Object included |
| `test_filter_region_invalid_lat` | lat_min=100 | Raises `FilterError` |
| `test_filter_time_window_fresh_tle` | TLE age = 3 days | Object included |
| `test_filter_time_window_stale_leo` | LEO object, TLE age = 10 days | Object excluded |
| `test_filter_time_window_stale_heo` | HEO object, TLE age = 10 days | Object included |
| `test_catalog_statistics_returns_dict` | 5 objects | Returns dict |
| `test_catalog_statistics_total_count` | 5 objects | `stats["total_count"] == 5` |
| `test_catalog_statistics_by_regime` | Mix of LEO/GEO objects | Both appear in `by_regime` |
| `test_make_debris_object` | ISS TLE | Returns `DebrisObject` with correct altitude |

---

### 4.4 `test_conjunction.py`

| Test Name | Input | Expected Behavior |
|---|---|---|
| `test_distance_3d_single_point` | Two 3D positions 5 km apart | Returns 5.0 |
| `test_distance_3d_vectorized` | Two (T,3) arrays | Returns shape `(T,)` |
| `test_distance_3d_zero` | Same position for both | Returns 0.0 |
| `test_closest_approach_finds_minimum` | Two trajectories crossing | Returns correct minimum |
| `test_closest_approach_tca_time` | Known crossing at t=10 | `tca_jd` corresponds to t=10 |
| `test_find_conjunctions_no_events` | Two non-crossing trajectories | Returns empty list |
| `test_find_conjunctions_detects_close_pass` | Two converging trajectories < 5km | Returns one event |
| `test_find_conjunctions_pair_ordering` | Objects "B" and "A" | `event.object_a_id < event.object_b_id` alphabetically |
| `test_collision_probability_zero_miss` | miss_dist=0.0 | Returns probability = 1.0 |
| `test_collision_probability_large_miss` | miss_dist=1000.0 km | Returns probability ≈ 0.0 |
| `test_collision_probability_range` | Any valid inputs | Returns value in [0.0, 1.0] |

**Synthetic Test Data for Conjunction:**
Create two trajectories that are guaranteed to pass within 3 km at timestep 50:

```python
def crossing_trajectories():
    T = 288
    traj_a = np.zeros((T, 3))
    traj_b = np.zeros((T, 3))
    # Object A moves +x at 1 km/step
    traj_a[:, 0] = np.arange(T) * 1.0
    # Object B starts 100 km away, moves -x at 1 km/step + small offset
    traj_b[:, 0] = 200.0 - np.arange(T) * 1.0
    traj_b[:, 1] = 2.5  # 2.5 km perpendicular offset = miss distance
    return traj_a, traj_b
    # They approach closest at step T//2 with distance ≈ 2.5 km
```

---

### 4.5 `test_visibility.py`

| Test Name | Input | Expected Behavior |
|---|---|---|
| `test_visible_from_location_returns_array` | ISS trajectory, observer | Returns ndarray shape `(T,)` |
| `test_visible_elevation_range` | ISS trajectory | All values in [-90, 90] degrees |
| `test_passes_over_location_returns_list` | ISS TLE, observer, 24h | Returns list |
| `test_pass_event_fields` | Any pass | `aos_jd < tca_jd < los_jd` |
| `test_pass_max_elevation_positive` | Any pass | `max_elevation_deg >= min_elevation_deg` |
| `test_pass_duration_positive` | Any pass | `duration_seconds > 0` |
| `test_no_passes_above_horizon` | GEO object, polar observer | Empty list or low elevation |

---

### 4.6 Integration Tests (`test_integration.py`)

```python
def test_full_pipeline_end_to_end():
    """Test: load → filter → propagate → conjunctions does not error."""
    raw_lines = load_fixture_tle_file("sample_catalog.tle")
    tles = load_tle_catalog(raw_lines)
    catalog = [make_debris_object(t) for t in tles]
    
    leo = filter_altitude(catalog, min_km=200, max_km=2000)
    current = filter_time_window(leo, TEST_JD_START, TEST_JD_END)
    
    assert len(current) > 0, "Pipeline should have survivors"
    
    time_steps = np.arange(0, 24*60, 5.0)
    trajectories = propagate_many([obj.tle for obj in current], time_steps)
    
    assert len(trajectories) == len(current)
    
    times_jd = TEST_JD_START + time_steps / 1440.0
    events = find_conjunctions(trajectories, times_jd, threshold_km=50.0)
    
    assert isinstance(events, list)
    # All events should have valid fields:
    for e in events:
        assert e.miss_distance_km <= 50.0
        assert e.object_a_id < e.object_b_id
        assert e.risk_level in ("LOW", "MEDIUM", "HIGH", "CRITICAL")
```

---

## 5. Test Data Strategy

| Data Type | Source |
|---|---|
| Valid TLEs | Fixed, hardcoded ISS and debris TLEs with known epochs |
| Invalid TLEs | Manually crafted with corrupted checksums, wrong lengths |
| Synthetic trajectories | Python-generated NumPy arrays with known geometric properties |
| Reference positions | NASA Horizons or online SGP4 calculators for correctness checks |

**Rule:** No test may make a network request. TLE data for tests is embedded in `conftest.py` or `.tle` fixture files in `tests/fixtures/`.

---

## 6. Running Tests

```bash
# Run all tests
pytest tests/

# With coverage
pytest tests/ --cov=astra --cov-report=term-missing

# Run a single module
pytest tests/test_conjunction.py

# Run with verbose output
pytest tests/ -v

# Run only fast (non-propagation) tests
pytest tests/ -m "not slow"
```
