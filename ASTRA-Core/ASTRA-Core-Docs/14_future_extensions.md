# 14 — ASTRA Core: Future Extensions

---

## 1. Purpose

This document catalogs **planned future enhancements** to ASTRA Core that are explicitly out of scope for the initial implementation but must be architecturally compatible with the current design.

All current design decisions (pure functions, no global state, module separation) have been taken with these extensions in mind.

---

## 2. Extension 1: Atmospheric Drag and Re-Entry Prediction

**Status:** Planned, Phase 2

**Description:** Predict time of re-entry for decaying LEO objects by modeling ballistic coefficient and atmospheric density variations.

**Proposed Module:** `astra.reentry`

**Key Functions:**
```python
def estimate_reentry_date(
    tle: SatelliteTLE,
    bstar_drag: float,
) -> tuple[float, float]:  # (earliest_jd, latest_jd) with uncertainty

def decay_rate_km_per_day(
    altitude_km: float,
    solar_flux_index: float,  # F10.7 cm solar radio flux
) -> float
```

**Dependency Addition:** Will require `nrlmsise00` (atmospheric model library) or tabulated NRLMSISE-00 data.

**Current Architecture Compatibility:** Pure function design ensures this can be a new module with zero modifications to existing modules.

---

## 3. Extension 2: Maneuver Detection and Prediction

**Status:** Planned, Phase 2

**Description:** Detect orbital maneuvers by comparing consecutive TLE epochs and identifying anomalous delta-V events.

**Proposed Module:** `astra.maneuver`

**Key Functions:**
```python
def detect_maneuver(
    tle_old: SatelliteTLE,
    tle_new: SatelliteTLE,
) -> Optional[ManeuverEvent]:

def predict_post_maneuver_orbit(
    tle: SatelliteTLE,
    delta_v_km_s: np.ndarray,  # (3,) maneuver vector
    maneuver_time_jd: float,
) -> SatelliteTLE  # New TLE after maneuver
```

---

## 4. Extension 3: Parallel Propagation with ProcessPoolExecutor

**Status:** Planned, Phase 2 (Performance)

**Description:** For catalogs above 10,000 survivors, distribute `propagate_many()` across CPU cores.

**Implementation Pattern:**
```python
from concurrent.futures import ProcessPoolExecutor

def propagate_many_parallel(
    satellites: list[SatelliteTLE],
    time_steps: np.ndarray,
    n_workers: int = 4,
) -> TrajectoryMap:
    batch_size = len(satellites) // n_workers
    batches = chunk(satellites, batch_size)
    
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = [
            executor.submit(propagate_many, batch, time_steps)
            for batch in batches
        ]
        results = [f.result() for f in futures]
    
    return merge_trajectory_maps(results)
```

**Current Compatibility:** Since `propagate_many` is already a pure function, this extension requires only wrapping — zero internal changes.

---

## 5. Extension 4: Orbit Determination from Observations

**Status:** Planned, Phase 3

**Description:** Given a set of ground-based angle observations (azimuth, elevation, time), compute an initial orbit estimate using Gauss or Laplace method.

**Proposed Module:** `astra.od` (orbit determination)

**This requires:** Gauss method numerical solver — may use `scipy.optimize`.

---

## 6. Extension 5: Historical TLE Archive Analysis

**Status:** Planned, Phase 3

**Description:** Analyze how an orbital object's elements have changed over time using a series of TLE snapshots. Enables trend analysis of decay, maneuver history, and catalog evolution.

**New Data Model:**
```python
@dataclass(frozen=True)
class TLETimeSeries:
    norad_id: str
    tles: tuple[SatelliteTLE, ...]  # ordered by epoch
```

---

## 7. Extension 6: Covariance Propagation

**Status:** Planned, Phase 3 (Advanced Conjunction)

**Description:** Propagate position uncertainty covariance matrices alongside trajectories to improve collision probability estimates beyond the simplified Chan method.

**Current Impact:** `ConjunctionEvent` already has a `collision_probability` field — this extension improves the accuracy of that field without any model changes.

---

## 8. Extension 7: Sensor Tasking Optimization

**Status:** Planned, Phase 3

**Description:** Given a list of objects of interest and a network of ground sensors, compute an optimal observation schedule to maximize catalog coverage.

**Proposed Module:** `astra.sensor`

---

## 9. Extension 8: Space Weather Integration

**Status:** Planned, Phase 2+

**Description:** Integrate real-time solar flux (F10.7), Kp index, and geomagnetic storm data to improve atmospheric density estimates for LEO decay predictions.

**Data Source Integration Plan:**
- NOAA Space Weather data via their public API (future, not in ASTRA Core)
- ASTRA Core will receive pre-fetched space weather data as function arguments (maintaining no-I/O rule)

```python
def propagate_with_drag(
    satellite: SatelliteTLE,
    time_steps: np.ndarray,
    f107_daily: float,     # Solar radio flux
    kp_index: float,       # Geomagnetic index
) -> TrajectoryMap:
    ...
```

---

## 10. Extension 9: Real-Time Streaming Support

**Status:** Planned, Phase 4

**Description:** Support streaming TLE updates into the analysis pipeline without restarting the full catalog load.

**Proposed Pattern:**
```python
def update_catalog(
    existing_catalog: list[SatelliteTLE],
    new_tles: list[SatelliteTLE],
) -> list[SatelliteTLE]:
    """Return updated catalog with new_tles replacing same NORAD IDs."""
```

**Current Compatibility:** Immutable `SatelliteTLE` objects make this trivial — replace objects by NORAD ID, return new list.

---

## 11. Architecture Stability Contract

The following elements of ASTRA Core are **permanent and will not change** in future extensions:

| Element | Permanent? | Note |
|---|---|---|
| Pure function design | ✅ Yes | All new modules must also be pure |
| `SatelliteTLE` fields | ✅ Yes | Only new optional fields may be added |
| `OrbitalState` fields | ✅ Yes | Position always TEME, km |
| `TrajectoryMap` type | ✅ Yes | Core data exchange type |
| No I/O in compute functions | ✅ Yes | Permanent rule |
| Filtering before propagation | ✅ Yes | Permanent performance principle |
| No O(n²) over full catalog | ✅ Yes | Permanent performance rule |
| `astra.errors` hierarchy | ✅ Yes | New exception classes may be added only |

---

## 12. Extension Integration Checklist

For any future module added to ASTRA Core, verify:

- [ ] Module is a pure function collection (no side effects)
- [ ] Module has a single, clearly defined responsibility
- [ ] Module does not import from modules at its same level or above in the dependency graph
- [ ] All data types use `astra.models` types or standard Python/NumPy types
- [ ] All new exception types inherit from `AstraError`
- [ ] Module has ≥ 90% test coverage before merge
- [ ] Module is documented with the same template as this documentation set
- [ ] `pyproject.toml` is updated with any new optional dependencies
- [ ] `astra/__init__.py` is updated to expose public API
