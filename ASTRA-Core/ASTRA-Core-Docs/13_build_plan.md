# 13 — ASTRA Core: Build Plan

---

## 1. Overview

This document defines the **exact build order** for implementing ASTRA Core from scratch. Every step depends on the previous steps being complete. This order is mandatory — later modules cannot function without earlier ones.

---

## 2. Prerequisites

Before writing any code:
1. Create virtual environment: `python -m venv .venv`
2. Activate: `.venv/Scripts/activate` (Windows) or `source .venv/bin/activate`
3. Install runtime dependencies: `pip install sgp4>=2.21 skyfield>=1.46 numpy>=1.24`
4. Install dev dependencies: `pip install pytest pytest-cov mypy ruff hypothesis`
5. Create project structure (see step below)

---

## 3. Step 0: Project Scaffolding

Create the following empty files:

```
astra/
├── __init__.py
├── errors.py
├── models.py
├── constants.py
├── time.py
├── tle.py
├── orbit.py
├── debris.py
├── conjunction.py
├── visibility.py
└── utils.py
tests/
├── conftest.py
├── fixtures/
│   └── sample_catalog.tle   ← real TLE fixture data
├── test_tle.py
├── test_orbit.py
├── test_debris.py
├── test_conjunction.py
├── test_visibility.py
├── test_time.py
├── test_utils.py
└── test_integration.py
pyproject.toml
ruff.toml
.pre-commit-config.yaml
```

**`pyproject.toml`** must be configured immediately (see `07_external_dependencies.md`).

---

## 4. Step 1: Constants and Errors (No Dependencies)

**Implement:** `astra/constants.py` and `astra/errors.py`

These have zero imports from the ASTRA package. They are the foundation everything else rests on.

**`constants.py`** — define all SIMULATION_STEPS, EARTH_RADIUS_KM, orbital regime boundaries, CONJUNCTION_THRESHOLD_KM, etc.

**`errors.py`** — Implement the full hierarchy: `AstraError`, `InvalidTLEError`, `PropagationError`, `FilterError`, `CoordinateError`. See `10_error_handling.md` for complete code.

**Verification:** `pytest tests/test_errors.py` — run basic instantiation and string representation tests.

---

## 5. Step 2: Data Models

**Implement:** `astra/models.py`

Define all frozen dataclasses: `SatelliteTLE`, `OrbitalState`, `DebrisObject`, `ConjunctionEvent`, `Observer`, `PassEvent`, `FilterConfig`.

Define type aliases: `TrajectoryMap`, `CandidatePairs`, `TimeArray`.

**Rules:**
- All dataclasses use `@dataclass(frozen=True)`
- Run `mypy astra/models.py --strict` — must pass clean

**Verification:** `pytest tests/test_models.py` — test that dataclasses are instantiable, frozen (mutation raises `FrozenInstanceError`), and equality works.

---

## 6. Step 3: Time Utilities

**Implement:** `astra/time.py`

Implement `convert_time()` supporting all four output formats: `"jd"`, `"datetime"`, `"skyfield"`, `"iso"`.

This function is needed by `astra.tle` for epoch parsing.

**Test:** Verify round-trip conversions:
- `convert_time("2025-01-01T00:00:00Z", "jd")` → known Julian Date value (2460676.5)
- Convert back to ISO and compare

**Verification:** `pytest tests/test_time.py`

---

## 7. Step 4: TLE Parsing (`astra.tle`)

**Implement:** `astra/tle.py`

### Order of implementation within `tle.py`:

1. `_compute_checksum(line)` — private helper
2. `_parse_epoch_to_jd(epoch_str)` — private helper, implements Algorithm 6 from `06_algorithm_design.md`
3. `validate_tle(name, line1, line2)` — returns bool
4. `parse_tle(name, line1, line2)` — raises `InvalidTLEError`
5. `_chunk_tle_lines(lines)` — private helper to chunk flat list into triplets
6. `load_tle_catalog(tle_lines)` — batch parser

**Key Implementation Detail for `_parse_epoch_to_jd`:**
```
YY < 57 → year = 2000 + YY
YY ≥ 57 → year = 1900 + YY
```

**Verification:** `pytest tests/test_tle.py` — all 15 test cases must pass.

---

## 8. Step 5: Math Utilities (`astra.utils`)

**Implement:** `astra/utils.py`

1. `haversine_distance()` — standard formula
2. `orbital_elements(line2)` — parse and return dict from TLE Line 2
3. `orbit_period()` — `(24*60) / mean_motion`

These are needed by `make_debris_object()` in `astra.debris`.

**Verification:** `pytest tests/test_utils.py`

---

## 9. Step 6: Orbit Propagation (`astra.orbit`)

**Implement:** `astra/orbit.py`

### Order within `orbit.py`:

1. `propagate_orbit()` — single object, single time
2. `propagate_many()` — batch using `sgp4_array()` — **CRITICAL: use C extension**
3. `propagate_trajectory()` — convenience wrapper over `propagate_many`
4. `ground_track()` — uses `skyfield` for TEME→geodetic conversion

**Critical Implementation Check:**
```python
# VERIFY this import works (C extension):
from sgp4.api import Satrec, sgp4_array
satrec = Satrec.twoline2rv(line1, line2)
# If this imports successfully, C extension is active
```

**Correctness Test:**
- Propagate ISS TLE at epoch + 0 minutes
- Compare to a known reference position (from NASA Horizons for the same epoch)
- Tolerance: 5 km (acceptable for SGP4 precision)

**Verification:** `pytest tests/test_orbit.py`

---

## 10. Step 7: Debris Filtering (`astra.debris`)

**Implement:** `astra/debris.py`

### Order:

1. `make_debris_object(tle)` — derives all orbital elements using `sgp4` satrec + `utils.orbital_elements()`
2. `filter_altitude()` — simple altitude range check
3. `filter_region()` — inclination-based geographic filter
4. `filter_time_window()` — TLE age check
5. `catalog_statistics()` — summary computation

**Key implementation for `make_debris_object`:**
- Initialize `sgp4.api.Satrec.twoline2rv(line1, line2)` to access `satrec.no` (mean motion), `satrec.ecco` (eccentricity), etc.
- Use `orbital_elements(line2)` for direct field parsing from TLE text

**Verification:** `pytest tests/test_debris.py`

---

## 11. Step 8: Conjunction Detection (`astra.conjunction`)

**Implement:** `astra/conjunction.py`

### Order:

1. `distance_3d()` — vectorized norm
2. `collision_probability()` — Chan method
3. `closest_approach()` — find min over time axis
4. `_classify_risk()` — private helper
5. `_coarse_filter_pairs()` — orbital element pre-filter (Phase 1 of conjunction algorithm)
6. `_build_candidate_pairs()` — spatial grid bucketing (Phase 2)
7. `find_conjunctions()` — main pipeline integrating all phases

**CRITICAL: Do NOT propagate inside `find_conjunctions()`. It operates ONLY on precomputed `TrajectoryMap`.**

**Verification:** `pytest tests/test_conjunction.py`

---

## 12. Step 9: Visibility Calculation (`astra.visibility`)

**Implement:** `astra/visibility.py`

### Order:

1. `visible_from_location()` — elevation angle array using skyfield
2. `_detect_passes()` — private: find contiguous above-horizon intervals
3. `passes_over_location()` — full pass prediction (AOS, TCA, LOS)

**Verification:** `pytest tests/test_visibility.py`

---

## 13. Step 10: Public API (`astra/__init__.py`)

Wire up all public exports as defined in `03_library_design.md` Section 4.

**Verification:**
```python
import astra
dir(astra)  # Verify expected names are present
```

---

## 14. Step 11: Integration Tests

**Run:** `pytest tests/test_integration.py`

This runs the full pipeline end-to-end:
1. Load TLE catalog from fixture
2. Filter
3. Propagate
4. Find conjunctions
5. Verify outputs are valid

---

## 15. Step 12: Optimization Review

After all tests pass:

1. **Profile `propagate_many()` on 1,000 objects:**
   ```bash
   python -m cProfile -s cumulative astra_benchmark.py
   ```
2. **Check `find_conjunctions()` performance against `N=500` trajectory set**
3. **Memory check:** Run `memory_profiler` on full pipeline
4. **Type check:** `mypy astra/ --strict` — zero errors required
5. **Lint:** `ruff check astra/` — zero errors required

---

## 16. Build Order Summary

```
Step 0: Scaffolding (files + venv + dependencies)
    ↓
Step 1: constants.py + errors.py
    ↓
Step 2: models.py
    ↓
Step 3: time.py
    ↓
Step 4: tle.py  ← needs: errors, models, time
    ↓
Step 5: utils.py  ← needs: nothing (numpy only)
    ↓
Step 6: orbit.py  ← needs: sgp4, skyfield, models, errors, time
    ↓
Step 7: debris.py  ← needs: models, errors, utils
    ↓
Step 8: conjunction.py  ← needs: models, utils (NO orbit, NO tle)
    ↓
Step 9: visibility.py  ← needs: orbit, models, time, skyfield
    ↓
Step 10: __init__.py  ← public API surface
    ↓
Step 11: Integration tests
    ↓
Step 12: Optimization + type check + lint
```

---

## 17. Estimated Implementation Timeline

| Step | Complexity | Estimate |
|---|---|---|
| 0: Scaffolding | Low | 30 min |
| 1: Constants + Errors | Low | 1 hour |
| 2: Models | Medium | 2 hours |
| 3: Time conversion | Medium | 1 hour |
| 4: TLE parsing | High (checksum, epoch) | 3 hours |
| 5: Utils | Low | 1 hour |
| 6: Orbit propagation | High (sgp4 API, sgp4_array) | 4 hours |
| 7: Debris filtering | Medium | 2 hours |
| 8: Conjunction detection | Very High (multi-phase algorithm) | 6 hours |
| 9: Visibility | High (skyfield integration) | 3 hours |
| 10: Public API | Low | 30 min |
| 11: Integration tests | Medium | 2 hours |
| 12: Optimization | Medium | 2 hours |
| **Total** | | **~30 hours** |
