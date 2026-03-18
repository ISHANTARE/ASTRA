# 02 — ASTRA Core: Architecture

---

## 1. Architectural Position

ASTRA Core occupies the **library layer** of the full ASTRA system stack:

```
┌──────────────────────────────────────────────────┐
│                FRONTEND UI LAYER                 │
│    React · Three.js · WebGL (Future Layer)       │
├──────────────────────────────────────────────────┤
│               BACKEND API LAYER                  │
│           FastAPI / Flask (Future Layer)         │
│    Responsible for: HTTP routing, auth, I/O      │
├══════════════════════════════════════════════════╡
║             ASTRA CORE LIBRARY                   ║  ← THIS SYSTEM
║    Pure Python · No web dependencies             ║
║    Responsible for: computation orchestration    ║
╠══════════════════════════════════════════════════╣
│            EXTERNAL DEPENDENCIES                 │
│       sgp4   │   skyfield   │   numpy            │
└──────────────────────────────────────────────────┘
```

**Contract:** The Backend API layer calls ASTRA Core functions passing plain Python data structures. ASTRA Core returns plain Python data structures. No HTTP objects, database connections, or file handles ever cross this boundary.

---

## 2. Internal Module Architecture

ASTRA Core is organized as a single Python package (`astra`) with seven sub-modules:

```
astra/
├── __init__.py          # Public API surface exports
├── tle.py               # TLE parsing and validation
├── orbit.py             # SGP4 propagation, trajectory generation
├── debris.py            # Debris catalog filtering and statistics
├── conjunction.py       # Conjunction detection and collision probability
├── visibility.py        # Ground station pass prediction
├── time.py              # Time conversion utilities
└── utils.py             # General math and geometry helpers
└── errors.py            # Custom exception classes
```

### Module Dependency Graph

```
          ┌──────────┐
          │  errors  │ (no dependencies — used by all)
          └────┬─────┘
               │
    ┌──────────▼──────────┐
    │       time.py       │ (depends on: skyfield, datetime)
    └──────────┬──────────┘
               │
    ┌──────────▼──────────┐
    │       tle.py        │ (depends on: sgp4, time)
    └──────────┬──────────┘
               │
    ┌──────────▼──────────┐
    │      orbit.py       │ (depends on: sgp4, skyfield, numpy, tle, time)
    └──────────┬──────────┘
               │
        ┌──────┴──────┐
        │             │
┌───────▼───────┐  ┌──▼──────────────┐
│   debris.py   │  │  visibility.py  │
│ (→ orbit)     │  │  (→ orbit,time) │
└───────┬───────┘  └─────────────────┘
        │
┌───────▼──────────┐
│  conjunction.py  │
│  (→ orbit,utils) │
└──────────────────┘
        │
┌───────▼──────────┐
│    utils.py      │ (depends on: numpy only)
└──────────────────┘
```

**Rule:** Lower-level modules never import from higher-level modules. `tle.py` never imports from `orbit.py`. `utils.py` never imports from any domain module.

---

## 3. Data Flow Architecture

### 3.1 Orbit Propagation Flow

```
Input: SatelliteTLE, time_range
        │
        ▼
tle.parse_tle() → SatelliteTLE object
        │
        ▼
time.convert_time() → skyfield Time objects
        │
        ▼
orbit.propagate_orbit() → OrbitalState
        │
        ▼
Output: OrbitalState (TEME position/velocity + metadata)
```

### 3.2 Multi-Stage Debris Filtering Flow

```
Input: List[DebrisObject], FilterConfig
        │
        ▼
Stage 1: debris.filter_altitude()
         (Eliminates objects outside altitude band)
         │
         ▼
Stage 2: debris.filter_region()
         (Eliminates objects outside geographic region)
         │
         ▼
Stage 3: debris.filter_time_window()
         (Eliminates objects with no activity in window)
         │
         ▼
Stage 4: orbit.propagate_many()
         (Propagate ONLY surviving objects)
         │
         ▼
Output: Dict[str, np.ndarray]  ← trajectory arrays
```

### 3.3 Conjunction Detection Flow

```
Input: Dict[str, np.ndarray] (precomputed trajectories)
        │
        ▼
Stage 1: conjunction.find_conjunctions() coarse filter
         (Orbital element proximity: semi-major axis, inclination)
         (→ eliminates 90%+ of pairs without propagation)
         │
         ▼
Stage 2: Candidate pair generation
         (Spatial grid bucketing)
         │
         ▼
Stage 3: conjunction.closest_approach()
         (Vectorized distance computation over time axis)
         │
         ▼
Stage 4: conjunction.collision_probability()
         (Chan method for qualifying events only)
         │
         ▼
Output: List[ConjunctionEvent]
```

### 3.4 Visibility Flow

```
Input: SatelliteTLE, Observer, time_range
        │
        ▼
orbit.propagate_trajectory() → positions over time
        │
        ▼
visibility.visible_from_location()
  (topocentric transforms, elevation angle computation)
        │
        ▼
visibility.passes_over_location()
  (rise/peak/set detection from elevation curve)
        │
        ▼
Output: List[PassEvent]
```

---

## 4. Computation Orchestration Model

ASTRA Core follows a **pull-based lazy evaluation** model:

1. **Filter first**: No propagation occurs until filtering is complete
2. **Compute once**: Trajectories are precomputed once per object, stored in memory, reused
3. **Batch compute**: NumPy vectorized operations across all objects/timesteps simultaneously
4. **Short-circuit**: Any filtering stage can return an empty set, terminating the pipeline immediately

```python
# Correct orchestration pattern (conceptual):
candidates = filter_altitude(catalog, min_km=200, max_km=1200)
candidates = filter_region(candidates, lat_range, lon_range)
candidates = filter_time_window(candidates, t_start, t_end)

# Only now do we propagate:
trajectories = propagate_many(candidates, time_steps)

# Conjunction analysis uses ONLY precomputed trajectories:
events = find_conjunctions(trajectories, threshold_km=5.0)
```

---

## 5. Module Boundary Contracts

Each module has a strict input/output contract using defined data model types:

| Module | Accepts | Returns |
|---|---|---|
| `astra.tle` | Raw TLE strings | `SatelliteTLE` objects |
| `astra.orbit` | `SatelliteTLE`, time arrays | `OrbitalState`, trajectory arrays |
| `astra.debris` | `List[DebrisObject]`, filter params | Filtered `List[DebrisObject]` |
| `astra.conjunction` | Trajectory arrays | `List[ConjunctionEvent]` |
| `astra.visibility` | `SatelliteTLE`, `Observer`, time range | `List[PassEvent]` |
| `astra.time` | Datetime strings, floats, epochs | Standardized time objects |
| `astra.utils` | Coordinates, orbital elements | Computed scalar/vector values |

---

## 6. Prohibited Architectural Patterns

The following patterns are **strictly forbidden** in ASTRA Core:

| Pattern | Reason |
|---|---|
| Import of `flask`, `fastapi`, `django` | Web layer must remain separate |
| File I/O (`open()`, `json.load()`) inside computation functions | No side effects in pure functions |
| Database calls (`psycopg2`, `SQLAlchemy`) | Data access is the API layer's responsibility |
| Module-level mutable state (`catalog = []`, `_cache = {}`) | Breaks determinism and testability |
| O(n²) loops over full catalog without pre-filtering | Computationally infeasible at scale |
| Propagation inside pairwise comparison loops | Each object must be propagated exactly once |
| Manual orbital mechanics implementations | Use `sgp4`/`skyfield` exclusively |

---

## 7. Threading and Concurrency Model

ASTRA Core is designed as **single-threaded, pure-function based**. Parallelism is the caller's responsibility.

- All functions are stateless and safe to call from multiple threads simultaneously
- No shared mutable objects exist at module level
- NumPy operations internally use BLAS/LAPACK multi-threading (transparent optimization)
- Future: callers may use `concurrent.futures.ProcessPoolExecutor` to parallelize `propagate_many` across object batches

---

## 8. Coordinate Reference Frames

| Frame | Usage | Conversion |
|---|---|---|
| **TEME** (True Equator Mean Equinox) | SGP4 native output | Converted by `skyfield` |
| **GCRS/ECI** (J2000) | Internal analysis frame | Via `skyfield.api` GCRS |
| **ITRS/ECEF** | Earth-fixed frame for visibility | Via `skyfield` ITRS |
| **Topocentric (AzEl)** | Observer-relative visibility | Via `skyfield` observer method |

**Rule:** All inter-module position vectors are in **GCRS/ECI km** unless explicitly documented otherwise.
