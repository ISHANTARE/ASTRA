# 09 — ASTRA Core: API Design (Internal Function API)

---

## 1. Purpose of This Document

This document defines the complete **internal function API** of ASTRA Core — the callable interface that backend developers and AI agents will use to integrate ASTRA Core into larger systems. It is a reference of all public callables with their full signatures.

This is NOT a REST API specification. ASTRA Core has no HTTP interface.

---

## 2. Calling Conventions

### 2.1 Imports

All public functions are importable from the top-level `astra` package:

```python
import astra

# Or explicit imports:
from astra import parse_tle, propagate_many, find_conjunctions
from astra.models import SatelliteTLE, DebrisObject, Observer
from astra.errors import InvalidTLEError, PropagationError
```

### 2.2 Return Value Policy

- Functions return typed values only (no `None` on success)
- Empty results return empty collections (`[]`, `{}`) not `None`
- Failed operations raise typed exceptions, never return `None`

### 2.3 Time Convention

All time values passed to functions are **Julian Dates (float)**. Never passing `datetime` objects directly into computational functions — use `astra.time.convert_time()` first.

```python
from astra.time import convert_time

t_start_jd = convert_time("2025-01-01T00:00:00Z", to_format="jd")
t_end_jd   = convert_time("2025-01-02T00:00:00Z", to_format="jd")
```

---

## 3. Complete API Reference

### 3.1 `astra.tle` — TLE Parsing

```python
# Parse a single TLE
def parse_tle(name: str, line1: str, line2: str) -> SatelliteTLE

# Validate (no exception, returns bool)
def validate_tle(name: str, line1: str, line2: str) -> bool

# Parse full catalog from raw text lines
def load_tle_catalog(tle_lines: list[str]) -> list[SatelliteTLE]
```

**Usage Example:**

```python
raw_lines = celestrak_response_text.splitlines()
catalog: list[SatelliteTLE] = astra.load_tle_catalog(raw_lines)
print(f"Loaded {len(catalog)} satellites")
```

---

### 3.2 `astra.orbit` — Propagation

```python
# Single satellite, single time
def propagate_orbit(
    satellite: SatelliteTLE,
    epoch_jd: float,
    t_since_minutes: float
) -> OrbitalState

# Batch: N satellites × T timesteps
def propagate_many(
    satellites: list[SatelliteTLE],
    time_steps: np.ndarray   # (T,) minutes since each sat's epoch
) -> TrajectoryMap           # dict[norad_id, ndarray(T,3)]

# Single satellite over time range
def propagate_trajectory(
    satellite: SatelliteTLE,
    t_start_jd: float,
    t_end_jd: float,
    step_minutes: float = 5.0
) -> tuple[np.ndarray, np.ndarray]   # (times_jd(T,), positions(T,3))

# Convert TEME positions to lat/lon ground track
def ground_track(
    positions_teme: np.ndarray,  # (T, 3)
    times_jd: np.ndarray         # (T,)
) -> list[tuple[float, float]]   # [(lat, lon), ...]
```

**Usage Example:**

```python
import numpy as np
from astra import propagate_many

time_steps = np.arange(0, 24*60, 5)  # 0 to 1440 minutes, every 5 min = 288 steps
trajectories = propagate_many(survivors, time_steps)
# trajectories["25544"]  →  shape (288, 3), km, TEME
```

---

### 3.3 `astra.debris` — Filtering

```python
# Filter by altitude band
def filter_altitude(
    objects: list[DebrisObject],
    min_km: float,
    max_km: float
) -> list[DebrisObject]

# Filter by geographic region
def filter_region(
    objects: list[DebrisObject],
    lat_min_deg: float,
    lat_max_deg: float,
    lon_min_deg: float,
    lon_max_deg: float
) -> list[DebrisObject]

# Filter by TLE recency
def filter_time_window(
    objects: list[DebrisObject],
    t_start_jd: float,
    t_end_jd: float
) -> list[DebrisObject]

# Compute catalog statistics
def catalog_statistics(
    objects: list[DebrisObject]
) -> dict[str, Any]

# Factory: build DebrisObject from SatelliteTLE
def make_debris_object(tle: SatelliteTLE) -> DebrisObject
```

**Usage Example (Full Pipeline):**

```python
from astra import load_tle_catalog
from astra.debris import make_debris_object, filter_altitude, filter_region, filter_time_window, catalog_statistics

# Load
tles = load_tle_catalog(raw_text_lines)
catalog = [make_debris_object(t) for t in tles]

# Stats before
stats = catalog_statistics(catalog)

# Filter chain
leo = filter_altitude(catalog, min_km=160, max_km=2000)
indian_region = filter_region(leo, lat_min_deg=8.0, lat_max_deg=37.0,
                               lon_min_deg=68.0, lon_max_deg=97.0)
current = filter_time_window(indian_region, t_start_jd=t_start, t_end_jd=t_end)

print(f"{len(catalog)} → {len(leo)} → {len(indian_region)} → {len(current)}")
```

---

### 3.4 `astra.conjunction` — Conjunction Detection

```python
# Vectorized 3D distance
def distance_3d(
    pos_a: np.ndarray,  # (T,3) or (3,)
    pos_b: np.ndarray   # (T,3) or (3,)
) -> np.ndarray         # (T,) or scalar

# Closest approach between two trajectories
def closest_approach(
    trajectory_a: np.ndarray,  # (T,3)
    trajectory_b: np.ndarray,  # (T,3)
    times_jd: np.ndarray       # (T,)
) -> tuple[float, float, int]  # (min_dist_km, tca_jd, t_index)

# Full conjunction detection pipeline
def find_conjunctions(
    trajectories: TrajectoryMap,   # dict[norad_id, ndarray(T,3)]
    times_jd: np.ndarray,          # (T,)
    threshold_km: float = 5.0,
    coarse_threshold_km: float = 50.0
) -> list[ConjunctionEvent]

# Collision probability (Chan method)
def collision_probability(
    miss_distance_km: float,
    relative_velocity_km_s: float,
    combined_radius_m: float = 10.0
) -> float
```

**Usage Example:**

```python
from astra import find_conjunctions

events = find_conjunctions(
    trajectories=trajectory_map,
    times_jd=times_jd_array,
    threshold_km=5.0
)

for event in events:
    print(f"{event.object_a_id} ↔ {event.object_b_id}: "
          f"{event.miss_distance_km:.2f} km, risk={event.risk_level}")
```

---

### 3.5 `astra.visibility` — Pass Prediction

```python
# Elevation angles at each timestep
def visible_from_location(
    positions_teme: np.ndarray,  # (T,3) km TEME
    times_jd: np.ndarray,        # (T,)
    observer: Observer
) -> np.ndarray                  # (T,) elevation angles in degrees

# Detect complete passes (AOS/TCA/LOS)
def passes_over_location(
    satellite: SatelliteTLE,
    observer: Observer,
    t_start_jd: float,
    t_end_jd: float,
    step_minutes: float = 1.0
) -> list[PassEvent]
```

**Usage Example:**

```python
from astra import passes_over_location
from astra.models import Observer

iss_tle = parse_tle(name, line1, line2)
bangalore = Observer(
    name="Bangalore",
    latitude_deg=12.97,
    longitude_deg=77.59,
    elevation_m=920.0,
    min_elevation_deg=10.0
)

passes = passes_over_location(iss_tle, bangalore, t_start_jd, t_end_jd)
for p in passes:
    print(f"AOS: {p.aos_jd}, Max El: {p.max_elevation_deg:.1f}°, Duration: {p.duration_seconds:.0f}s")
```

---

### 3.6 `astra.time` — Time Conversion

```python
# Universal time converter
def convert_time(
    value: str | datetime | float,
    to_format: str  # "jd", "datetime", "skyfield", "iso"
) -> float | datetime | skyfield.timelib.Time | str
```

---

### 3.7 `astra.utils` — Utilities

```python
# Great-circle distance between two lat/lon points
def haversine_distance(
    lat1_deg: float, lon1_deg: float,
    lat2_deg: float, lon2_deg: float
) -> float  # km

# Extract Keplerian elements from TLE Line 2
def orbital_elements(line2: str) -> dict[str, float]

# Orbital period from mean motion
def orbit_period(mean_motion_rev_per_day: float) -> float  # minutes
```

---

## 4. Complete End-to-End Usage Example

```python
import numpy as np
import astra
from astra.models import Observer
from astra.time import convert_time
from astra.debris import make_debris_object, filter_altitude, filter_time_window

# === 1. LOAD CATALOG ===
with open("catalog.tle") as f:
    raw_lines = f.read().splitlines()

tles = astra.load_tle_catalog(raw_lines)
print(f"Loaded {len(tles)} objects")

# === 2. BUILD DEBRIS OBJECTS ===
catalog = [make_debris_object(t) for t in tles]

# === 3. FILTER ===
t_start = convert_time("2025-06-01T00:00:00Z", to_format="jd")
t_end   = convert_time("2025-06-02T00:00:00Z", to_format="jd")

leo      = filter_altitude(catalog, min_km=200, max_km=2000)
current  = filter_time_window(leo, t_start_jd=t_start, t_end_jd=t_end)
print(f"Survivors after filtering: {len(current)}")

# === 4. PROPAGATE SURVIVORS ONLY ===
time_steps = np.arange(0, 24 * 60, 5.0)  # shape (288,)
trajectories = astra.propagate_many(
    [obj.tle for obj in current],
    time_steps
)
times_jd = t_start + time_steps / 1440.0  # Convert to JD array

# === 5. FIND CONJUNCTIONS ===
events = astra.find_conjunctions(
    trajectories=trajectories,
    times_jd=times_jd,
    threshold_km=5.0
)
print(f"Conjunction events found: {len(events)}")
for e in events[:5]:
    print(f"  {e.object_a_id} ↔ {e.object_b_id}: {e.miss_distance_km:.2f} km [{e.risk_level}]")
```

---

## 5. Error Handling in Calling Code

```python
from astra.errors import InvalidTLEError, PropagationError, AstraError

try:
    tle = astra.parse_tle(name, line1, line2)
except InvalidTLEError as e:
    print(f"Bad TLE for {name}: {e}")

try:
    trajectories = astra.propagate_many(tles, time_steps)
except PropagationError as e:
    print(f"Propagation failed: {e}")
except AstraError as e:
    print(f"ASTRA error: {e}")
```
