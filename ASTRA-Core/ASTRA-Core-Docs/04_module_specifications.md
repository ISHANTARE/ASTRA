# 04 — ASTRA Core: Module Specifications

---

## Overview

This document defines each module's **responsibility, function signatures, and boundary contracts**.
Every function listed here is part of the public API. Private helpers are prefixed with `_`.

---

## Module 1: `astra.tle`

### Responsibility

Parse, validate, and batch-load Two-Line Element (TLE) sets from raw text. This module is the **entry point** for all data entering ASTRA Core. It never performs orbital computation.

### Boundaries

| Imports From | Never Imports From |
|---|---|
| `astra.errors`, `astra.models` | `astra.orbit`, `astra.debris`, `astra.conjunction` |

### Public Functions

---

#### `parse_tle`

```python
def parse_tle(name: str, line1: str, line2: str) -> SatelliteTLE:
```

**Purpose:** Parse three raw TLE lines into a validated `SatelliteTLE` object.

**Inputs:**
- `name`: Object name string (up to 24 characters, stripped of leading/trailing whitespace)
- `line1`: TLE line 1 exactly 69 characters (stripped, no newline characters)
- `line2`: TLE line 2 exactly 69 characters (stripped, no newline characters)

**Outputs:**
- A fully populated `SatelliteTLE` instance on success

**Logic Steps:**
1. Strip all whitespace from `name`, `line1`, `line2`
2. Verify `line1` starts with `'1'`; raise `InvalidTLEError` if not
3. Verify `line2` starts with `'2'`; raise `InvalidTLEError` if not
4. Verify both lines are exactly 69 characters; raise `InvalidTLEError` if not
5. Verify line1 checksum (sum of digits mod 10 == field 69 value)
6. Verify line2 checksum
7. Extract NORAD ID: line1[2:7].strip()
8. Extract epoch: line1[18:32] → convert to Julian Date
9. Extract object_type from line1 classification character (field 1.2):
   - `'U'` → `"UNKNOWN"`, `'C'` → `"PAYLOAD"`, `'D'` → `"DEBRIS"`, `'R'` → `"ROCKET_BODY"`
10. Instantiate and return `SatelliteTLE`

**Raises:**
- `InvalidTLEError` with field `invalid_line` set to the offending line if any validation fails

---

#### `validate_tle`

```python
def validate_tle(name: str, line1: str, line2: str) -> bool:
```

**Purpose:** Non-destructive validation of TLE strings. Returns `True` if valid, `False` if not. Does NOT raise exceptions.

**Inputs:** Same as `parse_tle`

**Outputs:** `bool` — `True` if TLE is well-formed and checksums pass

**Logic:** Identical to `parse_tle` validation steps 1–8 but catches all `InvalidTLEError` internally and returns `False`.

---

#### `load_tle_catalog`

```python
def load_tle_catalog(tle_lines: list[str]) -> list[SatelliteTLE]:
```

**Purpose:** Parse a batch of TLE text lines (as typically received from CelesTrak API responses) into a list of `SatelliteTLE` objects. Invalid TLEs are **skipped with a logged warning**, not exception-raised.

**Inputs:**
- `tle_lines`: A flat list of strings, each being one line of a multi-TLE text block. The standard format is triplets: name, line1, line2.

**Outputs:**
- `list[SatelliteTLE]`: Successfully parsed objects only. Invalid TLEs are skipped.

**Logic Steps:**
1. Group `tle_lines` into triplets using `_chunk_tle_lines()`
2. For each triplet `(name, line1, line2)`:
   a. Call `validate_tle(name, line1, line2)`
   b. If valid: call `parse_tle()` and append to result
   c. If invalid: log warning with NORAD ID, skip
3. Return accumulated list
4. If result is empty and input was non-empty, raise `AstraError` (total parse failure)

---

## Module 2: `astra.orbit`

### Responsibility

SGP4-based orbit propagation. Generates position/velocity vectors at specified times. All computation uses `sgp4` and `skyfield`. No manual orbital equations.

### Boundaries

| Imports From | Never Imports From |
|---|---|
| `sgp4`, `skyfield`, `numpy`, `astra.models`, `astra.errors`, `astra.time` | `astra.debris`, `astra.conjunction` |

### Public Functions

---

#### `propagate_orbit`

```python
def propagate_orbit(
    satellite: SatelliteTLE,
    epoch_jd: float,
    t_since_minutes: float,
) -> OrbitalState:
```

**Purpose:** Propagate a single satellite to a single point in time using SGP4.

**Inputs:**
- `satellite`: Parsed `SatelliteTLE`
- `epoch_jd`: Reference epoch as Julian Date (typically `satellite.epoch_jd`)
- `t_since_minutes`: Minutes elapsed since epoch

**Outputs:** `OrbitalState` with position (km), velocity (km/s) in TEME frame

**Logic Steps:**
1. Initialize `sgp4` satellite object from `satellite.line1`, `satellite.line2` using `sgp4.twoline2rv()`
2. Compute absolute time: `t_jd = epoch_jd + t_since_minutes / 1440.0`
3. Call `satrec.sgp4(jd, fr)` where `jd`, `fr` = integer and fractional Julian Date
4. If `error_code != 0`, set `OrbitalState.error_code` accordingly (do NOT raise — caller decides)
5. Return `OrbitalState(norad_id, t_jd, position_km, velocity_km_s, error_code)`

---

#### `propagate_many`

```python
def propagate_many(
    satellites: list[SatelliteTLE],
    time_steps: np.ndarray,   # shape (T,), dtype float64, unit: minutes since each sat's epoch
) -> TrajectoryMap:           # dict[norad_id, ndarray shape (T, 3)]
```

**Purpose:** Vectorized batch propagation of multiple satellites across a time array. This is the **primary computation function** for large-scale operations.

**Inputs:**
- `satellites`: List of `SatelliteTLE` objects to propagate
- `time_steps`: 1D NumPy array of `T` time offsets in minutes since each satellite's own epoch

**Outputs:**
- `TrajectoryMap`: Dictionary mapping `norad_id → np.ndarray shape (T, 3)` in TEME frame, km.
- Satellites with propagation errors at any timestep store `np.nan` at that timestep row.

**Logic Steps:**
1. Initialize a `dict` to collect results
2. For each `satellite` in `satellites`:
   a. Initialize `sgp4` satrec from `satellite.line1/line2`
   b. Convert `time_steps` to Julian dates: `jd_array = satellite.epoch_jd + time_steps / 1440.0`
   c. Call `sgp4.sgp4_array(satrec, jd_array, jd_fraction_array)` for vectorized propagation
   d. Stack `[x, y, z]` columns into shape `(T, 3)` array
   e. Set rows with error_codes > 0 to `np.nan`
   f. Store in dict under `satellite.norad_id`
3. Return the collected dict

**Performance Note:** `sgp4_array` from the `sgp4` library is the vectorized propagation call that avoids Python-level loops over time steps. This is mandatory — do NOT loop over time steps manually.

---

#### `propagate_trajectory`

```python
def propagate_trajectory(
    satellite: SatelliteTLE,
    t_start_jd: float,
    t_end_jd: float,
    step_minutes: float = 5.0,
) -> tuple[np.ndarray, np.ndarray]:
    # Returns: (time_array shape (T,), positions shape (T, 3))
```

**Purpose:** Propagate a single satellite over a defined time window at a fixed step. Convenience function for analysis over a time range.

**Inputs:**
- `satellite`: Source `SatelliteTLE`
- `t_start_jd`: Window start as Julian Date
- `t_end_jd`: Window end as Julian Date
- `step_minutes`: Step size (default 5.0 minutes = ASTRA standard)

**Outputs:**
- Tuple of `(time_array, position_array)`:
  - `time_array`: shape `(T,)`, Julian Dates for each step
  - `position_array`: shape `(T, 3)`, TEME positions in km

**Logic:**
1. Generate `time_steps` in minutes from epoch: `np.arange(start_offset, end_offset, step_minutes)`
2. Delegate to `propagate_many([satellite], time_steps)`
3. Extract and return this satellite's trajectory

---

#### `ground_track`

```python
def ground_track(
    positions_teme: np.ndarray,   # shape (T, 3), km, TEME
    times_jd: np.ndarray,         # shape (T,), Julian Dates
) -> list[tuple[float, float]]:   # [(lat_deg, lon_deg), ...]
```

**Purpose:** Convert TEME Cartesian positions into geodetic (latitude, longitude) coordinates for ground track visualization.

**Inputs:**
- `positions_teme`: TEME position array from propagation
- `times_jd`: Corresponding Julian Date array

**Outputs:** List of `(latitude_deg, longitude_deg)` tuples, length T

**Logic Steps:**
1. Use `skyfield` to build a `TEME` position for each timestep
2. Use `skyfield.api` to convert TEME → ITRS/ECEF
3. Use `skyfield.api.wgs84.subpoint()` to extract geodetic coordinates
4. Return list of `(lat.degrees, lon.degrees)` tuples

---

## Module 3: `astra.debris`

### Responsibility

Pre-propagation filtering of debris catalogs. ALL filtering in this module operates on `DebrisObject` parameters (derived TLE fields, NOT propagated positions). Filtering is O(n) and involves no SGP4 calls.

### Boundaries

| Imports From | Never Imports From |
|---|---|
| `astra.models`, `astra.errors` | `astra.orbit`, `sgp4`, `skyfield` |

### Public Functions

---

#### `filter_altitude`

```python
def filter_altitude(
    objects: list[DebrisObject],
    min_km: float,
    max_km: float,
) -> list[DebrisObject]:
```

**Purpose:** Retain only objects whose mean orbital altitude falls within `[min_km, max_km]`.

**Logic:**
- Condition: `min_km <= obj.altitude_km <= max_km`
- Checks `obj.altitude_km` (mean of apogee and perigee altitudes)
- Also filters objects where `obj.perigee_km < min_km * 0.9` (will re-enter before analysis window — configurable tolerance)
- Returns filtered list preserving original order

---

#### `filter_region`

```python
def filter_region(
    objects: list[DebrisObject],
    lat_min_deg: float,
    lat_max_deg: float,
    lon_min_deg: float,
    lon_max_deg: float,
) -> list[DebrisObject]:
```

**Purpose:** Retain only objects whose ground track is expected to pass through a geographic bounding box during the analysis window.

**Logic Steps (approximation, no propagation):**
1. For each object, compute the latitude range accessible given inclination:
   - Object can reach latitudes `[-inclination_deg, +inclination_deg]`
2. If the region's latitude range `[lat_min_deg, lat_max_deg]` does NOT overlap with `[-incl, +incl]`, reject
3. Longitude filtering: All inclined orbits eventually pass all longitudes within their accessible latitude band, so:
   - For non-Sun-synchronous objects (incl < 90°): longitude filtering is not applied (too conservative)
   - For high-inclination or near-polar objects: **check only** if polar coverage region matches
4. Return filtered list

**Note:** This is an **over-inclusive** filter — it never falsely rejects objects that could pass the region. False positives are acceptable; false negatives are not.

---

#### `filter_time_window`

```python
def filter_time_window(
    objects: list[DebrisObject],
    t_start_jd: float,
    t_end_jd: float,
) -> list[DebrisObject]:
```

**Purpose:** Eliminate objects whose TLE epoch is too stale to produce reliable predictions within the time window.

**Logic Steps:**
1. Compute age of TLE: `age_days = t_start_jd - obj.tle.epoch_jd`
2. Reject objects where `age_days > 14.0` (TLE older than 2 weeks — configurable)
3. For LEO objects (`altitude_km < 2000`): apply stricter threshold of `age_days > 7.0`
4. Return filtered list

---

#### `catalog_statistics`

```python
def catalog_statistics(
    objects: list[DebrisObject],
) -> dict[str, Any]:
```

**Purpose:** Compute summary statistics across a debris catalog for analysis and reporting.

**Returns dict with:**
- `total_count`: int
- `by_type`: dict mapping object_class → count
- `by_regime`: dict with keys `"LEO"`, `"MEO"`, `"GEO"`, `"HEO"` → count
- `altitude_mean_km`: float
- `altitude_std_km`: float
- `altitude_min_km`: float
- `altitude_max_km`: float
- `inclination_distribution`: dict with keys `"equatorial"` (0–10°), `"inclined"` (10–80°), `"polar"` (80–90°), `"retrograde"` (90–180°) → count

---

## Module 4: `astra.conjunction`

### Responsibility

Detect close-approach events between pairs of orbital objects using precomputed trajectory arrays. **Never performs propagation.** Uses only trajectories provided by the caller.

### Boundaries

| Imports From | Never Imports From |
|---|---|
| `numpy`, `astra.models`, `astra.utils`, `astra.errors` | `astra.tle`, `astra.orbit`, `astra.debris`, `sgp4`, `skyfield` |

### Public Functions

---

#### `distance_3d`

```python
def distance_3d(
    pos_a: np.ndarray,    # shape (T, 3) or (3,)
    pos_b: np.ndarray,    # shape (T, 3) or (3,)
) -> np.ndarray:          # shape (T,) or scalar
```

**Purpose:** Compute Euclidean distance between two position arrays (or single points). Vectorized.

**Logic:** `np.linalg.norm(pos_a - pos_b, axis=-1)`

---

#### `closest_approach`

```python
def closest_approach(
    trajectory_a: np.ndarray,   # shape (T, 3)
    trajectory_b: np.ndarray,   # shape (T, 3)
    times_jd: np.ndarray,       # shape (T,)
) -> tuple[float, float, int]:  # (min_distance_km, time_of_ca_jd, t_index)
```

**Purpose:** Find the minimum separation between two object trajectories over a time window.

**Logic Steps:**
1. Compute distances: `distances = distance_3d(trajectory_a, trajectory_b)` → shape `(T,)`
2. Find minimum: `t_idx = np.argmin(distances)`
3. `min_dist = distances[t_idx]`, `tca_jd = times_jd[t_idx]`
4. Return `(min_dist, tca_jd, t_idx)`

---

#### `find_conjunctions`

```python
def find_conjunctions(
    trajectories: TrajectoryMap,        # dict[norad_id → ndarray(T,3)]
    times_jd: np.ndarray,               # shape (T,)
    threshold_km: float = 5.0,
    coarse_threshold_km: float = 50.0,
) -> list[ConjunctionEvent]:
```

**Purpose:** Find all conjunction events in a set of precomputed trajectories. Implements multi-stage filtering to avoid O(n²) full pairwise comparison. See `06_algorithm_design.md` for complete algorithm.

**Logic (high-level — see algorithm doc for full detail):**
1. **Coarse filter**: For each pair, compute max possible separation using orbital element bounding box. Pairs with minimum possible separation > `coarse_threshold_km * 2` are rejected without computing distances.
2. **Spatial grid pre-filter**: At each timestep, bucket objects into 3D altitude/angle grid cells. Only pairs sharing a cell or adjacent cells are candidates.
3. **Fine computation**: For each candidate pair, run `closest_approach()`.
4. **Threshold check**: Retain pairs where `min_distance_km <= threshold_km`.
5. **Event construction**: Compute relative velocity, collision probability, risk level.
6. Return `List[ConjunctionEvent]` sorted by `miss_distance_km` ascending.

---

#### `collision_probability`

```python
def collision_probability(
    miss_distance_km: float,
    relative_velocity_km_s: float,
    combined_radius_m: float = 10.0,
) -> float:
```

**Purpose:** Compute collision probability using the Chan analytical method.

**Inputs:**
- `miss_distance_km`: Minimum miss distance from `closest_approach()`
- `relative_velocity_km_s`: Relative speed at TCA
- `combined_radius_m`: Combined hard-body radius (default 10 m for debris-debris)

**Returns:** Probability as float in `[0.0, 1.0]`

**Logic (Chan method):**
1. Convert miss distance to meters: `r_miss = miss_distance_km * 1000`
2. Combined volume radius: R_c = `combined_radius_m`
3. Sigma = assumed covariance (mission-dependent; default: `sigma = miss_distance_km * 1000 * 0.2` — 20% of miss distance as position uncertainty)
4. P_c = `(R_c² / (sigma² + R_c²)) * exp(-r_miss² / (2*(sigma² + R_c²)))`
5. Clamp result to `[0.0, 1.0]` and return

---

## Module 5: `astra.visibility`

### Responsibility

Compute satellite passes over ground observer locations. Uses `skyfield` for topocentric coordinate transformations.

### Boundaries

| Imports From | Never Imports From |
|---|---|
| `skyfield`, `numpy`, `astra.models`, `astra.orbit`, `astra.time` | `astra.debris`, `astra.conjunction` |

### Public Functions

---

#### `visible_from_location`

```python
def visible_from_location(
    positions_teme: np.ndarray,    # shape (T, 3), km, TEME
    times_jd: np.ndarray,          # shape (T,), Julian Dates
    observer: Observer,
) -> np.ndarray:                   # shape (T,), elevation angles in degrees
```

**Purpose:** Compute topocentric elevation angle of the satellite at each time step, as seen from the observer's location.

**Inputs:** TEME trajectory and corresponding Julian Dates, plus observer definition

**Outputs:** Array of elevation angles (degrees). Negative = below horizon.

**Logic Steps:**
1. Use `skyfield.api.load.timescale()` to build a time object from `times_jd`
2. Build skyfield `EarthSatellite` from the satellite's TLE (reconstructed from trajectory source)
3. Build skyfield `wgs84.latlon(lat, lon, elevation_m)` observer
4. Compute `(sat - observer).at(t)` for all times using skyfield's `difference` method
5. Extract `alt, az, dist` from `altaz()` call
6. Return `alt.degrees` as NumPy array

---

#### `passes_over_location`

```python
def passes_over_location(
    satellite: SatelliteTLE,
    observer: Observer,
    t_start_jd: float,
    t_end_jd: float,
    step_minutes: float = 1.0,
) -> list[PassEvent]:
```

**Purpose:** Detect all distinct passes (AOS, TCA, LOS) of a satellite over a ground observer within a time window.

**Logic Steps:**
1. Generate fine-grained time array (1-minute steps for pass detection)
2. Propagate satellite over window using `orbit.propagate_trajectory()`
3. Compute elevation angles via `visible_from_location()`
4. **Pass detection**: Find contiguous intervals where `elevation_deg >= observer.min_elevation_deg`
   - AOS = first index crossing threshold
   - TCA = index of maximum elevation in interval
   - LOS = last index before dropping below threshold
5. Construct `PassEvent` for each interval
6. Return list of `PassEvent` sorted by `aos_jd`

---

## Module 6: `astra.time`

### Responsibility

All time conversion and representation. Provides a unified interface between Python `datetime`, Julian Dates, `skyfield` Time objects, and ISO 8601 strings.

### Boundaries

| Imports From | Never Imports From |
|---|---|
| `datetime`, `skyfield`, `numpy` | Any `astra` domain module |

### Public Functions

---

#### `convert_time`

```python
def convert_time(
    value: str | datetime | float,
    to_format: str,     # "jd", "datetime", "skyfield", "iso"
) -> float | datetime | skyfield.timelib.Time | str:
```

**Purpose:** Universal time format converter.

**Supported `to_format` values:**
- `"jd"` → returns `float` Julian Date
- `"datetime"` → returns Python `datetime` (UTC aware)
- `"skyfield"` → returns `skyfield.timelib.Time`
- `"iso"` → returns ISO 8601 string `"2025-01-01T00:00:00Z"`

**Supported input types:**
- `str`: ISO 8601 format `"YYYY-MM-DDTHH:MM:SSZ"`
- `datetime`: Python timezone-aware or naive (assumed UTC)
- `float`: Julian Date

---

## Module 7: `astra.utils`

### Responsibility

Pure mathematical and geometric utility functions. No domain concepts — only math.

### Boundaries

| Imports From | Never Imports From |
|---|---|
| `numpy`, `math` | Any `astra` domain module |

### Public Functions

---

#### `haversine_distance`

```python
def haversine_distance(
    lat1_deg: float, lon1_deg: float,
    lat2_deg: float, lon2_deg: float,
) -> float:  # Great-circle distance in km
```

**Logic:** Standard Haversine formula with Earth radius = 6371.0 km.

---

#### `orbital_elements`

```python
def orbital_elements(
    line2: str,
) -> dict[str, float]:
```

**Purpose:** Extract Keplerian orbital elements from a TLE line 2 string.

**Returns dict with:**
- `inclination_deg`, `raan_deg`, `eccentricity`, `arg_perigee_deg`, `mean_anomaly_deg`, `mean_motion_rev_per_day`

---

#### `orbit_period`

```python
def orbit_period(mean_motion_rev_per_day: float) -> float:
    # Returns orbital period in minutes
```

**Logic:** `T_minutes = (24 * 60) / mean_motion_rev_per_day`
