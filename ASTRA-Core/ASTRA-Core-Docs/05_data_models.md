# 05 — ASTRA Core: Data Models

---

## 1. Overview

All persistent data structures in ASTRA Core are defined in `astra/models.py` as **frozen dataclasses**. No raw tuples, no untyped dicts, no dynamic attribute access. Every inter-function data transfer uses one of these types.

---

## 2. `SatelliteTLE`

### Purpose

Represents a parsed Two-Line Element set for a single orbital object. This is the **entry point** for all computations. Any satellite, debris, or rocket body that can be analyzed in ASTRA Core must first exist as a `SatelliteTLE`.

### Definition

```python
@dataclass(frozen=True)
class SatelliteTLE:
    norad_id: str       # NORAD Catalog Number (5 digits, zero-padded), e.g. "25544"
    name: str           # Object name, e.g. "ISS (ZARYA)"
    line1: str          # Raw TLE Line 1, exactly 69 characters
    line2: str          # Raw TLE Line 2, exactly 69 characters
    epoch_jd: float     # TLE epoch as Julian Date (days since J2000)
    object_type: str    # "PAYLOAD", "ROCKET_BODY", "DEBRIS", "UNKNOWN"
```

### Field Details

| Field | Type | Constraints | Notes |
|---|---|---|---|
| `norad_id` | `str` | 1–9 digits, numeric-only | Unique identifier |
| `name` | `str` | max 24 characters | May contain spaces |
| `line1` | `str` | exactly 69 chars | Raw TLE line, no newline |
| `line2` | `str` | exactly 69 chars | Raw TLE line, no newline |
| `epoch_jd` | `float` | > 2400000.0 | Parsed from TLE field 1.4 |
| `object_type` | `str` | one of 4 enum values | Derived from TLE classification field |

### Usage

```python
# Created by astra.tle.parse_tle()
tle = parse_tle("ISS (ZARYA)", line1, line2)

# Used by propagation functions
state = propagate_orbit(tle, epoch_jd=tle.epoch_jd, t_since_minutes=0.0)
```

### Data Flow

```
Raw TLE text strings
    ↓ parse_tle()
SatelliteTLE
    ↓ propagate_orbit() / propagate_many()
OrbitalState / trajectory arrays
```

---

## 3. `OrbitalState`

### Purpose

Represents the **complete kinematic state** of a single orbital object at a single instant in time. This is the output of SGP4 propagation for a single object at a single timestep.

### Definition

```python
@dataclass(frozen=True)
class OrbitalState:
    norad_id: str           # Identifier linking back to SatelliteTLE
    t_jd: float             # Julian Date of this state
    position_km: np.ndarray # shape (3,): [x, y, z] in TEME frame, km
    velocity_km_s: np.ndarray  # shape (3,): [vx, vy, vz] in TEME frame, km/s
    error_code: int         # SGP4 error code: 0 = success, >0 = failure
```

### Field Details

| Field | Type | Constraints | Notes |
|---|---|---|---|
| `norad_id` | `str` | matches source TLE | Reference back to source |
| `t_jd` | `float` | valid Julian Date | Time of state |
| `position_km` | `np.ndarray` | shape `(3,)`, float64 | TEME Cartesian, km |
| `velocity_km_s` | `np.ndarray` | shape `(3,)`, float64 | TEME Cartesian, km/s |
| `error_code` | `int` | 0–6 | SGP4 internal error codes |

### SGP4 Error Codes

| Code | Meaning |
|---|---|
| 0 | Success |
| 1 | Mean elements out of range |
| 2 | Mean motion out of range |
| 3 | Pert elements out of range |
| 4 | Semi-latus rectum < 0 |
| 5 | Epoch elements sub-orbital |
| 6 | Satellite decayed |

### Usage

```python
state = propagate_orbit(tle, epoch_jd=..., t_since_minutes=0.0)

if state.error_code != 0:
    raise PropagationError(...)

x, y, z = state.position_km
```

### Note on Frozen Arrays

Because `OrbitalState` is frozen, `np.ndarray` fields are not truly immutable. The dataclass `frozen=True` prevents reassignment of the attribute itself (`state.position_km = ...` fails) but not in-place mutation (`state.position_km[0] = 99` would succeed). Callers must not mutate array fields.

---

## 4. `DebrisObject`

### Purpose

Represents a cataloged orbital object (satellite, debris, rocket body) with pre-derived orbital parameters for use in filtering computations. Contains the `SatelliteTLE` as the authoritative source plus derived metrics that enable fast filtering **without propagation**.

### Definition

```python
@dataclass(frozen=True)
class DebrisObject:
    tle: SatelliteTLE           # Authoritative TLE source
    altitude_km: float          # Mean orbital altitude above Earth's surface (km)
    inclination_deg: float      # Orbital inclination in degrees
    period_minutes: float       # Orbital period in minutes
    raan_deg: float             # Right Ascension of the Ascending Node (degrees)
    eccentricity: float         # Orbital eccentricity (0.0 = circular, <1.0 = elliptical)
    apogee_km: float            # Apogee altitude (km above surface)
    perigee_km: float           # Perigee altitude (km above surface)
    object_class: str           # Same as tle.object_type
```

### Field Details

| Field | Type | Derivation Method | Notes |
|---|---|---|---|
| `tle` | `SatelliteTLE` | Direct from parse | Source of truth |
| `altitude_km` | `float` | `(apogee + perigee) / 2` | Mean altitude |
| `inclination_deg` | `float` | TLE Line 2 field 2.3 | Degrees 0–180 |
| `period_minutes` | `float` | `2π / n` where `n` = mean motion | Derived from TLE |
| `raan_deg` | `float` | TLE Line 2 field 2.4 | 0–360 degrees |
| `eccentricity` | `float` | TLE Line 2 field 2.5 | 0.0 ≤ e < 1.0 |
| `apogee_km` | `float` | `a * (1 + e) - R_earth` | km above surface |
| `perigee_km` | `float` | `a * (1 - e) - R_earth` | km above surface |
| `object_class` | `str` | From TLE classification | "DEBRIS", etc. |

### Usage

```python
# Create DebrisObject from SatelliteTLE
tle = parse_tle(name, line1, line2)
debris = make_debris_object(tle)  # derives all orbital elements

# Fast filtering (no propagation needed)
leo_objects = filter_altitude(catalog, min_km=160, max_km=2000)
```

### Factory Function

```python
def make_debris_object(tle: SatelliteTLE) -> DebrisObject:
    """Derive DebrisObject from SatelliteTLE using orbital mechanics.
    
    Derives: altitude, inclination, period, RAAN, eccentricity,
             apogee, perigee from TLE fields via sgp4 satrec object.
    """
```

---

## 5. `ConjunctionEvent`

### Purpose

Represents a detected **close approach event** between two orbital objects at a specific time. This is the primary output of conjunction analysis.

### Definition

```python
@dataclass(frozen=True)
class ConjunctionEvent:
    object_a_id: str        # NORAD ID of first object
    object_b_id: str        # NORAD ID of second object
    tca_jd: float           # Time of Closest Approach, Julian Date
    miss_distance_km: float # Minimum separation distance at TCA, km
    relative_velocity_km_s: float   # Relative speed at TCA, km/s
    collision_probability: float    # Chan method probability (0.0–1.0)
    risk_level: str         # "LOW", "MEDIUM", "HIGH", "CRITICAL"
    position_a_km: np.ndarray   # shape (3,): TEME position of object A at TCA
    position_b_km: np.ndarray   # shape (3,): TEME position of object B at TCA
```

### Field Details

| Field | Type | Notes |
|---|---|---|
| `object_a_id` | `str` | Always lexicographically smaller to avoid duplicates |
| `object_b_id` | `str` | Always lexicographically larger |
| `tca_jd` | `float` | Julian Date of closest approach |
| `miss_distance_km` | `float` | Minimum distance found over time window |
| `relative_velocity_km_s` | `float` | Scalar magnitude of velocity difference at TCA |
| `collision_probability` | `float` | 0.0–1.0 from Chan analytical method |
| `risk_level` | `str` | Classified by thresholds (see below) |
| `position_a_km` | `np.ndarray` | shape `(3,)`, TEME, km |
| `position_b_km` | `np.ndarray` | shape `(3,)`, TEME, km |

### Risk Level Classification

| Risk Level | Condition |
|---|---|
| `"LOW"` | P_c < 1e-6 and miss_distance > 2 km |
| `"MEDIUM"` | P_c ∈ [1e-6, 1e-4) or miss_distance ∈ [0.5, 2) km |
| `"HIGH"` | P_c ∈ [1e-4, 1e-2) or miss_distance ∈ [0.1, 0.5) km |
| `"CRITICAL"` | P_c ≥ 1e-2 or miss_distance < 0.1 km |

---

## 6. `Observer`

### Purpose

Represents a **ground-based observation station** from which satellite visibility is calculated. Required by all functions in `astra.visibility`.

### Definition

```python
@dataclass(frozen=True)
class Observer:
    name: str               # Station name, e.g. "ISRO Bangalore"
    latitude_deg: float     # WGS84 geodetic latitude, -90 to +90
    longitude_deg: float    # WGS84 longitude, -180 to +180
    elevation_m: float      # Station elevation above MSL, meters
    min_elevation_deg: float = 10.0  # Minimum elevation angle for visible pass (default 10°)
```

### Field Details

| Field | Type | Constraints | Notes |
|---|---|---|---|
| `name` | `str` | Any string | Human-readable label |
| `latitude_deg` | `float` | -90.0 to +90.0 | Positive = North |
| `longitude_deg` | `float` | -180.0 to +180.0 | Positive = East |
| `elevation_m` | `float` | ≥ 0.0 | Meters above MSL |
| `min_elevation_deg` | `float` | 0.0 to 90.0 | Minimum for "visible" |

### Usage

```python
observer = Observer(
    name="ISRO Bangalore",
    latitude_deg=12.9716,
    longitude_deg=77.5946,
    elevation_m=920.0,
    min_elevation_deg=10.0
)

passes = passes_over_location(tle, observer, t_start, t_end)
```

---

## 7. `PassEvent`

### Purpose

Represents a **satellite pass** over a ground observer — a time interval during which the satellite's elevation angle exceeds the observer's minimum elevation threshold.

### Definition

```python
@dataclass(frozen=True)
class PassEvent:
    norad_id: str           # NORAD ID of the satellite
    observer_name: str      # Name of the ground observer
    aos_jd: float           # Acquisition of Signal: rise above min elevation (JD)
    tca_jd: float           # Time of Closest Approach (max elevation) (JD)
    los_jd: float           # Loss of Signal: drops below min elevation (JD)
    max_elevation_deg: float   # Maximum elevation reached during pass
    azimuth_at_aos_deg: float  # Azimuth angle at AOS
    azimuth_at_los_deg: float  # Azimuth angle at LOS
    duration_seconds: float    # Total pass duration: (los_jd - aos_jd) * 86400
```

---

## 8. `FilterConfig`

### Purpose

Encapsulates filter parameters for the multi-stage debris filtering pipeline. Passed as a single configuration object to avoid long parameter lists.

### Definition

```python
@dataclass(frozen=True)
class FilterConfig:
    min_altitude_km: Optional[float] = None
    max_altitude_km: Optional[float] = None
    lat_min_deg: Optional[float] = None
    lat_max_deg: Optional[float] = None
    lon_min_deg: Optional[float] = None
    lon_max_deg: Optional[float] = None
    t_start_jd: Optional[float] = None
    t_end_jd: Optional[float] = None
    object_types: Optional[tuple[str, ...]] = None  # e.g. ("DEBRIS", "ROCKET_BODY")
    max_objects: Optional[int] = None               # Hard cap on survivors
```

---

## 9. Type Aliases

Defined in `astra/models.py` for common container types:

```python
from typing import Optional
import numpy as np

# A map from NORAD ID to trajectory array
# Shape of each array: (T, 3) where T = number of timesteps
# Units: km, frame: TEME
TrajectoryMap = dict[str, np.ndarray]

# A list of NORAD ID pairs (candidate conjunction pairs)
CandidatePairs = list[tuple[str, str]]

# Time array: array of Julian Dates
TimeArray = np.ndarray  # shape (T,), dtype float64
```

---

## 10. Data Flow Summary

```
Raw TLE Text (strings)
        │
        ▼ parse_tle()
SatelliteTLE  ←── identity object for all computation
        │
        ├──▶ make_debris_object()
        │         │
        │         ▼
        │    DebrisObject  ←── used for pre-propagation filtering
        │         │
        │    filter_altitude()
        │    filter_region()
        │    filter_time_window()
        │         │
        │         ▼
        │    [DebrisObject survivors]
        │         │
        └──▶ propagate_many() ◄─────── only survivors propagated
                  │
                  ▼
            TrajectoryMap  {norad_id: ndarray(T,3)}
                  │
                  ├──▶ find_conjunctions()
                  │         │
                  │         ▼
                  │    List[ConjunctionEvent]
                  │
                  └──▶ ground_track()
                            │
                            ▼
                       List[(lat, lon)]

Observer ──▶ passes_over_location(SatelliteTLE, Observer)
                  │
                  ▼
             List[PassEvent]
```
