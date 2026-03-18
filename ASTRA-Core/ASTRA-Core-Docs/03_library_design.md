# 03 — ASTRA Core: Library Design

---

## 1. Design Philosophy

ASTRA Core is a **functional computation library**. Every design decision flows from one central principle:

> **Functions receive data, transform data, and return data. They do nothing else.**

This is not a stylistic preference. It is a hard requirement for:
- Scientific correctness (deterministic behavior)
- Testability (isolated unit tests)
- Composability (functions can be chained in pipelines)
- Scalability (pure functions are trivially parallelizable)

---

## 2. The Eight Mandatory Design Principles

### Principle 1: Pure Functions

Every function in ASTRA Core must be a **pure function**:
- Output is **entirely determined by inputs**
- Function produces **no side effects**
- Function does **not modify its inputs**

```python
# CORRECT: Pure function
def filter_altitude(
    objects: list[DebrisObject],
    min_km: float,
    max_km: float
) -> list[DebrisObject]:
    return [o for o in objects if min_km <= o.altitude_km <= max_km]

# WRONG: Modifies input (impure)
def filter_altitude(objects, min_km, max_km):
    objects[:] = [o for o in objects if min_km <= o.altitude_km <= max_km]
```

### Principle 2: Deterministic Behavior

Given the same inputs, a function must **always produce the same output**.

- No random numbers unless a seed is explicitly passed
- No time-dependent behavior (use explicit `datetime` inputs, never `datetime.now()` inside logic)
- No floating-point non-determinism from different orderings

### Principle 3: No Hidden Global State

Module-level mutable variables are **forbidden**.

```python
# FORBIDDEN:
_catalog: list[DebrisObject] = []  # mutable global

def load_catalog(path: str) -> None:
    global _catalog
    _catalog = parse(path)  # side effect

# REQUIRED:
def load_tle_catalog(tle_lines: list[str]) -> list[SatelliteTLE]:
    return [parse_tle(block) for block in chunk(tle_lines, 3)]
```

### Principle 4: No Side Effects

Functions must not:
- Read from or write to disk
- Make network requests
- Log to stdout/stderr (library-level logging via `logging` module is acceptable as configurable, not mandatory output)
- Modify database contents

### Principle 5: Strict Module Separation

Each module owns exactly one concern. Cross-module calls must flow **only downward** in the dependency hierarchy (see architecture doc).

```python
# OK: orbit.py importing from tle.py (downward dependency)
from astra.tle import SatelliteTLE

# FORBIDDEN: tle.py importing from orbit.py (upward dependency)
from astra.orbit import propagate_orbit  # NEVER
```

### Principle 6: Vector-First Computation

Single-object operations are convenience wrappers. The primary computational primitive is always **batched/vectorized**.

```python
# Primary API (vectorized):
def propagate_many(
    satellites: list[SatelliteTLE],
    time_steps: np.ndarray  # shape: (T,)
) -> dict[str, np.ndarray]:  # {norad_id: shape (T, 3)}
    ...

# Convenience wrapper (single object):
def propagate_orbit(
    satellite: SatelliteTLE,
    time_step: float
) -> OrbitalState:
    return propagate_many([satellite], np.array([time_step]))[satellite.norad_id][0]
```

### Principle 7: Explicit Data Models

All data passed between functions is represented by **typed dataclass instances**, never raw tuples, raw dicts, or ambiguously typed primitives.

```python
# FORBIDDEN:
def propagate(tle_line1, tle_line2, epoch_jd):
    return (x, y, z, vx, vy, vz)  # What are these? What units?

# REQUIRED:
def propagate_orbit(satellite: SatelliteTLE, t: float) -> OrbitalState:
    ...  # Returns fully typed, documented OrbitalState
```

### Principle 8: Testable Functions

Every function must be testable in isolation with:
- Known input values
- Computable expected outputs
- No external dependencies (database, network, filesystem)

---

## 3. Package Structure Design

```
astra/
├── __init__.py        # Re-exports public API symbols
├── errors.py          # AstraError, InvalidTLEError, PropagationError
├── models.py          # All dataclass definitions (single source of truth)
├── time.py            # Time conversion utilities
├── tle.py             # TLE parsing and validation
├── orbit.py           # Propagation engine
├── debris.py          # Filtering pipeline
├── conjunction.py     # Conjunction detection
├── visibility.py      # Ground station visibility
└── utils.py           # Math/geometry helpers
```

**Note on `models.py`:** All dataclasses reside in a single file. This eliminates circular imports. Modules import from `astra.models`, never from each other's model definitions.

---

## 4. Public API Surface

The `astra/__init__.py` file exposes a curated public API:

```python
# astra/__init__.py

# Data models
from astra.models import (
    SatelliteTLE,
    OrbitalState,
    DebrisObject,
    ConjunctionEvent,
    Observer,
    PassEvent,
    FilterConfig,
)

# Error types
from astra.errors import (
    AstraError,
    InvalidTLEError,
    PropagationError,
)

# TLE functions
from astra.tle import parse_tle, validate_tle, load_tle_catalog

# Orbit functions
from astra.orbit import (
    propagate_orbit,
    propagate_many,
    propagate_trajectory,
    ground_track,
)

# Debris functions
from astra.debris import (
    filter_altitude,
    filter_region,
    filter_time_window,
    catalog_statistics,
)

# Conjunction functions
from astra.conjunction import (
    distance_3d,
    closest_approach,
    find_conjunctions,
    collision_probability,
)

# Visibility functions
from astra.visibility import (
    visible_from_location,
    passes_over_location,
)

# Time functions
from astra.time import convert_time

# Utility functions
from astra.utils import (
    haversine_distance,
    orbital_elements,
    orbit_period,
)
```

---

## 5. Naming Conventions

| Category | Convention | Example |
|---|---|---|
| Modules | lowercase, single word | `orbit.py`, `debris.py` |
| Classes / Dataclasses | PascalCase | `SatelliteTLE`, `ConjunctionEvent` |
| Functions | snake_case, verb_noun | `propagate_orbit`, `filter_altitude` |
| Constants | UPPER_SNAKE_CASE | `SIMULATION_STEPS = 288` |
| Private helpers | leading underscore | `_chunk_tle_lines()` |
| Type aliases | PascalCase | `TrajectoryMap = dict[str, np.ndarray]` |
| Parameters (distance) | always `_km` suffix | `threshold_km`, `min_km`, `max_km` |
| Parameters (angle) | always `_deg` suffix | `elevation_deg`, `lat_deg` |
| Parameters (time) | always `_utc` or `_jd` suffix | `epoch_utc`, `t_start_jd` |

---

## 6. Type Annotation Policy

All public functions use **full Python type annotations** (PEP 484). `from __future__ import annotations` is used at the top of each module for forward references.

```python
from __future__ import annotations
from typing import Optional
import numpy as np
from astra.models import SatelliteTLE, OrbitalState

def propagate_orbit(
    satellite: SatelliteTLE,
    epoch_jd: float,
    t_since_minutes: float,
) -> OrbitalState:
    ...
```

`np.ndarray` arrays must be documented with shape and dtype in docstrings:

```python
def propagate_many(
    satellites: list[SatelliteTLE],
    time_steps: np.ndarray,  # shape: (T,), dtype: float64, unit: minutes since epoch
) -> dict[str, np.ndarray]:  # {norad_id: ndarray shape (T, 3), unit: km, frame: TEME}
    ...
```

---

## 7. Docstring Standard

All public functions use **Google-style docstrings**:

```python
def filter_altitude(
    objects: list[DebrisObject],
    min_km: float,
    max_km: float,
) -> list[DebrisObject]:
    """Filter debris objects by mean orbital altitude band.

    Uses each object's precomputed mean altitude (semi-major axis minus Earth
    radius) to classify objects into altitude regimes. Does not perform
    propagation. O(n) in catalog size.

    Args:
        objects: List of debris objects to filter. Must have altitude_km set.
        min_km: Lower bound of altitude band, inclusive. Must be >= 0.
        max_km: Upper bound of altitude band, inclusive. Must be > min_km.

    Returns:
        List of DebrisObject instances where min_km <= altitude_km <= max_km.
        Order is preserved from input list.

    Raises:
        ValueError: If min_km >= max_km or if min_km < 0.

    Example:
        >>> leos = filter_altitude(catalog, min_km=160, max_km=2000)
    """
```

---

## 8. Constants

Defined in a single `astra/constants.py` file (or as module-level constants in the relevant module):

```python
# Simulation parameters
SIMULATION_WINDOW_HOURS: int = 24
SIMULATION_STEP_MINUTES: int = 5
SIMULATION_STEPS: int = 288  # 24 * 60 / 5

# Earth parameters
EARTH_RADIUS_KM: float = 6371.0
EARTH_MU_KM3_S2: float = 398600.4418  # gravitational parameter

# Orbital regime boundaries (km altitude above surface)
LEO_MIN_KM: float = 160.0
LEO_MAX_KM: float = 2000.0
MEO_MIN_KM: float = 2000.0
MEO_MAX_KM: float = 35786.0
GEO_ALTITUDE_KM: float = 35786.0
HEO_MIN_KM: float = 35786.0

# Conjunction thresholds
CONJUNCTION_THRESHOLD_KM: float = 5.0   # fine-grained detection
COARSE_FILTER_THRESHOLD_KM: float = 50.0  # coarse pre-filter

# Collision probability reference volume (Chan formula)
COLLISION_VOLUME_SCALE_M: float = 10.0   # combined hard-body radius in meters
```

---

## 9. Error Handling Design

ASTRA Core uses a custom exception hierarchy (detailed in `10_error_handling.md`):

```
AstraError (base)
├── InvalidTLEError        ← TLE parsing/validation failures
├── PropagationError       ← SGP4 propagation failures
└── FilterError            ← Invalid filter configuration
```

All exceptions carry:
- A human-readable message
- The offending input that caused the error
- Contextual metadata as keyword arguments

---

## 10. Immutability Contract

All data model instances are **frozen dataclasses** (immutable after creation):

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class SatelliteTLE:
    norad_id: str
    name: str
    line1: str
    line2: str
    epoch_jd: float
```

**Rationale:** Immutable data models eliminate an entire class of bugs where functions accidentally mutate shared state. If transformation is needed, a new instance is created.
