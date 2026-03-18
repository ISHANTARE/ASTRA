# 10 — ASTRA Core: Error Handling

---

## 1. Error Handling Philosophy

ASTRA Core uses **explicit, typed exceptions** for all error conditions. There are no silent failures, no `None` returns on error, no bare `except Exception` catches. Every error condition has a defined exception class that carries full diagnostic information.

---

## 2. Exception Hierarchy

```
BaseException
└── Exception
    └── AstraError                    ← Base for all ASTRA exceptions
        ├── InvalidTLEError           ← TLE parsing and validation failures
        ├── PropagationError          ← SGP4 propagation failures
        ├── FilterError               ← Invalid filter configuration
        └── CoordinateError           ← Coordinate frame conversion failures
```

---

## 3. `AstraError` (Base Class)

```python
class AstraError(Exception):
    """Base class for all ASTRA Core exceptions.
    
    All ASTRA exceptions derive from this class, allowing callers
    to catch AstraError for broad handling or specific subclasses
    for targeted handling.
    """
    
    def __init__(self, message: str, **context: Any) -> None:
        super().__init__(message)
        self.message = message
        self.context = context  # Arbitrary diagnostic key-value pairs
    
    def __str__(self) -> str:
        if self.context:
            ctx_str = ", ".join(f"{k}={v!r}" for k, v in self.context.items())
            return f"{self.message} [{ctx_str}]"
        return self.message
```

**When Raised:**
- As a fallback for ASTRA-level errors that don't fit a more specific subclass
- When `load_tle_catalog()` receives input that results in zero successfully parsed objects

---

## 4. `InvalidTLEError`

```python
class InvalidTLEError(AstraError):
    """Raised when a TLE string fails parsing or validation."""
    
    def __init__(
        self,
        message: str,
        norad_id: Optional[str] = None,
        object_name: Optional[str] = None,
        invalid_line: Optional[str] = None,
        reason: Optional[str] = None,
    ) -> None:
        super().__init__(message, norad_id=norad_id, object_name=object_name,
                         invalid_line=invalid_line, reason=reason)
        self.norad_id = norad_id
        self.object_name = object_name
        self.invalid_line = invalid_line
        self.reason = reason
```

**When Raised:**

| Condition | `reason` value |
|---|---|
| Line 1 does not start with `'1'` | `"line1_bad_start"` |
| Line 2 does not start with `'2'` | `"line2_bad_start"` |
| Line 1 length != 69 characters | `"line1_wrong_length"` |
| Line 2 length != 69 characters | `"line2_wrong_length"` |
| Line 1 checksum mismatch | `"line1_checksum_invalid"` |
| Line 2 checksum mismatch | `"line2_checksum_invalid"` |
| NORAD ID field is non-numeric | `"norad_id_invalid"` |
| Epoch field is malformed | `"epoch_parse_failed"` |
| Eccentricity out of range [0, 1) | `"eccentricity_invalid"` |
| Mean motion ≤ 0 | `"mean_motion_invalid"` |

**Example:**

```python
try:
    tle = parse_tle("ISS", bad_line1, line2)
except InvalidTLEError as e:
    # e.reason = "line1_checksum_invalid"
    # e.object_name = "ISS"
    # e.invalid_line = bad_line1
    log.warning(f"Invalid TLE for {e.object_name}: {e.reason}")
```

**In `load_tle_catalog`:** `InvalidTLEError` is caught internally and the offending TLE is **skipped with a logger.warning()**. It is NOT re-raised during batch loading unless ALL TLEs fail.

---

## 5. `PropagationError`

```python
class PropagationError(AstraError):
    """Raised when SGP4 propagation fails for a satellite."""
    
    def __init__(
        self,
        message: str,
        norad_id: Optional[str] = None,
        error_code: Optional[int] = None,
        t_jd: Optional[float] = None,
    ) -> None:
        super().__init__(message, norad_id=norad_id, error_code=error_code, t_jd=t_jd)
        self.norad_id = norad_id
        self.error_code = error_code
        self.t_jd = t_jd
```

**When Raised:**

| Condition | `error_code` |
|---|---|
| `propagate_orbit()` called with invalid satellite epoch (TLE is too old for requested time, causing SGP4 divergence) | 1 |
| Mean motion outside physical bounds | 2 |
| Eccentricity oscillation causes numerical breakdown | 3 |
| Semi-latus rectum becomes negative | 4 |
| Epoch elements classify satellite as sub-orbital | 5 |
| Satellite has re-entered atmosphere (decayed) | 6 |

**Special Rule for `propagate_many()`:**
`propagate_many` does NOT raise `PropagationError`. Instead, it stores `np.nan` in the affected trajectory rows and returns normally. Callers who require complete trajectories must check for `np.isnan` rows.

`PropagationError` is raised ONLY by `propagate_orbit()` when called in strict mode, or by callers who explicitly check the `error_code` field of `OrbitalState`.

---

## 6. `FilterError`

```python
class FilterError(AstraError):
    """Raised when filter parameters are invalid or contradictory."""
    
    def __init__(
        self,
        message: str,
        parameter: Optional[str] = None,
        value: Optional[Any] = None,
    ) -> None:
        super().__init__(message, parameter=parameter, value=value)
        self.parameter = parameter
        self.value = value
```

**When Raised:**

| Condition | `parameter` |
|---|---|
| `min_km >= max_km` in `filter_altitude()` | `"min_km"` |
| `min_km < 0` | `"min_km"` |
| `lat_min_deg > lat_max_deg` | `"lat_min_deg"` |
| Latitude out of range [-90, 90] | `"lat_min_deg"` or `"lat_max_deg"` |
| `t_start_jd >= t_end_jd` in `filter_time_window()` | `"t_start_jd"` |
| `step_minutes <= 0` in any propagation function | `"step_minutes"` |

---

## 7. `CoordinateError`

```python
class CoordinateError(AstraError):
    """Raised when coordinate conversion fails or inputs are invalid."""
    
    def __init__(
        self,
        message: str,
        frame: Optional[str] = None,  # e.g. "TEME", "GCRS", "ITRS"
    ) -> None:
        super().__init__(message, frame=frame)
        self.frame = frame
```

**When Raised:**
- `ground_track()` receives position arrays with `np.nan` values (failed propagation rows)
- Skyfield fails to perform TEME → ITRS conversion (e.g., missing Earth orientation data)

---

## 8. Error Handling Reference by Function

| Function | Raises | When |
|---|---|---|
| `parse_tle()` | `InvalidTLEError` | Any TLE validation failure |
| `validate_tle()` | Never | Returns bool |
| `load_tle_catalog()` | `AstraError` | All TLEs failed to parse |
| `propagate_orbit()` | `PropagationError` | SGP4 error_code > 0 |
| `propagate_many()` | Never raises, stores NaN | Any propagation failures |
| `propagate_trajectory()` | `PropagationError` | If ALL timesteps fail for one satellite |
| `ground_track()` | `CoordinateError` | NaN positions received |
| `filter_altitude()` | `FilterError` | Invalid altitude bounds |
| `filter_region()` | `FilterError` | Invalid lat/lon bounds |
| `filter_time_window()` | `FilterError` | Invalid time window |
| `find_conjunctions()` | `AstraError` | Empty trajectory map passed |
| `closest_approach()` | `ValueError` | Arrays have different shapes |
| `passes_over_location()` | `PropagationError` | Propagation fully fails |
| `convert_time()` | `ValueError` | Unrecognized format string |

---

## 9. Logging Policy

ASTRA Core uses Python's standard `logging` module (not `print`):

```python
import logging
log = logging.getLogger("astra")
```

| Situation | Log Level |
|---|---|
| Skipped TLE in `load_tle_catalog()` | `WARNING` |
| NaN propagation row in `propagate_many()` | `DEBUG` |
| Filter stage removed all objects | `WARNING` |
| Conjunction pipeline found 0 candidates | `INFO` |
| Successful catalog load | `INFO` |

**Rules:**
- ASTRA Core has NO `logging.basicConfig()` calls — log configuration is the caller's responsibility
- Log messages contain NORAD ID where applicable for traceability
- No `stderr` or `stdout` writes — only `logging` module

---

## 10. Complete `errors.py` Module

```python
# astra/errors.py
from __future__ import annotations
from typing import Any, Optional


class AstraError(Exception):
    """Base class for all ASTRA Core exceptions."""
    
    def __init__(self, message: str, **context: Any) -> None:
        super().__init__(message)
        self.message = message
        self.context = context
    
    def __str__(self) -> str:
        if self.context:
            ctx = ", ".join(f"{k}={v!r}" for k, v in self.context.items() if v is not None)
            return f"{self.message} [{ctx}]" if ctx else self.message
        return self.message


class InvalidTLEError(AstraError):
    """Raised when TLE parsing or validation fails."""
    
    def __init__(
        self,
        message: str,
        norad_id: Optional[str] = None,
        object_name: Optional[str] = None,
        invalid_line: Optional[str] = None,
        reason: Optional[str] = None,
    ) -> None:
        super().__init__(
            message,
            norad_id=norad_id,
            object_name=object_name,
            invalid_line=invalid_line,
            reason=reason,
        )
        self.norad_id = norad_id
        self.object_name = object_name
        self.invalid_line = invalid_line
        self.reason = reason


class PropagationError(AstraError):
    """Raised when SGP4 orbit propagation fails."""
    
    def __init__(
        self,
        message: str,
        norad_id: Optional[str] = None,
        error_code: Optional[int] = None,
        t_jd: Optional[float] = None,
    ) -> None:
        super().__init__(
            message,
            norad_id=norad_id,
            error_code=error_code,
            t_jd=t_jd,
        )
        self.norad_id = norad_id
        self.error_code = error_code
        self.t_jd = t_jd


class FilterError(AstraError):
    """Raised when filter parameters are invalid."""
    
    def __init__(
        self,
        message: str,
        parameter: Optional[str] = None,
        value: Optional[Any] = None,
    ) -> None:
        super().__init__(message, parameter=parameter, value=value)
        self.parameter = parameter
        self.value = value


class CoordinateError(AstraError):
    """Raised when coordinate frame conversion fails."""
    
    def __init__(
        self,
        message: str,
        frame: Optional[str] = None,
    ) -> None:
        super().__init__(message, frame=frame)
        self.frame = frame
```
