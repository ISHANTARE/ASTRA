# 12 — ASTRA Core: Code Style

---

## 1. Overview

ASTRA Core follows a strict, enforced code style. All style rules are automatically enforced by `ruff` (linting + formatting) and `mypy` (type checking). No manual code review should be required for style compliance.

---

## 2. Formatter: `ruff format`

`ruff format` is the ONLY formatter for ASTRA Core. Do NOT use `black`, `autopep8`, or `yapf`.

**Configuration (`pyproject.toml`):**

```toml
[tool.ruff]
line-length = 100
target-version = "py310"

[tool.ruff.format]
quote-style = "double"
indent-style = "space"
skip-magic-trailing-comma = false
```

**Rules:**
- Line length: 100 characters maximum
- String quotes: double-quotes everywhere (`"value"`, not `'value'`)
- Indentation: 4 spaces (no tabs)
- Trailing commas: Use in multi-line argument lists (enables clean diffs)

---

## 3. Linter: `ruff check`

```toml
[tool.ruff.lint]
select = [
    "E",    # pycodestyle errors
    "W",    # pycodestyle warnings
    "F",    # pyflakes
    "I",    # isort
    "N",    # pep8-naming
    "UP",   # pyupgrade
    "B",    # flake8-bugbear
    "SIM",  # flake8-simplify
    "ANN",  # flake8-annotations (type hints)
]
ignore = [
    "ANN101",  # Missing self type annotation
    "ANN102",  # Missing cls type annotation
]
```

---

## 4. Type Checker: `mypy`

All ASTRA Core code must pass `mypy --strict`.

```toml
[tool.mypy]
python_version = "3.10"
strict = true
warn_return_any = true
warn_unused_configs = true
```

**Rules:**
- Every function parameter must be annotated
- Every function return type must be annotated
- `Optional[T]` is preferred over `T | None` for parameter defaults
- `from __future__ import annotations` at top of every module
- `np.ndarray` must have a type comment specifying shape and dtype in docstrings

---

## 5. Import Order

Enforce with `ruff --select I`:

```python
# 1. Future imports
from __future__ import annotations

# 2. Standard library
import math
from datetime import datetime
from typing import Any, Optional

# 3. Third-party libraries
import numpy as np
from sgp4.api import Satrec, sgp4_array
from skyfield.api import EarthSatellite, load

# 4. Internal ASTRA imports
from astra.errors import InvalidTLEError, PropagationError
from astra.models import DebrisObject, OrbitalState, SatelliteTLE
```

---

## 6. Naming Conventions

| Item | Convention | Example |
|---|---|---|
| Module | lowercase, no underscore | `orbit.py`, `tle.py` |
| Class | PascalCase | `SatelliteTLE`, `ConjunctionEvent` |
| Function | snake_case, verb first | `propagate_many`, `filter_altitude` |
| Method | snake_case | `to_julian_date()` |
| Constant | UPPER_SNAKE_CASE | `SIMULATION_STEPS = 288` |
| Private helper | leading underscore | `_parse_epoch()` |
| Variable | snake_case | `trajectory_map`, `t_since_min` |
| Parameter (km) | `_km` suffix | `min_km`, `threshold_km` |
| Parameter (deg) | `_deg` suffix | `inclination_deg`, `lat_min_deg` |
| Parameter (jd) | `_jd` suffix | `t_start_jd`, `epoch_jd` |
| Boolean variable | `is_` or `has_` prefix | `is_valid`, `has_error` |

---

## 7. Function Design Rules

### 7.1 Single Responsibility
Every function does exactly one thing. If a function does two things, split it.

### 7.2 Max Function Length
Public functions: ≤ 50 lines of actual code (docstring excluded).
Private helpers: ≤ 30 lines.

### 7.3 Argument Count
Maximum 6 parameters per function. If more are needed, use a dataclass (e.g., `FilterConfig`).

### 7.4 No Optional Returns
Never return `None` on success. Functions return typed values or raise exceptions.

```python
# WRONG:
def find_something() -> Optional[SatelliteTLE]:
    ...
    return None  # on failure

# RIGHT:
def find_something() -> SatelliteTLE:
    ...
    raise AstraError("Not found")  # on failure, or return []
```

---

## 8. Dataclass Rules

```python
# All dataclasses are frozen
@dataclass(frozen=True)
class MyModel:
    field: type       # No default
    optional: Optional[type] = None    # With default: field = None

# Do NOT use __post_init__ for validation in frozen dataclasses
# Use factory functions instead:
def make_debris_object(tle: SatelliteTLE) -> DebrisObject:
    ...
```

---

## 9. NumPy Style Rules

```python
# Array shape comments are MANDATORY for all ndarray parameters:
def compute(
    positions: np.ndarray,   # shape (T, 3), dtype float64, unit km, frame TEME
) -> np.ndarray:             # shape (T,), unit km

# Use axis keyword explicitly for clarity:
distances = np.linalg.norm(diff, axis=1)   # NOT np.linalg.norm(diff)

# Use np.nan for missing/invalid values, never -1 or 0 as sentinel:
positions[error_mask] = np.nan   # CORRECT
positions[error_mask] = -1.0     # WRONG

# Never construct ndarray in a loop:
# WRONG:
result = []
for row in data:
    result.append(compute(row))
array = np.array(result)

# RIGHT:
array = np.vectorize(compute)(data)
# or use pre-allocated array:
array = np.empty((n, 3), dtype=np.float64)
array[:] = compute_all(data)  # vectorized
```

---

## 10. Docstring Standard

All PUBLIC functions use Google-style docstrings:

```python
def function_name(
    param1: Type1,
    param2: Type2,
) -> ReturnType:
    """Summary line (one sentence, imperative mood).

    Extended description if needed. Explain WHY, not just WHAT.
    Describe any important algorithm choices or performance notes.

    Args:
        param1: Description. Units if applicable. Constraints if any.
        param2: Description.

    Returns:
        Description of return value. Type is documented here if complex.
        For ndarray: include shape and dtype.

    Raises:
        ExceptionType: When this is raised and why.
        OtherException: Description.

    Example:
        >>> result = function_name(input1, input2)
        >>> assert ...
    """
```

Private functions (`_name`) require at minimum a one-line docstring.

---

## 11. Prohibited Patterns

```python
# PROHIBITED: Using bare except
try:
    ...
except:         # catches SystemExit, KeyboardInterrupt, etc.
    pass

# REQUIRED:
try:
    ...
except sgp4.api.SatrecError as e:
    raise PropagationError(...) from e

# PROHIBITED: String formatting with %
msg = "Error in %s" % norad_id

# REQUIRED: f-strings
msg = f"Error in {norad_id}"

# PROHIBITED: Mutable default argument
def f(items=[]):   # Bug: shared across calls

# REQUIRED:
def f(items: Optional[list] = None):
    if items is None:
        items = []

# PROHIBITED: Type: ignore comments (suppress mypy)
value: Any = compute()   # type: ignore

# FIX IT PROPERLY instead

# PROHIBITED: Global variables at module level
catalog = []   # mutable global

# PROHIBITED: print() statements in library code
print("Computing...")   # no stdout in library

# REQUIRED: Use logging
import logging
log = logging.getLogger("astra")
log.info("Computing...")
```

---

## 12. Configuration File: `ruff.toml`

Place in project root:

```toml
line-length = 100
target-version = "py310"

[lint]
select = ["E", "W", "F", "I", "N", "UP", "B", "SIM"]
ignore = ["ANN101", "ANN102"]

[format]
quote-style = "double"
indent-style = "space"
```

---

## 13. Pre-commit Hook

Add to `.pre-commit-config.yaml`:

```yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.1.0
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format
  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.0.0
    hooks:
      - id: mypy
        args: [--strict]
```
