# 07 — ASTRA Core: External Dependencies

---

## 1. Dependency Philosophy

ASTRA Core uses the **minimum viable set of dependencies**: only libraries that are:
- Battle-tested in aerospace/scientific computing
- Maintained by established organizations (NASA, Skyfield community, NumPy team)
- Available on PyPI with stable versioning
- Free of platform-specific proprietary code

**Strict Rule:** New dependencies must be approved before adding. No web framework, database driver, or I/O library may be added to the core library.

---

## 2. Required Dependencies

### 2.1 `sgp4`

| Property | Value |
|---|---|
| **Package** | `sgp4` |
| **PyPI** | https://pypi.org/project/sgp4/ |
| **Maintained By** | Brandon Rhodes |
| **Minimum Version** | 2.21 |
| **Purpose** | SGP4/SDP4 orbital propagation from TLE data |

**What ASTRA Uses:**
- `sgp4.io.twoline2rv()` — Initialize satellite record from two TLE lines
- `sgp4.api.Satrec` — Modern C-based satellite record (preferred over legacy Satrec)
- `sgp4.api.sgp4_array()` — **Vectorized** propagation over time array (critical for performance)
- `sgp4.api.jday()` — Julian Date calculation utility

**Integration Rules:**
- ALWAYS use `Satrec.twoline2rv()` from `sgp4.api` (C extension, fastest version)
- ALWAYS use `sgp4_array()` for batch propagation — NEVER loop over timesteps
- NEVER implement SGP4 equations manually
- Replace `sgp4.io.twoline2rv()` (Python version) with `sgp4.api.Satrec` (C version)

**Key Data:**
- SGP4 output position: TEME frame, km
- SGP4 output velocity: TEME frame, km/s
- Error codes: 0 = success, 1–6 = various failure modes

---

### 2.2 `skyfield`

| Property | Value |
|---|---|
| **Package** | `skyfield` |
| **PyPI** | https://pypi.org/project/skyfield/ |
| **Maintained By** | Brandon Rhodes |
| **Minimum Version** | 1.46 |
| **Purpose** | Astronomical coordinate transforms, Earth satellite observation, time handling |

**What ASTRA Uses:**
- `skyfield.api.load.timescale()` — Build time objects from Julian Dates
- `skyfield.api.EarthSatellite` — High-level satellite wrapper for visibility calculations
- `skyfield.api.wgs84.latlon()` — Build ground observer location
- `skyfield.positionlib` — Coordinate difference and topocentric altaz
- `skyfield.api.wgs84.subpoint()` — Convert geocentric position to geodetic lat/lon
- TEME → GCRS → ITRS coordinate transformations (internal to skyfield)

**Integration Rules:**
- Use skyfield ONLY in `astra.orbit.ground_track()` and `astra.visibility` module
- Do NOT use skyfield inside `astra.conjunction` or `astra.debris`
- skyfield's `EarthSatellite` internally uses sgp4 — do not double-propagate
- For bulk propagation (`propagate_many`), use `sgp4.api.sgp4_array()` directly — NOT `skyfield.EarthSatellite`
- Use skyfield's timescale for all time conversions requiring UT1/TDT distinctions

**Performance Note:** skyfield's `EarthSatellite.at()` is not vectorized at scale. Use it only for visibility (single satellite, fine-grained pass detection). For bulk trajectory generation, use `sgp4_array` directly.

---

### 2.3 `numpy`

| Property | Value |
|---|---|
| **Package** | `numpy` |
| **PyPI** | https://pypi.org/project/numpy/ |
| **Minimum Version** | 1.24 |
| **Purpose** | Vectorized array computation, linear algebra, mathematical operations |

**What ASTRA Uses:**
- `np.ndarray` — Primary data structure for all trajectory arrays
- `np.linalg.norm(axis=1)` — Vectorized 3D distance computation
- `np.argmin`, `np.argmax` — Time-of-closest-approach detection
- `np.arange`, `np.linspace` — Time array generation
- `np.isnan`, `np.where` — Masking and filtering on arrays
- `np.stack`, `np.concatenate` — Array assembly from propagation results

**Integration Rules:**
- ALL distance calculations must operate on full 2D arrays — NEVER loop over time
- Trajectory arrays are always `dtype=float64` (double precision)
- Use `axis=1` for per-row operations on `(T, 3)` arrays
- Use `np.nan` to mark propagation errors in position arrays

---

## 3. Development Dependencies

These are used only for testing and development, NOT in the distributed package:

| Package | Version | Purpose |
|---|---|---|
| `pytest` | ≥ 7.0 | Test runner |
| `pytest-cov` | ≥ 4.0 | Code coverage |
| `mypy` | ≥ 1.0 | Static type checking |
| `ruff` | ≥ 0.1 | Linting and formatting |
| `hypothesis` | ≥ 6.0 | Property-based testing |
| `numpy.testing` | (bundled) | Array equality assertions |

---

## 4. `pyproject.toml` Specification

```toml
[project]
name = "astra-core"
version = "0.1.0"
description = "Orbital analysis engine for space debris workflows"
requires-python = ">=3.10"
dependencies = [
    "sgp4>=2.21",
    "skyfield>=1.46",
    "numpy>=1.24",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0",
    "pytest-cov>=4.0",
    "mypy>=1.0",
    "ruff>=0.1",
    "hypothesis>=6.0",
]

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.backends.legacy:build"
```

---

## 5. Known Limitations and Workarounds

| Library | Limitation | ASTRA Workaround |
|---|---|---|
| `sgp4` | Deep-space SDP4 model not always activated automatically | Use `Satrec.twoline2rv()` which handles both SGP4 and SDP4 based on period |
| `skyfield` | `EarthSatellite.at()` loops internally for some transforms | Use `sgp4_array()` directly for bulk propagation |
| `numpy` | Float32 may cause orbital precision loss | Always use `float64` for position arrays |
| `sgp4` | Error code 6 (satellite decayed) on very old TLEs | Filter by TLE age in `filter_time_window()` before propagating |

---

## 6. Coordinate Frame Summary by Library

| Operation | Library Used | Output Frame |
|---|---|---|
| TLE propagation (bulk) | `sgp4.api.sgp4_array` | TEME |
| Topocentric visibility | `skyfield.EarthSatellite` | AzEl (Observer-relative) |
| Ground track (lat/lon) | `skyfield.wgs84.subpoint` | Geodetic WGS84 |
| Time conversion | `skyfield.api.timescale` | UT1/TT/UTC consistent |
| Frame transforms | `skyfield` internal | TEME → GCRS → ITRS |
