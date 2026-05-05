# Changelog

All notable changes to **astra-core-engine** are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).\
This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).\
The canonical version string lives in `file astra/version.py` and `file pyproject.toml`.

---


## [3.6.1] — 2026-05-05

### ⚠️ Breaking Changes

- **`ASTRA_STRICT_MODE` is now `True` by default.** This matches the library's 
  "flight-grade" documentation claims. Previously silent failures will now raise
  exceptions. Users relying on relaxed behavior should set:
  ```python
  astra.config.ASTRA_STRICT_MODE = False
  ```
  or via environment: `export ASTRA_STRICT_MODE=false`

### Added

- **`SGP4ErrorCode` enum** in `astra.errors` — Documents all SGP4 error codes
  with human-readable descriptions. Use `SGP4ErrorCode.describe()` for error messages.
  ```python
  from astra.errors import SGP4ErrorCode
  if state.error_code != SGP4ErrorCode.OK:
      print(SGP4ErrorCode(state.error_code).describe())
  ```

- **`geodetic_to_ecef_wgs84()` function** in `astra.frames` — Inverse coordinate
  transform for round-trip geodetic ↔ ECEF conversions. Enables:
  ```python
  lat, lon, alt = ecef_to_geodetic_wgs84(x, y, z)
  x, y, z = geodetic_to_ecef_wgs84(lat, lon, alt)  # NEW
  ```

- **Illumination status in `PassEvent`** — Two new fields:
  - `satellite_illuminated: bool` — True if satellite is sunlit at TCA
  - `observer_in_darkness: bool` — True if observer is in Earth's shadow at TCA

- **`azimuth_at_tca_deg` in `PassEvent`** — Azimuth at Time of Closest Approach
  (previously only AOS and LOS azimuths were available).

- **Thread-safe banner** — Startup banner now uses `threading.Lock()` to prevent
  race conditions in multi-threaded environments.

### Changed

- **EOP fetch failures now raise in strict mode** — Previously returned zeros silently,
  causing up to 35km ground-track errors. Now raises `EphemerisError` in strict mode.

- **Illumination computation failures return safe defaults** — Now returns `(False, False)`
  instead of `(True, False)`, preventing false claims of visibility when uncertain.

- **Monte Carlo Pc errors logged and raised** — Previously swallowed silently. Now logs
  in relaxed mode and raises `ValueError` in strict mode.

- **Covariance dimension mismatch raises in strict mode** — Previously only warned and
  set Pc to None. Now raises `ValueError` when mixing 3x3 and 6x6 covariances.

### Fixed

- **Phantom test `assert True`** in `test_new_defect_fixes.py` — Replaced with actual
  verification that frames module loaded correctly with expected functions.

### Security

- **Stricter error handling** — Many previously silent failures now surface errors,
  preventing hidden data quality issues from corrupting analysis results.

---


## [3.6.0] — 2026-04-25

### Added

- **`file tests/test_audit_remediation.py`** — 9-class, 30-test deterministic offline\
  regression suite covering every finding from the post-v3.6.0 audit.\
  Tests assert physical invariants, not just types or list lengths.

- `astra.run_conjunction_sweep` — High-level conjunction pipeline orchestrator\
  (see Finding #8 above). Exported at top-level namespace and added to `__all__`.

- `astra.parse_cdm_kvn` / `astra.export_cdm_kvn` — KVN CDM read-write pair\
  (see Finding #9 above). Both exported at top-level namespace and added to `__all__`.

- `_srp_illumination_factor_dual_cone_njit` — New Numba-compiled exact\
  spherical-cap intersection SRP shadow function (FM-1B Fix). Implements\
  Montenbruck & Gill §3.4.2 and Vallado Algorithm 34. Handles full\
  sunlight, umbra, penumbra, and annular eclipse cases with C¹-continuous\
  transitions.

- `_srp_illumination_factor_planar_njit` — Old planar circle-circle\
  formula preserved under a new name for regression comparison in tests.

- **`file tests/test_srp_shadow_upgrade.py`** — Comprehensive 8-class test suite:\
  boundary conditions, monotonicity, C¹ derivative continuity, multi-regime\
  accuracy (LEO/GEO/HEO), annular eclipse, degenerate geometry guards, and\
  Monte Carlo solid-angle validation.

- `SPACEBOOK_ENABLED` **flag in** `astra.config` — Centralises the\
  `ASTRA_SPACEBOOK_ENABLED` environment variable read into a single\
  authoritative location (CF-6 Fix). All Spacebook-guarded modules\
  now read from `astra.config.SPACEBOOK_ENABLED` instead of calling\
  `os.environ.get(...)` individually.

- `set_spacebook_enabled(enabled)` **public API** — Runtime toggle for\
  Spacebook I/O (CF-6 Fix). Thread-safe; mirrors `set_strict_mode()`.\
  Exposed as `astra.set_spacebook_enabled`. Enables clean offline/CI\
  testing without environment-variable workarounds.

- `R_GAS`**,** `G0_STD_KM_S2` **constants** — Two new physical scalars in\
  `astra.constants` that document and guard the Numba-inlined values\
  previously only present as unexplained literals in the propagator\
  (FM-9A Fix).

- **`file test_v360_audit_fixes.py`** — New regression test module covering\
  all six v3.6.0 audit findings with deterministic, offline unit tests.

### Changed

- `srp_illumination_factor` **/** `srp_illumination_factor_njit` — Public\
  wrappers now delegate to the dual-cone algorithm (FM-1B Fix).
- **`file tests/test_phy_d_penumbra_smoothness.py`** — Extended with four new\
  tests: HEO correctness, GEO correctness, C¹ smoothness assertion, and\
  planar-vs-dual-cone 1% agreement check in LEO.
- **`file spacebook.py` Spacebook guard** — Replaced module-level constant\
  `SPACEBOOK_ENABLED` with a `_spacebook_enabled()` function that reads\
  from `astra.config` on every call. Runtime overrides via\
  `set_spacebook_enabled()` now propagate immediately (CF-6 Fix).
- **`file frames.py` EOP guard** — Both `get_eop_correction` and `teme_to_ecef`\
  now read `SPACEBOOK_ENABLED` from `astra.config` instead of calling\
  `os.environ.get(...)` inline (CF-6 Fix).
- **`file test_orbit.py` upgraded to physical invariants** — All shape-only\
  "phantom tests" replaced with vis-viva velocity checks, radius-never-\
  subterranean assertions, and `propagate_many` / `propagate_orbit`\
  cross-validation to 1 m accuracy (FM-5 Fix). Added three new tests:\
  `test_propagate_many_radius_never_subterranean`,\
  `test_propagate_many_agrees_with_propagate_orbit`,\
  `test_propagate_trajectory_radius_bounds`,\
  `test_ground_track_altitude_iss_leo`.

### Fixed

- **Finding #1 — FM-1 (Critical): STM covariance failure always raises** `PropagationError`**.**\
  `propagate_covariance_stm` previously silently returned the initial covariance when\
  `solve_ivp` failed in non-STRICT mode. This froze uncertainty at epoch, producing\
  artificially low Pc values and life-threatening false-negative miss classifications.\
  The fallback path is now unconditionally removed — a failed STM always raises\
  `PropagationError` with a clear diagnosis. (`file astra/covariance.py`)

- **Finding #2 — FM-3 (Critical): STM Jacobian now honours** `use_nrlmsise=True`**.**\
  `_acceleration_njit` (the Cowell-STM drag kernel) previously used the exponential\
  density profile even when `DragConfig.model == "NRLMSISE00"` was passed to the Cowell\
  propagator. This caused the trajectory and its covariance to diverge physically by\
  3-5× during solar maximum. Fixed by calling `_nrlmsise00_density_njit` inside the\
  Numba kernel when `use_nrlmsise=True`. (`file astra/covariance.py`)

- **Finding #3 — FM-9 (High): Constants leakage in propagator signatures resolved.**\
  `srp_illumination_factor` default argument `earth_radius_km=6378.137` replaced with\
  `EARTH_EQUATORIAL_RADIUS_KM` from `file constants.py`. Numba `@njit` functions retain\
  inlined literals (Numba constraint) — these are already covered by compile-time\
  `assert` guards in `file constants.py`. (`file astra/propagator.py`)

- **Finding #4 — FM-1 (High): UT1-UTC fallback warning now quantifies spatial error.**\
  All three UT1-UTC fallback paths in `file orbit.py` (`propagate_orbit`, `propagate_many`,\
  `propagate_many_generator`) now log a WARNING that explicitly states: "up to \~400 m\
  of along-track position error at LEO velocities" and directs users to\
  `astra.data_pipeline.load_eop_data()` for remediation. (`file astra/orbit.py`)

- **Finding #5 — FM-6 (High):** `export_ocm_xml` **was already implemented.**\
  Audit finding was stale — `export_ocm_xml` exists in `file astra/ocm.py` (line 494) and is\
  correctly exported at `astra.export_ocm_xml`. No code change needed; confirmed by\
  round-trip test in `file test_audit_remediation.py`.

- **Finding #6 — FM-2 (Medium): Silent** `pass` **in** `fetch_celestrak_comprehensive` **replaced.**\
  A bare `except AstraError: pass` silently produced a partial catalog without any\
  indication of the data gap. Replaced with `logger.warning(...)` identifying the\
  failing group, stating the catalog is INCOMPLETE, and directing users to retry.\
  (`file astra/data.py`)

- **Finding #7 — FM-5 (Medium): Phantom tests hardened with orbital element assertions.**\
  `TestFetchSpacetrackOMM.test_fetch_group_returns_omm_list` and siblings previously\
  only checked `isinstance`. A zeroed `SatelliteOMM` would pass. Now asserts\
  `norad_id`, `inclination_rad` (against `math.radians(51.6442)`), `mean_motion_rad_min`,\
  `eccentricity`, and `epoch_jd` against the known mock payload values.\
  (`file tests/test_spacetrack.py`)

- **Finding #8 — FM-7 (Medium):** `run_conjunction_sweep` **high-level pipeline added.**\
  New public function `astra.run_conjunction_sweep(catalog, t_start_jd, t_end_jd)`\
  collapses the previous 5-step manual orchestration (fetch → DebrisObject map →\
  time grid → `propagate_many` → `find_conjunctions`) into a single call.\
  Handles NaN trajectory filtering, catalog-to-DebrisObject conversion, and all\
  internal wiring. Full docstring with usage example included.\
  (`file astra/conjunction.py`, `file astra/__init__.py`)

- **Finding #9 — FM-4 (Low):** `export_cdm_kvn` **+** `parse_cdm_kvn` **implemented.**\
  The library could parse both XML and KVN CDMs but only export XML, breaking\
  interoperability with legacy ground systems (ODAS, older STK pipelines).\
  Added `export_cdm_kvn(cdm, originator)` producing CCSDS 508.0-B-1 KVN and\
  `parse_cdm_kvn(kvn_string)` with full physical validation (negative miss distance,\
  out-of-range Pc). Both are exported at the `astra` top-level namespace.\
  Round-trip tested: `parse_cdm_kvn(export_cdm_kvn(cdm)) == cdm`. (`file astra/cdm.py`)

- **FM-1B (SRP Shadow Model Limitations)** — Replaced the planar\
  circle-circle intersection approximation with exact spherical-cap geometry.\
  Key improvements:

  - Error reduced from O(α²) (up to 8% at HEO) to floating-point precision\
    at all orbital altitudes.
  - C¹-continuous dν/dγ eliminates the force impulse artifact the planar\
    formula injected into the DOP853 integrator at shadow entry/exit.
  - Annular eclipse case (Sun apparent radius &gt; Earth) now correctly returns\
    partial illumination instead of ν = 0.

- **CF-6 (Decentralized Environment Variable Reads)** — `ASTRA_SPACEBOOK_ENABLED`\
  is now read exactly once at import time in `astra.config`. All other\
  modules delegate to `astra.config.SPACEBOOK_ENABLED`.

- **FM-9A (Constant Leakage — Numba Kernels)** — Added `assert_guards`\
  in `file constants.py` for every Numba-inlined literal that lacked one:\
  `J2` / `J2c`, `EARTH_MU_KM3_S2`, `G0_STD_KM_S2` (km/s² variant),\
  `R_GAS`, and `SUN_RADIUS_KM`. Each assertion fires at Python import\
  time and catches drift immediately.

- **CF-7 (Incomplete Strict Mode Coverage)** — `propagate_covariance_stm`\
  now raises `PropagationError` when `ASTRA_STRICT_MODE=True` and the\
  space-weather lookup fails, instead of silently falling back to the\
  reference density. In relaxed mode, a descriptive `WARNING` is logged\
  explaining the degradation.

- **FM-5 (Phantom Tests)** — `file test_orbit.py` completely rewritten: every\
  test now validates at least one physical invariant rather than a bare\
  type or length assertion.

---

## [3.5.0](https://github.com/ISHANTARE/ASTRA/compare/v3.4.0...v3.5.0) — 2026-04-19

### Added

- **`file tests/test_audit_remediation.py`** — 9-class, 30-test deterministic offline\
  regression suite covering every finding from the post-v3.6.0 audit.\
  Tests assert physical invariants, not just types or list lengths.

- `astra.run_conjunction_sweep` — High-level conjunction pipeline orchestrator\
  (see Finding #8 above). Exported at top-level namespace and added to `__all__`.

- `astra.parse_cdm_kvn` / `astra.export_cdm_kvn` — KVN CDM read-write pair\
  (see Finding #9 above). Both exported at top-level namespace and added to `__all__`.

- `SNCConfig` **dataclass** — `SNCConfig(q_psd_m2_s3, mode)` enables State\
  Noise Compensation / process-noise injection into long-duration covariance\
  propagations. Now part of the public API (`astra.SNCConfig`).

- `warmup()` **function** — `astra.warmup()` pre-compiles all Numba JIT\
  kernels (Cowell force model + `SpatialIndex`) at startup, eliminating\
  first-call latency in production worker pools.

- `ASTRA_NO_BANNER=1` **environment variable** — Suppresses the one-line\
  startup banner printed to `stderr`; prevents log pollution in\
  multi-process workers where each subprocess would otherwise emit it.

- `parse_stk_ephemeris` **public export** — Spacebook STK covariance block\
  parser now accessible at the top-level `astra` namespace.

- `xptle_to_satellite_omm` **public export** — Spacebook XP-TLE → `SatelliteOMM`\
  converter now in public API; also available via `astra.xptle_to_satellite_omm`.

- `get_eop_sb` **public export** — Spacebook Earth Orientation Parameter\
  retrieval available at top level.

- `validate_burn_sequence` **added to** `__all__` — Was missing from the\
  public export list; now accessible as `astra.validate_burn_sequence`.

- **Comprehensive API Surface Audit** — Systematically exposed 13 previously\
  hidden but fully functional public symbols. The `file __init__.py` and `__all__`\
  now correctly export:

  - The entire **OCM parser suite** (`parse_ocm`, `parse_ocm_xml`, `parse_ocm_kvn`, `export_ocm_xml`).
  - High-fidelity **Frame Transforms** (`teme_to_ecef`, `ecef_to_geodetic_wgs84`, `get_eop_correction`).
  - **Time & Julian Date Utilities** (`prefetch_iers_data_async`, `jd_utc_to_datetime`, `datetime_utc_to_jd`, etc.).
  - The `astra.constants` module for physical/orbital parameter access.
  - The `ConjunctionDataMessage` class type for strict hinting.

- `NumericalState.covariance_km2` **field** — Optional `ndarray(6, 6)` field\
  on the frozen `NumericalState` dataclass to carry propagated covariance\
  alongside the kinematic state.

### Changed

- **Segmented Cowell orchestrator** — `propagate_cowell` refactored using\
  Extract-Orchestrate pattern: separate `_coast_derivative` (6-DOF) and\
  `_powered_derivative` (7-DOF) functions. Slices the time span at engine\
  ignition / cutoff boundaries so the integrator never steps across a\
  force-model discontinuity. Cyclomatic complexity significantly reduced.
- **Full 6×6 STM covariance propagation** — `propagate_covariance_stm` now\
  uses the analytical J₂ partial-derivative Jacobian (`_compute_force_jacobian`,\
  Montenbruck & Gill §3.2.4) alongside the two-body term. Correctly drives\
  nodal-precession uncertainty growth in LEO. Co-rotating drag Jacobian\
  correction preserves covariance positive-definiteness.
- `NumericalState` **is now** `frozen=True` — Prevents accidental mutation of\
  integration results. Matches all other ASTRA output types (`OrbitalState`,\
  `FiniteBurn`, `ConjunctionEvent`).
- `NumericalState` **position guard** — `__post_init__` now validates against\
  the correct Earth equatorial radius (6378.137 km) instead of the previously\
  incorrect 6000 km threshold.
- **Space-weather CSV parser** — `_parse_sw_csv` uses header-name–based\
  column lookup (DATA-01) instead of hardcoded positional indices; robust to\
  CelesTrak format changes.
- **RTN frame basis corrected (PHY-02)** — `_build_rtn_matrix_njit` now\
  correctly computes T = N × R (not V × N). Affects all RTN-frame finite burns.
- **Structured logging on data ingestion** — `log.info`, `log.warning`, and\
  `log.error` messages added throughout the data pipeline for auditability.
- **CHANGELOG format** — Migrated to [Keep a Changelog](https://keepachangelog.com/)\
  with explicit `Added / Changed / Deprecated / Fixed / Removed` subsections.
- **`file docs/api.rst`** — Full hand-written summary tables added for every\
  capability group (ingestion, parsing, filtering, propagation, conjunction,\
  maneuvers, visibility, utilities, types, errors) before the auto-generated\
  module index.
- **Example `file 07_spacebook_pipeline.py`** — New end-to-end demonstration of\
  the Spacebook pipeline: connectivity probe, XP-TLE catalog, live space\
  weather + NRLMSISE-00 density comparison, EOP retrieval, synthetic 6×6\
  covariance with 1-σ interpretation, and conjunction screening.

### Deprecated

- `DragConfig.srp_cylindrical_shadow` — The shadow model has always been\
  a high-fidelity *conical* umbra/penumbra geometry. The old field name was\
  misleading. `srp_cylindrical_shadow` is now a `@property` alias that emits\
  `DeprecationWarning` on access. **Use** `srp_conical_shadow` **in new code.**

### Fixed

- **Finding #1 — FM-1 (Critical): STM covariance failure always raises** `PropagationError`**.**\
  `propagate_covariance_stm` previously silently returned the initial covariance when\
  `solve_ivp` failed in non-STRICT mode. This froze uncertainty at epoch, producing\
  artificially low Pc values and life-threatening false-negative miss classifications.\
  The fallback path is now unconditionally removed — a failed STM always raises\
  `PropagationError` with a clear diagnosis. (`file astra/covariance.py`)

- **Finding #2 — FM-3 (Critical): STM Jacobian now honours** `use_nrlmsise=True`**.**\
  `_acceleration_njit` (the Cowell-STM drag kernel) previously used the exponential\
  density profile even when `DragConfig.model == "NRLMSISE00"` was passed to the Cowell\
  propagator. This caused the trajectory and its covariance to diverge physically by\
  3-5× during solar maximum. Fixed by calling `_nrlmsise00_density_njit` inside the\
  Numba kernel when `use_nrlmsise=True`. (`file astra/covariance.py`)

- **Finding #3 — FM-9 (High): Constants leakage in propagator signatures resolved.**\
  `srp_illumination_factor` default argument `earth_radius_km=6378.137` replaced with\
  `EARTH_EQUATORIAL_RADIUS_KM` from `file constants.py`. Numba `@njit` functions retain\
  inlined literals (Numba constraint) — these are already covered by compile-time\
  `assert` guards in `file constants.py`. (`file astra/propagator.py`)

- **Finding #4 — FM-1 (High): UT1-UTC fallback warning now quantifies spatial error.**\
  All three UT1-UTC fallback paths in `file orbit.py` (`propagate_orbit`, `propagate_many`,\
  `propagate_many_generator`) now log a WARNING that explicitly states: "up to \~400 m\
  of along-track position error at LEO velocities" and directs users to\
  `astra.data_pipeline.load_eop_data()` for remediation. (`file astra/orbit.py`)

- **Finding #5 — FM-6 (High):** `export_ocm_xml` **was already implemented.**\
  Audit finding was stale — `export_ocm_xml` exists in `file astra/ocm.py` (line 494) and is\
  correctly exported at `astra.export_ocm_xml`. No code change needed; confirmed by\
  round-trip test in `file test_audit_remediation.py`.

- **Finding #6 — FM-2 (Medium): Silent** `pass` **in** `fetch_celestrak_comprehensive` **replaced.**\
  A bare `except AstraError: pass` silently produced a partial catalog without any\
  indication of the data gap. Replaced with `logger.warning(...)` identifying the\
  failing group, stating the catalog is INCOMPLETE, and directing users to retry.\
  (`file astra/data.py`)

- **Finding #7 — FM-5 (Medium): Phantom tests hardened with orbital element assertions.**\
  `TestFetchSpacetrackOMM.test_fetch_group_returns_omm_list` and siblings previously\
  only checked `isinstance`. A zeroed `SatelliteOMM` would pass. Now asserts\
  `norad_id`, `inclination_rad` (against `math.radians(51.6442)`), `mean_motion_rad_min`,\
  `eccentricity`, and `epoch_jd` against the known mock payload values.\
  (`file tests/test_spacetrack.py`)

- **Finding #8 — FM-7 (Medium):** `run_conjunction_sweep` **high-level pipeline added.**\
  New public function `astra.run_conjunction_sweep(catalog, t_start_jd, t_end_jd)`\
  collapses the previous 5-step manual orchestration (fetch → DebrisObject map →\
  time grid → `propagate_many` → `find_conjunctions`) into a single call.\
  Handles NaN trajectory filtering, catalog-to-DebrisObject conversion, and all\
  internal wiring. Full docstring with usage example included.\
  (`file astra/conjunction.py`, `file astra/__init__.py`)

- **Finding #9 — FM-4 (Low):** `export_cdm_kvn` **+** `parse_cdm_kvn` **implemented.**\
  The library could parse both XML and KVN CDMs but only export XML, breaking\
  interoperability with legacy ground systems (ODAS, older STK pipelines).\
  Added `export_cdm_kvn(cdm, originator)` producing CCSDS 508.0-B-1 KVN and\
  `parse_cdm_kvn(kvn_string)` with full physical validation (negative miss distance,\
  out-of-range Pc). Both are exported at the `astra` top-level namespace.\
  Round-trip tested: `parse_cdm_kvn(export_cdm_kvn(cdm)) == cdm`. (`file astra/cdm.py`)

- **Security (AUDIT-SEC-01)** — Space-weather CSV payload capped at 2 MB;\
  oversized responses are rejected to prevent memory-DoS from a compromised\
  upstream proxy.

- **Security (AUDIT-SEC-02)** — F10.7 values ≤ 0.0 are rejected as\
  non-physical to prevent structural failures from malicious data injection.

- **NaN guard (AUDIT-B-06)** — `NumericalState.__post_init__` position\
  validation threshold corrected from 6000 km to 6378.137 km (WGS-84\
  equatorial radius).

- **Sentinel return (AUDIT-C-01)** — `_srp_illumination_factor_njit` now\
  has an explicit `return 1.0` sentinel after all three geometric cases\
  (sunlit / umbra / penumbra) to satisfy Python's always-returns contract\
  and prevent `None` in Numba-less environments.

---

## [3.4.0](https://github.com/ISHANTARE/ASTRA/compare/v3.3.0...v3.4.0) — 2026-04-11

### Added

- **`file tests/test_audit_remediation.py`** — 9-class, 30-test deterministic offline\
  regression suite covering every finding from the post-v3.6.0 audit.\
  Tests assert physical invariants, not just types or list lengths.

- `astra.run_conjunction_sweep` — High-level conjunction pipeline orchestrator\
  (see Finding #8 above). Exported at top-level namespace and added to `__all__`.

- `astra.parse_cdm_kvn` / `astra.export_cdm_kvn` — KVN CDM read-write pair\
  (see Finding #9 above). Both exported at top-level namespace and added to `__all__`.

- `py.typed` **marker** — PEP 561 compliance; enables mypy strict mode for\
  downstream consumers.

- **Conical Earth shadow (PHY-D)** — Upgraded SRP from cylindrical to a\
  high-fidelity conical umbra/penumbra model (`_srp_illumination_factor_njit`)\
  modeling fractional solar illumination via a circle-circle intersection formula.

- **Spacebook strict-mode validation** — `ASTRA_STRICT_MODE` validates physical\
  units (`km/km/s`) on synthetic covariance matrices from Spacebook SynCoPate.

### Changed

- **Mass depletion (7-DOF)** — Built-in instantaneous mass tracking during\
  JIT Cowell integrations via Tsiolkovsky coupling.
- `ThreadPoolExecutor` **context managers** — Standardised across all\
  conjunction sweep workers to ensure clean process termination.
- **KD-tree spatial index** — `scipy.spatial.cKDTree` integrated into the\
  conjunction sweep, delivering \~14.8× acceleration over naïve O(N²) screening.

---

## [3.3.0](https://github.com/ISHANTARE/ASTRA/releases/tag/v3.3.0) — 2026-04-07

### Added

- **`file tests/test_audit_remediation.py`** — 9-class, 30-test deterministic offline\
  regression suite covering every finding from the post-v3.6.0 audit.\
  Tests assert physical invariants, not just types or list lengths.

- `astra.run_conjunction_sweep` — High-level conjunction pipeline orchestrator\
  (see Finding #8 above). Exported at top-level namespace and added to `__all__`.

- `astra.parse_cdm_kvn` / `astra.export_cdm_kvn` — KVN CDM read-write pair\
  (see Finding #9 above). Both exported at top-level namespace and added to `__all__`.

- **TLE + OMM unified pipeline** — `SatelliteState = Union[SatelliteTLE, SatelliteOMM]`\
  accepted by all physics functions; one API surface for both formats.

- **Space-Track fetcher** — `fetch_spacetrack_*` functions with session caching\
  and credential management via environment variables.

- **CelesTrak fetcher** — `fetch_celestrak_*` functions (TLE and OMM variants).

- **CDM parsing** — `parse_cdm_xml` backed by `defusedxml` for safe XML parsing.

- **Cowell propagation** — J₂–J₄ gravity, NRLMSISE-00 drag, JPL DE421\
  third-body gravity (Sun/Moon), SRP with cylindrical shadow, finite burns (7-DOF).

- **Conjunction screening** — KD-tree prefilter + cubic spline TCA refinement +\
  analytical and Monte Carlo P_c.

- `ASTRA_STRICT_MODE` — Fail-fast mode; missing data raises typed errors\
  instead of silently falling back.

- **Plotly optional extra** `[viz]` — `plot_trajectories` is lazy-loaded;\
  the core install stays lean for servers and CI.

### Upgrade notes (from 3.2.x)

- Install `pip install "astra-core-engine[viz]"` (or `plotly>=5.18`) if you\
  use `plot_trajectories`.
- Review **Using the library responsibly** in the README and the\
  **Limitations** page on Read the Docs for ephemeris span, P_c inputs, and\
  strict mode.

---

## Earlier releases

Tags before 3.3.0 (e.g. 3.2.0) predate this changelog file; see Git history\
and Zenodo archives for details.