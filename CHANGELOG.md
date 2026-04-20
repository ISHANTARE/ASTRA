# Changelog

All notable changes to **astra-core-engine** are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
The canonical version string lives in `astra/version.py` and `pyproject.toml`.

---

## [3.5.0] — 2026-04-19

### Added

- **`SNCConfig` dataclass** — `SNCConfig(q_psd_m2_s3, mode)` enables State
  Noise Compensation / process-noise injection into long-duration covariance
  propagations. Now part of the public API (`astra.SNCConfig`).
- **`warmup()` function** — `astra.warmup()` pre-compiles all Numba JIT
  kernels (Cowell force model + `SpatialIndex`) at startup, eliminating
  first-call latency in production worker pools.
- **`ASTRA_NO_BANNER=1` environment variable** — Suppresses the one-line
  startup banner printed to `stderr`; prevents log pollution in
  multi-process workers where each subprocess would otherwise emit it.
- **`parse_stk_ephemeris` public export** — Spacebook STK covariance block
  parser now accessible at the top-level `astra` namespace.
- **`xptle_to_satellite_omm` public export** — Spacebook XP-TLE → `SatelliteOMM`
  converter now in public API; also available via `astra.xptle_to_satellite_omm`.
- **`get_eop_sb` public export** — Spacebook Earth Orientation Parameter
  retrieval available at top level.
- **`validate_burn_sequence` added to `__all__`** — Was missing from the
  public export list; now accessible as `astra.validate_burn_sequence`.
- **Comprehensive API Surface Audit** — Systematically exposed 13 previously
  hidden but fully functional public symbols. The `__init__.py` and `__all__`
  now correctly export:
  - The entire **OCM parser suite** (`parse_ocm`, `parse_ocm_xml`, `parse_ocm_kvn`, `export_ocm_xml`).
  - High-fidelity **Frame Transforms** (`teme_to_ecef`, `ecef_to_geodetic_wgs84`, `get_eop_correction`).
  - **Time & Julian Date Utilities** (`prefetch_iers_data_async`, `jd_utc_to_datetime`, `datetime_utc_to_jd`, etc.).
  - The `astra.constants` module for physical/orbital parameter access.
  - The `ConjunctionDataMessage` class type for strict hinting.
- **`NumericalState.covariance_km2` field** — Optional `ndarray(6, 6)` field
  on the frozen `NumericalState` dataclass to carry propagated covariance
  alongside the kinematic state.

### Changed

- **Segmented Cowell orchestrator** — `propagate_cowell` refactored using
  Extract-Orchestrate pattern: separate `_coast_derivative` (6-DOF) and
  `_powered_derivative` (7-DOF) functions. Slices the time span at engine
  ignition / cutoff boundaries so the integrator never steps across a
  force-model discontinuity. Cyclomatic complexity significantly reduced.
- **Full 6×6 STM covariance propagation** — `propagate_covariance_stm` now
  uses the analytical J₂ partial-derivative Jacobian (`_compute_force_jacobian`,
  Montenbruck & Gill §3.2.4) alongside the two-body term. Correctly drives
  nodal-precession uncertainty growth in LEO. Co-rotating drag Jacobian
  correction preserves covariance positive-definiteness.
- **`NumericalState` is now `frozen=True`** — Prevents accidental mutation of
  integration results. Matches all other ASTRA output types (`OrbitalState`,
  `FiniteBurn`, `ConjunctionEvent`).
- **`NumericalState` position guard** — `__post_init__` now validates against
  the correct Earth equatorial radius (6378.137 km) instead of the previously
  incorrect 6000 km threshold.
- **Space-weather CSV parser** — `_parse_sw_csv` uses header-name–based
  column lookup (DATA-01) instead of hardcoded positional indices; robust to
  CelesTrak format changes.
- **RTN frame basis corrected (PHY-02)** — `_build_rtn_matrix_njit` now
  correctly computes T = N × R (not V × N). Affects all RTN-frame finite burns.
- **Structured logging on data ingestion** — `log.info`, `log.warning`, and
  `log.error` messages added throughout the data pipeline for auditability.
- **CHANGELOG format** — Migrated to [Keep a Changelog](https://keepachangelog.com/)
  with explicit `Added / Changed / Deprecated / Fixed / Removed` subsections.
- **`docs/api.rst`** — Full hand-written summary tables added for every
  capability group (ingestion, parsing, filtering, propagation, conjunction,
  maneuvers, visibility, utilities, types, errors) before the auto-generated
  module index.
- **Example `07_spacebook_pipeline.py`** — New end-to-end demonstration of
  the Spacebook pipeline: connectivity probe, XP-TLE catalog, live space
  weather + NRLMSISE-00 density comparison, EOP retrieval, synthetic 6×6
  covariance with 1-σ interpretation, and conjunction screening.

### Deprecated

- **`DragConfig.srp_cylindrical_shadow`** — The shadow model has always been
  a high-fidelity *conical* umbra/penumbra geometry. The old field name was
  misleading. `srp_cylindrical_shadow` is now a `@property` alias that emits
  `DeprecationWarning` on access. **Use `srp_conical_shadow` in new code.**

### Fixed

- **Security (AUDIT-SEC-01)** — Space-weather CSV payload capped at 2 MB;
  oversized responses are rejected to prevent memory-DoS from a compromised
  upstream proxy.
- **Security (AUDIT-SEC-02)** — F10.7 values ≤ 0.0 are rejected as
  non-physical to prevent structural failures from malicious data injection.
- **NaN guard (AUDIT-B-06)** — `NumericalState.__post_init__` position
  validation threshold corrected from 6000 km to 6378.137 km (WGS-84
  equatorial radius).
- **Sentinel return (AUDIT-C-01)** — `_srp_illumination_factor_njit` now
  has an explicit `return 1.0` sentinel after all three geometric cases
  (sunlit / umbra / penumbra) to satisfy Python's always-returns contract
  and prevent `None` in Numba-less environments.

---

## [3.4.0] — 2026-04-11

### Added

- **`py.typed` marker** — PEP 561 compliance; enables mypy strict mode for
  downstream consumers.
- **Conical Earth shadow (PHY-D)** — Upgraded SRP from cylindrical to a
  high-fidelity conical umbra/penumbra model (`_srp_illumination_factor_njit`)
  modeling fractional solar illumination via a circle-circle intersection formula.
- **Spacebook strict-mode validation** — `ASTRA_STRICT_MODE` validates physical
  units (`km/km/s`) on synthetic covariance matrices from Spacebook SynCoPate.

### Changed

- **Mass depletion (7-DOF)** — Built-in instantaneous mass tracking during
  JIT Cowell integrations via Tsiolkovsky coupling.
- **`ThreadPoolExecutor` context managers** — Standardised across all
  conjunction sweep workers to ensure clean process termination.
- **KD-tree spatial index** — `scipy.spatial.cKDTree` integrated into the
  conjunction sweep, delivering ~14.8× acceleration over naïve O(N²) screening.

---

## [3.3.0] — 2026-04-07

### Added

- **TLE + OMM unified pipeline** — `SatelliteState = Union[SatelliteTLE, SatelliteOMM]`
  accepted by all physics functions; one API surface for both formats.
- **Space-Track fetcher** — `fetch_spacetrack_*` functions with session caching
  and credential management via environment variables.
- **CelesTrak fetcher** — `fetch_celestrak_*` functions (TLE and OMM variants).
- **CDM parsing** — `parse_cdm_xml` backed by `defusedxml` for safe XML parsing.
- **Cowell propagation** — J₂–J₄ gravity, NRLMSISE-00 drag, JPL DE421
  third-body gravity (Sun/Moon), SRP with cylindrical shadow, finite burns (7-DOF).
- **Conjunction screening** — KD-tree prefilter + cubic spline TCA refinement +
  analytical and Monte Carlo P_c.
- **`ASTRA_STRICT_MODE`** — Fail-fast mode; missing data raises typed errors
  instead of silently falling back.
- **Plotly optional extra `[viz]`** — `plot_trajectories` is lazy-loaded;
  the core install stays lean for servers and CI.

### Upgrade notes (from 3.2.x)

- Install `pip install "astra-core-engine[viz]"` (or `plotly>=5.18`) if you
  use `plot_trajectories`.
- Review **Using the library responsibly** in the README and the
  **Limitations** page on Read the Docs for ephemeris span, P_c inputs, and
  strict mode.

---

## Earlier releases

Tags before 3.3.0 (e.g. 3.2.0) predate this changelog file; see Git history
and Zenodo archives for details.

[3.5.0]: https://github.com/ISHANTARE/ASTRA/compare/v3.4.0...v3.5.0
[3.4.0]: https://github.com/ISHANTARE/ASTRA/compare/v3.3.0...v3.4.0
[3.3.0]: https://github.com/ISHANTARE/ASTRA/releases/tag/v3.3.0