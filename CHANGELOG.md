# Changelog

All notable changes to **astra-core-engine** are summarized here. The canonical version string lives in `astra/version.py` and `pyproject.toml`.

## 3.4.0 — 2026-04-11

### Rigorous Audit & Core Enhancements

- **Typing Compliance**: Added `py.typed` and comprehensive typing enforcement (`Optional[Any]`).
- **High-Fidelity Space Physics**: Upgraded SRP to use a **conical Earth shadow** projection natively modeling penumbra fractional illumination.
- **Dynamic Propellant Mass**: Built-in instantaneous mass depletion tracking during JIT Cowell 7-DOF integrations.
- **Concurrency & Robustness**: Standardized `ThreadPoolExecutor` context managers ensuring clean process closures natively in conjunction sweeps.
- **Spatial Filtering Speedup**: `scipy.spatial.cKDTree` natively integrates into the collision sweep delivering ~14.8x acceleration across vast catalogs.
- **Spacebook Validation**: `ASTRA_STRICT_MODE` completely validates physical units (`km/km/s`) coming directly from empirical Synthetic Covariances in real-time.

## 3.3.0 — 2026-04-07

### Highlights

- **TLE + OMM** unified pipeline, **Space-Track** and **CelesTrak** fetchers, **CDM** parsing with `defusedxml`.
- **Cowell** propagation: J₂–J₄, empirical drag, DE421 Sun/Moon, **SRP** with optional **cylindrical Earth umbra**; finite burns (7-DOF).
- **Conjunction** screening with **KD-tree** prefilter, splines, analytical and MC **P_c**.
- `**ASTRA_STRICT_MODE`** for fail-fast behavior when data or policy requirements are not met.
- **Plotly** optional extra `**[viz]`**; core install excludes it. `**[test]**` includes pytest and Plotly for CI.

### Upgrade notes (from 3.2.x)

- Add `pip install "astra-core-engine[viz]"` (or `plotly>=5.18`) if you use `plot_trajectories`.
- Review **Using the library responsibly** in the README (and the **Limitations** page on Read the Docs) for ephemeris span, P_c inputs, and strict mode.

---

## Earlier releases

Older tags (e.g. 3.2.0) predate this changelog file in full; see Git history and Zenodo for details.