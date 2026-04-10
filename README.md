# ASTRA-Core v3.3.0 (Autonomous Space Traffic Risk Analyzer) 🛰️

![PyPI - Version](https://img.shields.io/pypi/v/astra-core-engine?color=blue&label=astra-core-engine)
[![Documentation Status](https://readthedocs.org/projects/astra-core/badge/?version=latest)](https://astra-core.readthedocs.io/en/latest/?badge=latest)
![License](https://img.shields.io/github/license/ISHANTARE/ASTRA)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19201701.svg)](https://doi.org/10.5281/zenodo.19201701)
![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)
![Python](https://img.shields.io/pypi/pyversions/astra-core-engine)

**The high-performance mathematical foundation for space situational awareness.**

ASTRA-Core is a rigorous Python astrodynamics engine for aerospace engineers, researchers, and developers. It propagates large catalogs, screens conjunctions, estimates collision probability, and predicts ground passes—using both legacy **TLE** and modern **OMM** data end-to-end.

📖 **API reference:** [Read the Docs](https://astra-core.readthedocs.io/en/latest/)

🧠 **How the math works:** [KNOWMORE.md](./KNOWMORE.md) — TLEs, OMM, SGP4, spatial screening, P_c, Cowell forces, and what the models assume.

---

## Orbital data: TLE vs OMM

ASTRA-Core supports **both** the legacy TLE format and the CCSDS **OMM (Orbit Mean-Elements Message)**. Physics APIs accept either via the unified `SatelliteState` type.

| Feature | `SatelliteTLE` (legacy) | `SatelliteOMM` (modern ★) |
|---------|-------------------------|---------------------------|
| **Source format** | 69-character text lines | JSON key-value pairs |
| **Mass (kg)** | Not in format | `mass_kg` |
| **Radar cross-section (m²)** | Not in format | `rcs_m2` |
| **Ballistic coefficient** | Not in format | `cd_area_over_mass` |
| **Collision radius** | Estimated default | From RCS / metadata when present |
| **Parsing** | Checksums, fixed columns | JSON — structured fields |
| **Backwards compatible** | Yes | Yes (same pipelines) |
| **Recommended for** | Legacy workflows | **New projects** |

> **Tip:** Prefer **OMM** when you care about drag, realistic cross-sections, or conjunction risk—the extra metadata flows into screening and Cowell without extra glue code.

---

## Data sources: Spacebook vs CelesTrak vs Space-Track

| | **Spacebook (COMSPOC)** | **CelesTrak** | **Space-Track.org** |
|---|-------------------------|---------------|---------------------|
| **Account** | Not required | Not required | Free registration |
| **Formats** | XP-TLE, STK Synthetic | TLE + OMM JSON | TLE + OMM JSON |
| **Coverage** | Highly precise objects | Large public catalogs | Authoritative catalog |
| **Updates** | Daily (live observational)| Periodic | Periodic (per provider)|
| **Notes** | Superior uncertainty/SW | Rate limits may apply | Session auth via env vars |

### Spacebook (High-Fidelity)

Spacebook provides synthetic covariance and standard/XP-TLEs along with highly precise Space Weather metrics. Spacebook overrides CelesTrak SW defaults when active.

```python
import astra

# Load XP-TLE catalog formatted transparently as SatelliteOMM with precision tags
xp_catalog = astra.fetch_xp_tle_catalog()
```

### CelesTrak (no account)

```python
import astra

tles = astra.fetch_celestrak_group("starlink")
omms = astra.fetch_celestrak_group_omm("starlink")
```

Requests identify the client as `ASTRA-Core/<version>`.

### Space-Track (environment variables)

Never hardcode passwords. Set once per machine:

```bash
# Windows (Command Prompt — restart shell after setx)
setx SPACETRACK_USER your@email.com
setx SPACETRACK_PASS yourpassword

# Linux / macOS (~/.bashrc or ~/.zshrc)
export SPACETRACK_USER=your@email.com
export SPACETRACK_PASS=yourpassword
```

```python
import astra

starlinks = astra.fetch_spacetrack_group("starlink")
catalog = astra.fetch_spacetrack_active()
```

If credentials are missing, ASTRA raises a clear error with setup hints.

---

## Key capabilities

* **Spacebook Integration:** Direct streaming of Spacebook XP-TLEs, true observational covariance matrices, and live Space Weather priorities—bypassing heuristic estimation models for flight-grade accuracy.
* **Dual format (TLE + OMM):** One API surface for parsing, propagation, filtering, and conjunctions.
* **SGP4 at scale:** Vectorized propagation (`propagate_many`, generators) with UT1-aware handling where ephemeris data are available.
* **Cowell propagation:** Dormand–Prince integration with **J₂–J₄**, empirical **drag** (space weather), **Sun/Moon** third-body gravity (**JPL DE421**), high-fidelity **solar radiation pressure** with **conical Earth shadow** (continuous penumbra modeling), and **7-DOF** finite burns with mass flow.
* **Conjunction screening:** KD-tree prefilter over time steps (~14.8x speedup), spline refinement for TCA, Spacebook EOP coordinate mapping, and dynamic effective radius from metadata when available.
* **Collision probability:** Analytical (Chan/Foster lineage), exact **2D Gaussian Quadrature** (`dblquad`), and **6D Monte Carlo** paths when full covariances are supplied; Seamless integration with Spacebook synthetic covariance matrices.
* **Catalog ingestion:** CelesTrak and Space-Track helpers plus local **OMM** files.
* **Pass prediction:** TEME → ground observer pipeline (ENU), coarse grid + refinement for AOS/TCA/LOS.
* **Optional 3D plots:** Interactive Plotly figures via the **`[viz]`** extra—core install stays lean for servers and CI.

---

## Installation

**Default (core physics, no Plotly):**

```bash
pip install astra-core-engine
```

**With 3D trajectory plotting:**

```bash
pip install "astra-core-engine[viz]"
```

**From source (development + tests):**

```bash
git clone https://github.com/ISHANTARE/ASTRA.git
cd ASTRA
pip install -e ".[test]"
```

Requires **Python 3.10+**. Core dependencies include NumPy, SciPy, Skyfield, SGP4, Requests, Numba, and defusedxml.

---

## Using the library responsibly

ASTRA-Core implements widely used models suitable for **research, education, integration prototypes, and operations-style workflows** when you understand the assumptions. It is **not** a certified conjunction or mission-closure product by itself—validate against your own requirements and reference tools if needed.

| Topic | What to know |
|-------|----------------|
| **Sun/Moon ephemeris** | Default kernel is **DE421** (roughly **1900–2050**). Very long or future-dated studies may need another ephemeris (e.g. DE440) and your own validation. |
| **Atmosphere** | Empirical **Jacchia-class** density, not NRLMSISE. Not intended for detailed re-entry or the densest LEO regimes alone. |
| **SRP** | Simple **cannonball** model; enhanced with high-fidelity **conical Earth shadow** capable of smoothly modeling fractional illumination through the penumbra. |
| **P_c** | Depends on **covariance quality**. Built-in `estimate_covariance()` is a **rough heuristic**—for serious thresholds, use **CDM-class covariances**. Turn on **strict mode** to avoid silent fallbacks. |
| **Monte Carlo P_c** | Uses a **straight-line** relative-motion model per sample; very **slow** co-orbital encounters need careful interpretation and finer time sampling. |
| **Catalog quality** | Stale or poor elements dominate error—always check epoch and data source. |

**Strict mode:** `astra.set_strict_mode(True)` or `astra.config.ASTRA_STRICT_MODE = True` makes many missing-data paths **raise** instead of warn-and-continue—recommended when building tools that must not guess.

More detail: [KNOWMORE.md](./KNOWMORE.md) and the **Limitations** page on [Read the Docs](https://astra-core.readthedocs.io/en/latest/).

---

## Quickstart

### TLE workflow

```python
import astra
import numpy as np

active_catalog = astra.fetch_celestrak_active()
objects = [astra.make_debris_object(tle) for tle in active_catalog]
leo_only = astra.filter_altitude(objects, min_km=200, max_km=2000)

sources = [obj.source for obj in leo_only]
times_jd = leo_only[0].source.epoch_jd + np.arange(0, 120, 5.0) / 1440.0
trajectories = astra.propagate_many(sources, times_jd)

events = astra.find_conjunctions(
    trajectories,
    times_jd=times_jd,
    elements_map={obj.source.norad_id: obj for obj in leo_only},
    threshold_km=5.0,
)
```

### OMM workflow (recommended for new code)

```python
import astra
import numpy as np

omm_catalog = astra.fetch_celestrak_active_omm()
# Or: omm_catalog = astra.load_omm_file("catalog.json")

objects = [astra.make_debris_object(omm) for omm in omm_catalog]
leo_only = astra.filter_altitude(objects, min_km=200, max_km=2000)

sources = [obj.source for obj in leo_only]
times_jd = leo_only[0].source.epoch_jd + np.arange(0, 120, 5.0) / 1440.0
trajectories = astra.propagate_many(sources, times_jd)

events = astra.find_conjunctions(
    trajectories,
    times_jd=times_jd,
    elements_map={obj.source.norad_id: obj for obj in leo_only},
    threshold_km=5.0,
)
print(f"Found {len(events)} conjunction events.")
```

### Space-Track catalog

```python
import astra

catalog = astra.fetch_spacetrack_active()
print(f"Loaded {len(catalog)} satellites.")
```

### Optional: Plotly (`[viz]` installed)

```python
from astra import plot_trajectories

fig = plot_trajectories({"25544": positions_array})
```

---

## Library API cheatsheet

Functions are available from the `astra` namespace.

### CelesTrak

| Function | Returns |
|----------|---------|
| `fetch_celestrak_active()` | `list[SatelliteTLE]` |
| `fetch_celestrak_group(group)` | `list[SatelliteTLE]` |
| `fetch_celestrak_comprehensive()` | `list[SatelliteTLE]` |
| `fetch_celestrak_active_omm()` | `list[SatelliteOMM]` |
| `fetch_celestrak_group_omm(group)` | `list[SatelliteOMM]` |
| `fetch_celestrak_comprehensive_omm()` | `list[SatelliteOMM]` |

### Space-Track

| Function | Returns |
|----------|---------|
| `fetch_spacetrack_group(group, format=...)` | OMM (default) or TLE |
| `fetch_spacetrack_active()` | Active catalog |
| `fetch_spacetrack_satcat()` | SATCAT-style records |
| `spacetrack_logout()` | End session |

### Spacebook (COMSPOC)

| Function | Returns |
|----------|---------|
| `fetch_xp_tle_catalog()` | Spacebook XP-TLE active subset |
| `fetch_historical_tle(date)` | Historical TLEs |
| `fetch_synthetic_covariance_stk(id)` | STK 6x6 observational errors |
| `get_space_weather_sb(jd)` | COMSPOC live SW parameters |

### OMM

* `parse_omm_json(text)` → `list[SatelliteOMM]`
* `parse_omm_record(dict)` → `SatelliteOMM`
* `load_omm_file(path)` → `list[SatelliteOMM]`
* `validate_omm(dict)` → `bool`

### TLE

* `load_tle_catalog(lines)` → `list[SatelliteTLE]`
* `parse_tle(name, l1, l2)` → `SatelliteTLE`
* `validate_tle(name, l1, l2)` → `bool`

### Filtering & debris

* `make_debris_object(source)` — `SatelliteTLE` or `SatelliteOMM`
* `filter_altitude`, `filter_region`, `filter_time_window`, `apply_filters`, `catalog_statistics`

### Propagation

* `propagate_orbit`, `propagate_many`, `propagate_many_generator`, `propagate_trajectory`, `ground_track`
* `propagate_cowell` — numerical Cowell + `DragConfig`

### Conjunctions & probability

* `find_conjunctions`, `closest_approach`, `distance_3d`
* `compute_collision_probability`, `compute_collision_probability_mc`
* `estimate_covariance`, `propagate_covariance_stm`, `rotate_covariance_rtn_to_eci`
* `parse_cdm_xml`

### Space weather & ephemeris helpers

* `get_space_weather`, `load_space_weather`, `atmospheric_density_empirical`
* `sun_position_de`, `moon_position_de`, etc.

### Visibility

* `passes_over_location`, `visible_from_location`

### Utilities & config

* `convert_time`, `vincenty_distance`, `orbit_period`, `orbital_elements`
* `set_strict_mode`, `astra.config.ASTRA_STRICT_MODE`

---

## Examples

| Script | Topic |
|--------|--------|
| `examples/01_basic_conjunctions.py` | Collision screening pipeline |
| `examples/02_visualize_swarm.py` | 3D LEO constellation plot |
| `examples/03_ground_station_visibility.py` | Pass prediction |
| `examples/04_omm_pipeline.py` | OMM end-to-end |
| `examples/05_compare_tle_omm.py` | TLE vs OMM |
| `examples/06_spacetrack_pipeline.py` | Space-Track |

---

## Changelog

Release notes: [CHANGELOG.md](./CHANGELOG.md).

---

## How to cite

```bibtex
@software{Tare_ASTRA_2026,
  author = {Tare, Ishan},
  title = {ASTRA: Autonomous Space Traffic Risk Analyzer},
  year = {2026},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/ISHANTARE/ASTRA}},
  version = {3.3.0}
}
```

---

## Author

**ASTRA Team — Ishan Tare**

© 2026 ASTRA Project · MIT License
