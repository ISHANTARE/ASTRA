# ASTRA-Core v3.0.0 (Autonomous Space Traffic Risk Analyzer) 🛰️

![PyPI - Version](https://img.shields.io/pypi/v/astra-core-engine?color=blue&label=astra-core-engine)
![License](https://img.shields.io/github/license/ISHANTARE/ASTRA)
![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)

**The High-Performance Mathematical Foundation for Space Situational Awareness.**

ASTRA-Core is the elite computational Python library powering the ASTRA ecosystem. Designed for aerospace engineers, researchers, and developers, it solves the complex, heavy-lifting astrodynamics required to track thousands of orbital objects simultaneously, predict collisions, and monitor congestion in Low Earth Orbit (LEO).

> 🧠 **Want to learn how the math works?** Check out our educational guide: [KNOWMORE.md](./KNOWMORE.md) to understand TLEs, SGP4, Sweep-and-Prune, and Collision Probabilities!

---

## 🚀 Key Features

* **High-Fidelity Cowell Method Propagation**: Integrate the exact equations of motion (DOP853) with an elite force model evaluating $J_2-J_4$ zonal harmonics, Atmospheric Drag, and Solar/Lunar third-body perturbations.
* **Maneuver Modeling & 7-DOF Flight Dynamics**: Formulate exact finite continuous burns using attitude-steered Dynamic VNB/RTN direction combinations with coupled mass expulsion tracking (Tsiolkovsky equation) directly in the integration loop.
* **Operations-Grade Physical Truth Pipelines**: Ditch analytical physics approximations for real-world automated feeds: JPL DE421 (Sub-arcsecond Moon/Sun Ephemerides) and CelesTrak Space Weather (F10.7/Kp data scaling Jacchia-class empirical atmospheric density models).
* **Temporal Octree Conjunction Analysis**: Implements a highly optimized, persistent 3D $O(n \log n)$ Temporal Octree spatial index to uniquely isolate candidate colliding trajectories across massive time integrations.
* **Continuous Time of Closest Approach (TCA)**: Uses interpolations to find the exact millisecond of closest approach, coupled with Dynamic LVLH Attitude Modes to project satellite cross-sections precisely at the impact geometry.
* **True Probability of Collision ($P_c$)**: Executes a true 6D minimum-distance Monte Carlo probability distribution across colliding volumes, propagated physically via a full 6x6 State Transition Matrix built natively from numerical force Jacobians.
* **Official Data Integration**: Directly parses active catalogs from CelesTrak and reads official U.S. Space Force CDM (Conjunction Data Message) XMLs.
* **Pass Predictions**: Calculate topocentric geometry to find when a satellite will be visible from a specific ground station.

---

## 📦 Installation

Available natively on PyPI for immediate use in your Python projects:

```bash
pip install astra-core-engine
```

**For development & contribution:**
If you want to modify the source code or run the test suite:

```bash
git clone https://github.com/ISHANTARE/ASTRA.git
cd ASTRA
pip install -e .[test]
```

---

## 💻 Technical Quickstart

Here is how you can use ASTRA-Core to fetch live satellite data and predict close calls within minutes.

### 1. Fetching Data and Mass Propagation

```python
import astra
import numpy as np

# 1. Fetch live TLEs from CelesTrak
print("Downloading live active satellite catalog...")
active_catalog = astra.fetch_celestrak_active()

# 2. Filter for Low Earth Orbit (LEO) only
objects = [astra.make_orbit_object(tle) for tle in active_catalog]
leo_only = astra.filter_altitude(objects, min_km=200, max_km=2000)

# 3. Propagate 10,000+ objects simultaneously across the next 2 hours
tles = [obj.tle for obj in leo_only]
time_steps = np.arange(0, 120, 5.0) # Minutes since Epoch
trajectories = astra.propagate_many(tles, time_steps)
```

### 2. Detecting Conjunctions (Collisions)

```python
# Scan for any satellites coming within 5km of each other
events = astra.find_conjunctions(
    trajectories, 
    time_points=leo_only[0].tle.epoch_jd + (time_steps / 1440.0), 
    catalog_map={obj.tle.norad_id: obj for obj in leo_only}, 
    threshold_km=5.0
)

for event in events:
    print(f"THREAT: SAT {event.primary_id} vs SAT {event.secondary_id}")
    print(f"Distance: {event.min_distance_km:.2f} km at TCA: {event.tca}")
```

---

## 📚 Library API Reference

ASTRA-Core is logically divided into highly specialized modules. We recommend reading the docstrings within the codebase for deep-dive argument types and return structures.

* **`astra.orbit`**: The SGP4 engine implementation, orbital state vectors, and trajectory arrays.
* **`astra.conjunction`**: Spline-based TCA finding, distance thresholds, and bounding-box spatial filters.
* **`astra.covariance`**: B-Plane mapping, error ellipsoid projections, and Mahalanobis probability logic.
* **`astra.maneuver`**: Inertial frame transformations (VNB/RTN) and instantaneous attitude-steered thrust computation.
* **`astra.visibility`**: Topocentric observer geometry (Pass Predictions/AER calculations from Lat/Lon).
* **`astra.data` & `astra.data_pipeline`**: Automated CelesTrak fetching, formal CDM/OMM active integration, dynamic live Space Weather, and NASA Skyfield integrations.
* **`astra.cdm`**: Specialized parsers for CCSDS Conjunction Data Message XMLs.

---

## 🏛️ Project Architecture

ASTRA has evolved! Previously, this repository contained both the core engine and an implicit frontend UI ("monolithic"). We have restructured to a much cleaner **microservice design**.

This repository (`ISHANTARE/ASTRA`) is now **exclusively** the high-performance Python astrodynamics library (`ASTRA-Core`), enabling standard Python tooling, robust testing via GitHub Actions, and seamless PyPI distribution. The WebGL visualizer is maintained as a separate, decoupled frontend application.

---

## 🚀 Examples

Want to see the math in action? Check out the `examples/` directory included in the repository source code:

* `examples/conjunction_demo.py` - Full collision prediction pipeline.
* `examples/visibility_demo.py` - When will the ISS pass over your specific coordinates?
* `examples/b_plane_demo.py` - Generating B-Plane probability analysis matrices.

---

## 👤 Author

**ISHAN TARE**  
*Computer Science Student*

© 2026 ASTRA Project
