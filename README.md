# ASTRA-Core v3.1.2 (Autonomous Space Traffic Risk Analyzer) 🛰️

![PyPI - Version](https://img.shields.io/pypi/v/astra-core-engine?color=blue&label=astra-core-engine)
![License](https://img.shields.io/github/license/ISHANTARE/ASTRA)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19201702.svg)](https://doi.org/10.5281/zenodo.19201702)
![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)

**The High-Performance Mathematical Foundation for Space Situational Awareness.**

ASTRA-Core is the elite computational Python library powering the ASTRA ecosystem. Designed for aerospace engineers, researchers, and developers, it solves the complex, heavy-lifting astrodynamics required to track thousands of orbital objects simultaneously, predict collisions, and monitor congestion across all orbital regimes.

> 🧠 **Want to learn how the math works?** Check out our educational guide: [KNOWMORE.md](./KNOWMORE.md) to understand TLEs, SGP4, Sweep-and-Prune, and Collision Probabilities!

---

## 🚀 Key Features

* **High-Fidelity Cowell Method Propagation**: Integrate the exact equations of motion (DOP853) with an elite force model evaluating $J_2-J_4$ zonal harmonics, Atmospheric Drag, and Solar/Lunar third-body perturbations.
* **Maneuver Modeling & 7-DOF Flight Dynamics**: Formulate exact finite continuous burns using attitude-steered Dynamic VNB/RTN direction combinations with coupled mass expulsion tracking (Tsiolkovsky equation) directly in the integration loop.
* **Operations-Grade Physical Truth Pipelines**: Ditch analytical physics approximations for real-world automated feeds: JPL DE421 (Sub-arcsecond Moon/Sun Ephemerides) and CelesTrak Space Weather (F10.7/Kp data scaling Jacchia-class empirical atmospheric density models).
* **Spatial KD-Tree Conjunction Analysis**: Implements a highly optimized, persistent 3D $O(n \log n)$ Spatial KD-Tree index to uniquely isolate candidate colliding trajectories across massive time integrations.
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
objects = [astra.make_debris_object(tle) for tle in active_catalog]
leo_only = astra.filter_altitude(objects, min_km=200, max_km=2000)

# 3. Propagate 10,000+ objects simultaneously across the next 2 hours
tles = [obj.tle for obj in leo_only]
time_steps = np.arange(0, 120, 5.0) # Minutes
times_jd = leo_only[0].tle.epoch_jd + (time_steps / 1440.0)
trajectories = astra.propagate_many(tles, times_jd)
```

### 2. Detecting Conjunctions (Collisions)

```python
# Scan for any satellites coming within 5km of each other
events = astra.find_conjunctions(
    trajectories, 
    times_jd=leo_only[0].tle.epoch_jd + (time_steps / 1440.0), 
    elements_map={obj.tle.norad_id: obj for obj in leo_only}, 
    threshold_km=5.0
)

for event in events:
    print(f"THREAT: SAT {event.primary_id} vs SAT {event.secondary_id}")
    print(f"Distance: {event.min_distance_km:.2f} km at TCA: {event.tca}")
```

---

## 📚 Library API Cheatsheet (Exposed Functions)

ASTRA-Core natively exposes all top-level functions directly from `astra.__init__`. Here are all the callable functions with a syntax implementation example for each:

### Data Acquisition & Parsing

* `fetch_celestrak_active()`: `catalog = astra.fetch_celestrak_active()`
* `fetch_celestrak_comprehensive()`: `catalog = astra.fetch_celestrak_comprehensive()`
* `fetch_celestrak_group(group)`: `gnss = astra.fetch_celestrak_group("gps-ops")`
* `parse_cdm_xml(filepath)`: `cdm = astra.parse_cdm_xml("warning.xml")`
* `load_tle_catalog(filepath)`: `tles = astra.load_tle_catalog("catalog.txt")`
* `parse_tle(name, l1, l2)`: `tle = astra.SatelliteTLE.from_strings("1 255...", "2 255...", name="ISS")`
* `validate_tle(l1, l2)`: `is_valid = astra.validate_tle(line1, line2)`

### Filtering & Debris Processing

* `make_debris_object(tle)`: `obj = astra.make_debris_object(tle)`
* `filter_altitude(objs, min, max)`: `leo = astra.filter_altitude(objects, 200, 2000)`
* `filter_region(objs, lat, lon)`: `overhead = astra.filter_region(objects, lat_bounds, lon_bounds)`
* `filter_time_window(objs, t1, t2)`: `visible = astra.filter_time_window(objects, start_jd, end_jd)`
* `apply_filters(objs, config)`: `subset = astra.apply_filters(objects, filter_config)`
* `catalog_statistics(objs)`: `stats_dict = astra.catalog_statistics(objects)`

### High-Performance Propagation & Orbit Math

* `propagate_cowell(state, duration_s, ...)`: `trajectory = astra.propagate_cowell(initial_state, duration_s=7200, dt_out=60.0)`
* `propagate_many(tles, times_jd)`: `traj_map = astra.propagate_many([tle1, tle2], times_jd)`
* `propagate_many_generator(tles, times_jd)`: `for jd_chunk, traj_chunk in astra.propagate_many_generator(tles, times_jd): pass`
* `propagate_orbit(tle, epoch, t_min)`: `state = astra.propagate_orbit(tle, tle.epoch_jd, 10.0)`
* `propagate_trajectory(tle, t1, t2, step)`: `times_jd, pos = astra.propagate_trajectory(tle, start_jd, end_jd, step_minutes=5.0)`
* `ground_track(positions, times)`: `lat_lon_alt = astra.ground_track(teme_pos, times_jd)`
* `orbital_elements(pos, vel)`: `elements = astra.orbital_elements(r, v)`
* `orbit_period(semi_major_axis)`: `period_s = astra.orbit_period(a_km)`

### Conjunctions & Covariance (O(n log n) cKDTree)

* `find_conjunctions(...)`: `events = astra.find_conjunctions(trajs, times_jd, obj_map, 5.0, 50.0)`
* `closest_approach(...)`: `tca, dist = astra.closest_approach(traj_a, traj_b, times)`
* `distance_3d(pos1, pos2)`: `d = astra.distance_3d(r1, r2)`
* `compute_collision_probability(...)`: `pc = astra.compute_collision_probability(r_rel, v_rel, cov)`
* `compute_collision_probability_mc(...)`: `pc = astra.compute_collision_probability_mc(r_rel, v_rel, cov, 10000)`
* `estimate_covariance(...)`: `cov = astra.estimate_covariance(tle, position, velocity)`
* `propagate_covariance_stm(...)`: `cov_t = astra.propagate_covariance_stm(cov_0, initial_state, t_span)`

### Visibility & Ground Stations

* `visible_from_location(...)`: `elevations = astra.visible_from_location(pos, times, observer)`
* `passes_over_location(...)`: `passes = astra.passes_over_location(tle, observer, t_start, t_end)`

### High-Fidelity Physics & Maneuvers

* `projected_area_m2(dim, quat, v_rel)`: `area = astra.projected_area_m2((1,2,3), q, v_dir)`
* `thrust_acceleration_inertial(...)`: `acc = astra.thrust_acceleration_inertial(burn, mass, t, state)`
* `rotation_vnb_to_inertial(pos, vel)`: `matrix = astra.rotation_vnb_to_inertial(r, v)`
* `rotation_rtn_to_inertial(pos, vel)`: `matrix = astra.rotation_rtn_to_inertial(r, v)`
* `frame_to_inertial(frame, pos, vel)`: `matrix = astra.frame_to_inertial(ManeuverFrame.VNB, r, v)`
* `validate_burn(burn)`: `is_valid = astra.validate_burn(burn_dataclass)`

### Space Weather & Data Pipelines

* `get_space_weather(jd)`: `f107, f107a, ap = astra.get_space_weather(t_jd)`
* `load_space_weather(filepath)`: `astra.load_space_weather("SW-All.csv")`
* `atmospheric_density_empirical(...)`: `rho = astra.atmospheric_density_empirical(alt, f107, f107a, ap)`
* `sun_position_de(jd)`: `r_sun = astra.sun_position_de(t_jd)`
* `moon_position_de(jd)`: `r_moon = astra.moon_position_de(t_jd)`

### Top-Level Utilities

* `haversine_distance(l1, ln1, l2, ln2)`: `dist_km = astra.haversine_distance(34.0, -118.0, 40.0, -74.0)`
* `convert_time(time_val, to_format)`: `jd = astra.convert_time("2026-01-01T00:00:00Z", "jd")`
* `plot_trajectories(trajs, events)`: `fig = astra.plot_trajectories(trajectories, conjunction_events)`

---

## 🚀 Examples

Want to see the math in action? Check out the `examples/` directory included in the repository source code:

* `examples/01_basic_conjunctions.py` - Full collision prediction pipeline using cKDTree.
* `examples/02_visualize_swarm.py` - 3D trajectory rendering of LEO satellite constellations.
* `examples/03_ground_station_visibility.py` - Predict when satellites will pass over your coordinates.

---

## 📝 How to Cite ASTRA

If you use ASTRA-Core in an academic paper, research project, or commercial product, please use the following BibTeX entry to provide attribution:

```bibtex
@software{Tare_ASTRA_2026,
  author = {Tare, Ishan},
  title = {ASTRA: Autonomous Space Traffic Risk Analyzer},
  year = {2026},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/ISHANTARE/ASTRA}},
  version = {3.1.2}
}
```

---

## 👤 Author

**ISHAN TARE**  
*Computer Science Student*

© 2026 ASTRA Project
