# ASTRA Core v2.0
**Autonomous Space Traffic Risk Analyzer - Computation Engine**

ASTRA Core is an elite, high-performance Python library designed for **Space Situational Awareness (SSA)**. It allows developers to track thousands of satellites, debris objects, and spacecraft simultaneously to predict potential space collisions (conjunctions) and calculate when satellites will pass over ground stations (visibility).

Unlike traditional orbital mechanics libraries that rely on slow Python loops, ASTRA Core is built from the ground up for **extreme horizontal scaling** using vectorized NumPy operations, mathematical spline interpolation, and advanced spatial filtering algorithms.

---

## 🚀 Orbital Mechanics 101 (For Beginners)

If you are new to astrodynamics, here is how ASTRA Core works under the hood:

### 1. What is a TLE?
A **Two-Line Element set (TLE)** is a standard text format used by organizations like NORAD and the US Space Force to describe the orbit of a satellite around Earth. 
ASTRA Core downloads these TLEs (which act like a "seed" or snapshot of a satellite's state) and uses them to predict the future.

### 2. How do we predict positions? (SGP4)
You cannot simply use Newton's laws of gravity to track satellites because the Earth is not a perfect sphere, the moon and sun pull on the orbits, and atmospheric drag slows satellites down. 
ASTRA Core uses the **SGP4 (Simplified General Perturbations #4)** mathematical model. SGP4 takes a TLE and computes these complex physical perturbations to calculate exactly where a satellite will be in 3D Cartesian space at any given time.

### 3. What is a Conjunction?
A conjunction occurs when two objects (like a satellite and a piece of space junk) approach each other dangerously close. ASTRA Core calculates the exact **Time of Closest Approach (TCA)** and mathematically estimates the **Probability of Collision ($P_c$)** based on their velocities and positional uncertainty (covariance).

---

## ✨ Key Features (v2.0)

* **Blazing Fast Vectorized Propagation:** Propagates 10,000+ objects simultaneously utilizing `sgp4_array`.
* **Live Celestial Data:** Native API integration with CelesTrak to download live, up-to-date satellite catalogs.
* **Exact TCA Root-Finding:** Utilizes `scipy` Cubic Hermite Splines to perfectly map curvilinear orbits, isolating exact sub-second intersection points without temporal discretization flaws.
* **True Encounter-Plane Probability:** Abandons standard isotropic approximations in favor of true 3x3 covariance matrix projection onto the Relative Velocity B-Plane (Mahalanobis distance evaluation).
* **Sweep-and-Prune AABB Filtering:** Filters millions of potential satellite pairs in milliseconds using radial bounding shells and Cartesian Axis-Aligned Bounding Boxes (AABB).
* **Interactive 3D Visualizations:** Out-of-the-box `plotly` rendering to visually demonstrate conjunctions over a 3D Earth wireframe.

---

## 📦 Installation

Ensure you have Python 3.10+ installed.

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/ASTRA.git
   cd ASTRA/ASTRA-Core
   ```
2. Install the required dependencies:
   ```bash
   pip install numpy scipy skyfield sgp4 plotly requests
   ```

---

## 💻 Quickstart Guide

Here is a complete, end-to-end example of how to use ASTRA Core to find collisions among the active satellite catalog.

```python
import numpy as np
import astra

# 1. Fetch live active satellite catalog from CelesTrak
print("Downloading live satellite catalog...")
active_catalog = astra.fetch_celestrak_active()

# 2. Build DebrisObjects and apply Filters (e.g., LEO satellites only)
objects = [astra.make_debris_object(tle) for tle in active_catalog]
leo_objects = astra.filter_altitude(objects, min_km=200, max_km=2000)

print(f"Tracking {len(leo_objects)} LEO objects...")

# 3. Define the Time Window (e.g., Next 2 hours every 5 minutes)
start_jd = leo_objects[0].tle.epoch_jd
time_steps = np.arange(0, 120, 5.0)  # 0 to 120 minutes
times_jd = start_jd + time_steps / 1440.0

# 4. Propagate all orbits simultaneously!
tles = [obj.tle for obj in leo_objects]
trajectories = astra.propagate_many(tles, time_steps)

# 5. Detect Conjunctions (< 5 km miss distance)
elements_map = {obj.tle.norad_id: obj for obj in leo_objects}
events = astra.find_conjunctions(
    trajectories, times_jd, elements_map, threshold_km=5.0
)

for event in events[:5]:
    print(f"Warning! Object {event.object_a_id} approaching Object {event.object_b_id}")
    print(f"Miss Distance: {event.miss_distance_km:.2f} km")
    print(f"Collision Probability: {event.collision_probability:.2e} ({event.risk_level})")

# 6. Render the 3D Interactive Plot
fig = astra.plot_trajectories(trajectories, events, title="LEO Conjunctions")
fig.show()
```

---

## 📚 Module Reference

* `astra.data`: Fetch live TLE and OMM data from external Space-Track/CelesTrak APIs.
* `astra.tle`: TLE string parsing, checksum validation, and catalog aggregation.
* `astra.orbit`: The core SGP4 propagation engine generating `TrajectoryMap` arrays.
* `astra.debris`: Tools for filtering massive catalogs by timeline, altitude, and geographic region.
* `astra.conjunction`: Advanced BVH Sweep-and-Prune filtering, Cartesian AABBs, and Cubic Spline interpolation for precise collision warning.
* `astra.covariance`: Astrodynamics math calculating Mahalanobis distances and Gaussian B-Plane collision probabilities.
* `astra.visibility`: Vectorized Topocentric geometry to detect pass events (AOS, TCA, LOS) for ground stations.
* `astra.plot`: 3D rendering engine.
