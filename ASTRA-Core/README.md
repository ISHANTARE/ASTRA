# ASTRA-Core v1.0.1 🚀

**The High-Performance Mathematical Foundation for Space Situational Awareness.**

ASTRA-Core is the computational engine powering the ASTRA ecosystem. It is an elite Python library designed for aerospace engineers, researchers, and developers who need to track thousands of orbital objects simultaneously with sub-second precision.

---

## 📋 The Science Behind ASTRA

ASTRA-Core manages the extreme complexity of Low Earth Orbit (LEO) through three primary mathematical modules:

### 1. Vectorized SGP4 Propagation

Standard orbital models for Earth involve non-spherical gravity (J2/J3/J4 effects), lunar-solar perturbations, and atmospheric drag. ASTRA-Core utilizes the **Simplified General Perturbations #4 (SGP4)** model, implemented with **vectorized NumPy arrays**. This allows you to propagate 15,000+ objects simultaneously at 60Hz without bottlenecking the CPU.

### 2. Conjunction & Collision Probability

ASTRA-Core doesn't just measure "miss distance"—it solves the **Probability of Collision ($P_c$)**.

* **Hermite Spline Interpolation**: Instead of checking positions every minute (which can miss a collision), ASTRA-Core fits cubic splines to trajectories to find the mathematical root of the closest approach point.
* **Mahalanobis B-Plane Mapping**: We project 3D error ellipsoids (covariance) onto a 2D encounter plane to calculate the true statistical likelihood of an impact.

### 3. Spatial Filtering (Sweep-and-Prune)

To find a collision between 20,000 satellites, you'd usually. check 200 million pairs ($O(n^2)$). ASTRA-Core uses a **1D Sweep-and-Prune algorithm** combined with **Axis-Aligned Bounding Boxes (AABB)** to prune 99.9% of non-threatening pairs in milliseconds.

---

## 📦 Installation

ASTRA-Core is available as a standalone package on PyPI:

```bash
pip install astra-core-engine
```

Alternatively, for development:

```bash
git clone https://github.com/ISHANTARE/ASTRA.git
cd ASTRA/ASTRA-Core
pip install .
```

---

## 💻 Technical Quickstart

```python
import astra
import numpy as np

# 1. Fetch live TLEs
print("Downloading live catalog...")
active_catalog = astra.fetch_celestrak_active()

# 2. Setup Debris Monitoring
objects = [astra.make_debris_object(tle) for tle in active_catalog]
leo_only = astra.filter_altitude(objects, min_km=200, max_km=2000)

# 3. Mass Propagation
tles = [obj.tle for obj in leo_only]
time_steps = np.arange(0, 120, 5.0) # Next 2 hours
trajectories = astra.propagate_many(tles, time_steps)

# 4. Filter for Conjunction Events (< 5km)
events = astra.find_conjunctions(
    trajectories, 
    leo_only[0].tle.epoch_jd + time_steps/1440.0, 
    {obj.tle.norad_id: obj for obj in leo_only}, 
    threshold_km=5.0
)

print(f"Detected {len(events)} potential threats.")
```

---

## 📚 Library Reference

* **`astra.orbit`**: The SGP4 engine and trajectory mapping.
* **`astra.conjunction`**: Spline-based TCA finding and AABB spatial filters.
* **`astra.covariance`**: Mahalanobis projection and B-Plane probability logic.
* **`astra.visibility`**: Topocentric observer geometry (Pass Predictions).
* **`astra.data`**: Automated CelesTrak and OMM data ingestion.

---

## 👤 Author

**ISHAN TARE**  
*Astrodynamics & Computer Science Student*

---

© 2026 ASTRA Project
