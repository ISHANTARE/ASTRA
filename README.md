# ASTRA: Autonomous Space Traffic Risk Analyzer 🛰️

![PyPI - Version](https://img.shields.io/pypi/v/astra-core-engine?color=blue&label=astra-core-engine)
![License](https://img.shields.io/github/license/ISHANTARE/ASTRA)
![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)

ASTRA is an advanced, full-stack ecosystem designed to solve the critical challenges of **Space Situational Awareness (SSA)**. From high-performance orbital dynamics calculations to immersive 3D visualizations, ASTRA provides the tools necessary to monitor the growing congestion in Low Earth Orbit (LEO).

---

## 🏗️ Project Architecture

ASTRA is divided into two primary pillars:

### 1. [ASTRA-Core (The Engine)](./ASTRA-Core)

A high-performance Python library published on PyPI. It handles the "heavy lifting" of astrodynamics:

* **SGP4 Propagation**: Real-time position and velocity calculation.
* **Conjunction Analysis**: Predicting collisions using Sweep-and-Prune spatial filters.
* **Probability Logic**: True Mahalanobis $P_c$ calculations based on 3x3 covariance matrices.

### 2. ASTRA-Platform (The Interface)

*(In Development)* A bespoke, WebGL-powered 3D dashboard inspired by **LeoLabs**.

* **Global Radar View**: Interactive 3D Earth with glowing additive particle swarms.
* **Timeline Scrubbing**: Physics-accurate time travel to watch conjunctions unfold.
* **Ground Station Visibility**: Predicting when satellites pass over your specific coordinates.

---

## 🚀 QuickStart: ASTRA-Core

You can use the ASTRA physics engine natively in your own Python projects:

```bash
pip install astra-core-engine
```

```python
import astra
catalog = astra.fetch_celestrak_active()
print(f"Tracking {len(catalog)} active satellites.")
```

For more detailed documentation, check out the [ASTRA-Core Documentation](./ASTRA-Core/README.md).

---

## 🗺️ Roadmap

* **v1.0.0**: Core Physics Engine release on PyPI.
* **v2.0.0**: Launch of the 3D Visualizer.
* **v3.0.0**: Automated maneuvering alerts and debris cloud modeling.

---

## 👤 Author

**ISHAN TARE**  
*Astrodynamics Student*

---

© 2026 ASTRA Project
