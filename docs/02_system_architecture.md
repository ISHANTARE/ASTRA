# System Architecture

## 1. High-Level Architecture
The ASTRA platform consists of three main subsystems interacting to deliver real-time visualization and risk analysis.

### 1️⃣ Visualization System (Frontend)
- **Purpose:** Display satellites and debris orbiting Earth.
- **Requirements:** 3D Earth rendering, satellite markers, orbit paths, real-time animation, interactive object information panels, and filters for object types.
- **Tech Stack:** React, Three.js, WebGL.
- **Pipeline:** TLE data → SGP4 orbit propagation (backend) → TEME coordinates → convert TEME to ECI/ECEF → (x,y,z) coordinates → 3D rendering (frontend).

### 2️⃣ Orbital Prediction Engine (Backend)
- **Purpose:** Predict satellite positions and detect close approaches.
- **Core Model:** SGP4 Orbit Propagation Model.
- **Inputs:** TLE Data, Time.
- **Outputs:** (x,y,z) position vector, velocity vector.
- **Parameters:** 24-hour window, 5-minute resolution (288 steps).

### 3️⃣ Orbital Analysis Engine (Backend)
- **Purpose:** Analyze orbital traffic and detect collision risks.
- **Computes:** Object density per orbit layer, close approach detection (distance), relative velocity of objects, and risk classification.
- **Orbital Spatial Grid Indexing Layer:** Partitions orbital space into coarse 3D buckets (altitude bands and angular sectors) to reduce the number of O(N²) comparisons during close approach detection.

## 2. Tech Stack Overview
- **Frontend Stack:** React, Three.js, WebGL
- **Backend Stack:** Python, FastAPI
- **Orbital Computation Libraries:** `sgp4`, `skyfield`, `numpy`
- **Database:** PostgreSQL
- **Cloud Deployment (Expected target):** Render (Backend), Vercel (Frontend), Supabase (Database)
