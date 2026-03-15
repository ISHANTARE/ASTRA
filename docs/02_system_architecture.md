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

## 3. Backend Module Structure
ASTRA uses a modular backend architecture to maintain clear separation between the API layer, orbital physics computation, filtering pipeline, conjunction analysis engine, data ingestion, and database access. This structure ensures the system remains maintainable and scalable as the research platform grows.

### High-Level Backend Structure
Conceptually, the backend is organized as follows:

```text
app/
├── api/
├── orbit/
├── filtering/
├── analysis/
├── data/
├── models/
├── database/
├── core/
└── utils/
```

Each module corresponds to a major subsystem defined in the architecture.

### Module Responsibilities
- `api/`: FastAPI route handlers. Responsible only for handling HTTP requests and orchestrating backend services.
- `orbit/`: Contains orbital physics logic including SGP4 propagation, coordinate frame transformations (TEME → ECI/ECEF), and trajectory precomputation.
- `filtering/`: Implements the multi-stage filtering pipeline to reduce computational load (region, altitude, orbital plane filtering, orbit intersection checks, and spatial grid indexing).
- `analysis/`: Contains analytical algorithms including candidate pair generation, distance calculations, Time of Closest Approach (TCA) detection, relative velocity computation, and conjunction event classification.
- `data/`: Handles ingestion and parsing of orbital datasets (e.g., pulling TLEs from CelesTrak, parsing, and dataset loading).
- `models/`: Defines internal data structures for orbital objects and conjunction events.
- `database/`: Handles database connections and schema definitions used to store orbital object catalogs.
- `core/`: Configuration settings and global constants used across the backend.
- `utils/`: General helper functions used across modules.

### Architectural Rules
To maintain clean architecture, developers and AI coding assistants must adhere to these rules:

- Orbital propagation logic must exist **only** inside the `orbit/` module.
- Filtering logic must exist **only** inside the `filtering/` module.
- Conjunction detection algorithms must exist **only** inside the `analysis/` module.
- API routes must **not contain heavy computational logic** and should only call service modules.

### Purpose of This Structure
This backend structure directly reflects the architectural layers of the ASTRA system design. It ensures:

- solid separation of concerns
- easier debugging
- a maintainable research codebase
- easier integration and reliability with AI coding assistants.
