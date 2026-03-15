# Data Model & Processing Pipeline

## 1. Data Source
- **Primary Source:** [CelesTrak](https://celestrak.org/)
- **Data Format:** TLE (Two-Line Element sets)
- **Expected Volume:** ~30,000+ total objects.

## 2. Orbital Definitions
| Orbit | Altitude Range   | Description |
|-------|-----------------|-------------|
| LEO   | 160 – 2000 km   | Low Earth Orbit |
| MEO   | 2000 – 35000 km | Medium Earth Orbit |
| GEO   | ~35786 km       | Geosynchronous/Geostationary Orbit |
| HEO   | Elliptical      | Highly Elliptical Orbit |
| ALL   | Any             | Global Dataset |

## 3. Multi-Stage Filtering Algorithm
To avoid O(N^2) complexity across 30k objects, use progressive filtering before deep SGP4 propagation.

- **Pipeline Flow:**
  `GLOBAL DATASET` ↓ `USER REGION FILTER` ↓ `ALTITUDE FILTER` ↓ `ORBITAL PLANE FILTER` ↓ `ORBIT INTERSECTION CHECK` ↓ `ORBITAL SPATIAL GRID INDEXING` ↓ `CANDIDATE PAIR GENERATION` ↓ `SGP4 PROPAGATION` ↓ `DISTANCE CALCULATION`

- **Stage 1 (Region Filter):** User selects target zone (e.g., LEO drops dataset from 30k to 10k).
- **Stage 2 (Altitude Filter):** Objects must have overlapping or comparable altitude ranges (e.g., 500km sat compared only to 450-600km objects).
- **Stage 3 (Orbital Plane Filter):** Check inclination similarity. Only compare objects with similar inclinations (e.g., 53° to 55°).
- **Stage 4 (Intersection Check):** Geometric check to see if orbit paths *can* physically intersect. Skip pairs that never cross.
- **Stage 5 (Orbital Spatial Grid Indexing):** Partition orbital space into coarse 3D buckets using altitude bands (e.g., 200-300km) and angular sectors (e.g., 0-10° longitude). Objects are mapped to a simple dict index `(alt_band, sector)`.
- **Stage 6 (Candidate Pair Generation):** Only generate comparisons among objects within the same spatial cell or adjacent cells to drastically reduce N² comparisons.
- **Stage 7 (Detailed Propagation):** Only remaining candidate pairs are propagated using SGP4 (with TEME to ECI/ECEF coordinate frame conversion).

## 4. Close Approach Detection & Risk Classification
- **Distance Formula:** `Distance = √((x1-x2)² + (y1-y2)² + (z1-z2)²)`
- **Time of Closest Approach (TCA) & Relative Velocity:**
  - For each candidate pair, calculate distances across all 288 simulation time steps.
  - Identify the index of the minimum distance in the array.
  - Extract the exact `closest_distance` and corresponding `TCA timestamp` from that index.
  - Retrieve the velocity vectors (`v1`, `v2`) for both objects at that specific time step.
  - Compute the **relative velocity**: `Relative Velocity = |v1 - v2|` (magnitude of the velocity difference vector).
  - Conceptual Logic:
    ```python
    tca_index = argmin(distances)
    closest_distance = distances[tca_index]
    tca_time = simulation_times[tca_index]
    relative_velocity = numpy.linalg.norm(v1[tca_index] - v2[tca_index])
    ```
- **Risk Tiers:**
  - `< 10 km`: Conjunction
  - `< 1 km`: High Risk
  - `< 100 m`: Extreme

## 5. Orbital Congestion Analysis
- Compute traffic density by altitude bands (e.g., 200-300km, 400-500km).
- Specifically track major constellation contributions (e.g., SpaceX Starlink, OneWeb, Amazon).
