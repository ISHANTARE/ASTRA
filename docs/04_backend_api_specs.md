# Backend API Specifications (FastAPI)

## Core Endpoints

### 1. `GET /api/dataset/refresh`

- **Description:** Pulls the latest TLE dataset from CelesTrak and stores it in the PostgreSQL database.
- **Trigger:** Scheduled daily or manual trigger.
- **Output:** `{ "status": "success", "objectsUpdated": 30542, "timestamp": "..." }`

### 2. `GET /api/objects`

- **Description:** Returns a list of orbital objects based on filter parameters (e.g., LEO, active satellites). Used by the frontend to render the initial 3D scene.
- **Parameters:** `region` (LEO|MEO|GEO|HEO|ALL), `type` (debris|satellite|rocket_body).
- **Output:** JSON array of objects with `id`, `name`, `type`, `tle_line1`, `tle_line2`.

### 3. `GET /api/object/{id}`

- **Description:** Returns detailed inspection data for a specific orbital object.
- **Parameters:** Path parameter `id` (e.g., NORAD catalog ID).
- **Output:** JSON object containing full TLE details, object type, and metadata.

### 4. `POST /api/predict/approaches`

- **Description:** Runs the orbital prediction engine and close approach detection logic. Internally performs Orbital Spatial Grid Indexing (coarse 3D bucketing) to heavily reduce candidate pairs before running SGP4.
- **Input Payload:**
  ```json
  {
    "region": "LEO",
    "altitude_range": [400, 600],
    "prediction_window_hours": 24,
    "time_resolution_mins": 5
  }
  ```
- **Process:** Applies multi-stage algorithm (geometric filters + spatial grid indexing) before propagating SGP4.
- **Output:**
  ```json
  {
    "objects_analyzed": 8421,
    "events": [
      {
        "object_1": "STARLINK-1234",
        "object_2": "UNKNOWN DEBRIS",
        "closest_distance_km": 0.45,
        "time_of_closest_approach_utc": "2024-05-10T14:22:00Z",
        "relative_velocity_km_s": 11.2,
        "risk_classification": "High Risk"
      }
    ]
  }
  ```

### 5. `GET /api/analytics/congestion`

- **Description:** Returns spatial density calculations grouped by altitude bands.
