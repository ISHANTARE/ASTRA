# Development Phases & Tasks

## Phase 1 — Core System Architecture (MVP)
The immediate focus for the developer AI.

### Milestone 1: Data & Backend
- [ ] Connect to CelesTrak and parse TLE data.
- [ ] Setup Supabase PostgreSQL schema for orbital objects.
- [ ] Build FastAPI server with GET endpoints.
- [ ] Implement SGP4 `skyfield` or `sgp4` logic for computing coordinates at given T.

### Milestone 2: Frontend Visualization
- [ ] Initialize React + Three.js project.
- [ ] Render 3D Earth with simple textures.
- [ ] Fetch TLE data and plot 30k dots using `InstancedMesh`.
- [ ] Build left-side filtering UI to subset the active points.

### Milestone 3: Analytical Engine
- [ ] Implement multi-stage progressive filtering algorithm (Stages 1-4).
- [ ] Implement orbital spatial grid indexing to reduce candidate pair generation (Stage 5-6).
- [ ] Build `/predict/approaches` API endpoint (Stage 7).
- [ ] Implement Time of Closest Approach (TCA) detection during conjunction analysis (after distance calculations).
- [ ] Compute relative velocity at TCA for each conjunction event.
- [ ] Connect frontend "Run Prediction" button to the backend API.
- [ ] Render prediction report UI (tables, high-risk markers).

## Phase 2 — Advanced Research Features (Future Scope)
- [ ] Debris cascade modeling (Kessler Syndrome simulation).
- [ ] Collision probability calculation (Probability of Collision - Pc).
- [ ] Historical congestion analysis using time-series data.
