# Frontend UI/UX Specifications (React + Three.js)

## 1. Core Interface Layout
- **Full Screen Background:** Black/Dark Canvas containing the 3D WebGL Earth and Orbit visualization.
- **Left Control Panel:** Interactive filtering system (Region selection, Altitude slider, Object type checkboxes).
- **Bottom Timeline Slider:** Controls the simulation time (T=0 to T+24h).
- **Right Analytics Panel:** Displays dynamic charts (orbital congestion bar chart) and the Close Approach Prediction output.

## 2. User Workflow Model
1. **Open Application:** The system loads the default global dataset visualization (points around the 3D Earth) without running any heavy pairwise computations.
2. **Select Orbital Region:** User clicks "LEO (Low Earth Orbit)". The 3D scene smoothly zooms in and filters out irrelevant objects.
3. **Run Prediction Analysis:** User clicks "Calculate 24h Risk". The frontend sends a POST request to `/api/predict/approaches` with the selected filters.
4. **Generate Risk Report:** The backend returns close approach events. The frontend draws highlight markers on the 3D globe where high-risk events will occur and displays a summary table in the Right Analytics Panel. The conjunction event table displaying the report should show:
   - Object Names
   - Closest Distance
   - Time of Closest Approach (TCA)
   - Relative Velocity
   - Risk Classification
   *(Optional UX Enhancement: The UI could also highlight the TCA moment directly on the bottom simulation timeline)*

## 3. WebGL Rendering Requirements
- Use **InstancedMesh** in Three.js to render 30,000+ objects smoothly.
- Color coding: (e.g., Active Sats = Green, Debris = Red, Rocket Bodies = Gray).
- Orbital Paths: Trail rendering for selected objects only (to preserve FPS).
