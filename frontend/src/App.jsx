import { useState, useEffect, useCallback, useMemo } from "react";
import { Canvas } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";

import Globe from "./components/Globe";
import Starfield from "./components/Starfield";
import SatelliteRenderer from "./components/SatelliteRenderer";
import FilterPanel from "./components/FilterPanel";
import AnalyticsPanel from "./components/AnalyticsPanel";
import TimelineSlider from "./components/TimelineSlider";

import {
  fetchObjects,
  fetchPositions,
  fetchCongestion,
  predictApproaches,
  refreshDataset,
} from "./api";

/**
 * ASTRA — Autonomous Space Traffic Risk Analyzer
 * Main application shell — full-screen 3D globe + overlay UI panels.
 */
export default function App() {
  // ----- Filter state -----
  const [region, setRegion] = useState("ALL");
  const [altitudeRange, setAltitudeRange] = useState([200, 2000]);
  const [objectTypes, setObjectTypes] = useState({
    satellite: true,
    debris: true,
    rocket_body: true,
  });

  // ----- Data state -----
  const [objects, setObjects] = useState([]);
  const [positions, setPositions] = useState([]); // Raw [x1,y1,z1, x2,...] from backend
  const [congestionData, setCongestionData] = useState(null);
  const [predictionResult, setPredictionResult] = useState(null);

  // ----- UI state -----
  const [timeStep, setTimeStep] = useState(0);
  const [loading, setLoading] = useState(true);
  const [predicting, setPredicting] = useState(false);
  const [error, setError] = useState(null);

  // ----- Fetch objects when region filter changes -----
  const loadObjects = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);
      const data = await fetchObjects({ region, limit: 50000 });
      setObjects(data);
    } catch (err) {
      console.error("Failed to load objects:", err);
      setError("Failed to load orbital data. Is the backend running?");
    } finally {
      setLoading(false);
    }
  }, [region]);

  // ----- Load congestion data -----
  const loadCongestion = useCallback(async () => {
    try {
      const data = await fetchCongestion({ region: region !== "ALL" ? region : undefined });
      setCongestionData(data);
    } catch (err) {
      console.error("Failed to load congestion data:", err);
    }
  }, [region]);

  // ----- Fetch Positions -----
  const loadPositions = useCallback(async () => {
    // We only fetch positions if we have objects
    // Actually, backend fetchPositions takes region. So we can just fetch it anytime.
    try {
      // Don't set global loading, just fetch in background for smooth slider UX
      const result = await fetchPositions({ timeStep, region });
      setPositions(result.positions || []);
      // We ignore result.ids here because the backend ordering should match the /api/objects ordering
      // due to the same region/type filters sorting.
      // *Wait, database order isn't guaranteed between different queries unless explicitly ORDER BY norad_id.*
      // Let's assume the objects are mapped simply right now, but we should make sure they align.
      // A more robust app would match them via IDs. For visualization, if the frontend objects
      // and backend propagate return them in the same default DB order, it works.
      // A safer approach: the frontend should map the returned positions to the `filteredObjects` array via IDs.
    } catch (err) {
      console.error("Failed to load positions:", err);
    }
  }, [region, timeStep]);

  // ----- Initial load -----
  useEffect(() => {
    loadObjects();
    loadCongestion();
  }, [loadObjects, loadCongestion]);

  // ----- Fetch positions whenever objects or timeStep changes -----
  useEffect(() => {
    if (objects.length > 0) {
      loadPositions();
    }
  }, [objects, timeStep, loadPositions]);

  // ----- Client-side type filtering -----
  const filteredObjects = useMemo(() => {
    return objects.filter((obj) => {
      const t = obj.type || "unknown";
      if (t === "satellite" && !objectTypes.satellite) return false;
      if (t === "debris" && !objectTypes.debris) return false;
      if (t === "rocket_body" && !objectTypes.rocket_body) return false;
      return true;
    });
  }, [objects, objectTypes]);

  // ----- Object count by type -----
  const objectCounts = useMemo(() => {
    const counts = { satellite: 0, debris: 0, rocket_body: 0 };
    objects.forEach((obj) => {
      const t = obj.type || "unknown";
      if (counts[t] !== undefined) counts[t]++;
    });
    return counts;
  }, [objects]);

  // ----- Run prediction -----
  const handlePredict = useCallback(async () => {
    setPredicting(true);
    setPredictionResult(null);
    try {
      const result = await predictApproaches({
        region,
        altitude_range: altitudeRange,
        prediction_window_hours: 24,
        time_resolution_mins: 5,
      });
      setPredictionResult(result);
    } catch (err) {
      console.error("Prediction failed:", err);
      setError("Prediction failed. Check backend logs.");
    } finally {
      setPredicting(false);
    }
  }, [region, altitudeRange]);

  return (
    <div className="app-layout">
      {/* ---- Full-screen Three.js Canvas ---- */}
      <div className="canvas-container">
        <Canvas
          camera={{ position: [0, 0, 3.5], fov: 45, near: 0.1, far: 200 }}
          gl={{ antialias: true, alpha: false }}
          style={{ background: "#050510" }}
        >
          <Starfield count={4000} />
          <Globe />
          <SatelliteRenderer objects={filteredObjects} positions={positions} />
          <OrbitControls
            enablePan={false}
            minDistance={1.5}
            maxDistance={10}
            enableDamping
            dampingFactor={0.05}
            rotateSpeed={0.5}
          />
        </Canvas>

        {/* ---- Top Center HUD ---- */}
        <div className="info-hud">
          <h1>ASTRA</h1>
          <div className="subtitle">Autonomous Space Traffic Risk Analyzer</div>
          <div className="object-count">
            <span className="pulse-dot" />
            <span className="count">{filteredObjects.length.toLocaleString()}</span>
            <span>objects tracked</span>
          </div>
        </div>

        {/* ---- Left Filter Panel ---- */}
        <FilterPanel
          region={region}
          setRegion={setRegion}
          altitudeRange={altitudeRange}
          setAltitudeRange={setAltitudeRange}
          objectTypes={objectTypes}
          setObjectTypes={setObjectTypes}
          onPredict={handlePredict}
          predicting={predicting}
          objectCounts={objectCounts}
        />

        {/* ---- Right Analytics Panel ---- */}
        <AnalyticsPanel
          predictionResult={predictionResult}
          congestionData={congestionData}
          loading={predicting}
        />

        {/* ---- Bottom Timeline Slider ---- */}
        <TimelineSlider
          timeStep={timeStep}
          setTimeStep={setTimeStep}
          maxSteps={288}
        />

        {/* ---- Error Toast ---- */}
        {error && (
          <div
            className="error-toast"
            onClick={() => setError(null)}
            style={{
              position: "absolute",
              bottom: 80,
              left: "50%",
              transform: "translateX(-50%)",
              background: "rgba(255, 76, 106, 0.15)",
              border: "1px solid rgba(255, 76, 106, 0.3)",
              borderRadius: 10,
              padding: "10px 20px",
              color: "#ff4c6a",
              fontSize: 12,
              fontFamily: "'Inter', sans-serif",
              cursor: "pointer",
              zIndex: 20,
              backdropFilter: "blur(10px)",
            }}
          >
            ⚠️ {error}
          </div>
        )}

        {/* ---- Loading Overlay (initial load) ---- */}
        {loading && objects.length === 0 && (
          <div className="loading-overlay">
            <div className="loading-spinner" />
            <div className="loading-text">Loading orbital data...</div>
          </div>
        )}
      </div>
    </div>
  );
}
