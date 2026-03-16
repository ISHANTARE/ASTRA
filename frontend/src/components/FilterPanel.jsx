import { useState } from "react";

const REGIONS = ["ALL", "LEO", "MEO", "GEO", "HEO"];

/**
 * Left-side filter panel.
 * Controls region selection, altitude range, object type toggles, and prediction trigger.
 * Per doc 05: Left Control Panel with Region selection, Altitude slider, Object type checkboxes.
 */
export default function FilterPanel({
  region,
  setRegion,
  altitudeRange,
  setAltitudeRange,
  objectTypes,
  setObjectTypes,
  onPredict,
  predicting,
  objectCounts = {},
}) {
  const handleTypeToggle = (type) => {
    setObjectTypes((prev) => ({ ...prev, [type]: !prev[type] }));
  };

  return (
    <div className="panel filter-panel" id="filter-panel">
      <div className="panel-title">
        <span className="icon">🛰️</span>
        Mission Control
      </div>

      {/* Region Selection */}
      <div className="filter-section">
        <div className="filter-label">Orbital Region</div>
        <div className="region-grid">
          {REGIONS.map((r) => (
            <button
              key={r}
              id={`region-btn-${r}`}
              className={`region-btn ${region === r ? "active" : ""}`}
              onClick={() => setRegion(r)}
            >
              {r}
            </button>
          ))}
        </div>
      </div>

      {/* Altitude Range */}
      <div className="filter-section">
        <div className="filter-label">Altitude Range (km)</div>
        <div className="altitude-slider-container">
          <div className="altitude-values">
            <span>{altitudeRange[0]} km</span>
            <span>{altitudeRange[1]} km</span>
          </div>
          <input
            type="range"
            id="altitude-min-slider"
            min={100}
            max={2000}
            step={50}
            value={altitudeRange[0]}
            onChange={(e) =>
              setAltitudeRange([
                Math.min(Number(e.target.value), altitudeRange[1] - 50),
                altitudeRange[1],
              ])
            }
          />
          <input
            type="range"
            id="altitude-max-slider"
            min={100}
            max={2000}
            step={50}
            value={altitudeRange[1]}
            onChange={(e) =>
              setAltitudeRange([
                altitudeRange[0],
                Math.max(Number(e.target.value), altitudeRange[0] + 50),
              ])
            }
          />
        </div>
      </div>

      {/* Object Type Toggles */}
      <div className="filter-section">
        <div className="filter-label">Object Type</div>
        <div className="type-checkboxes">
          {[
            { key: "satellite", label: "Active Satellites", dotClass: "satellite" },
            { key: "debris", label: "Debris Fragments", dotClass: "debris" },
            { key: "rocket_body", label: "Rocket Bodies", dotClass: "rocket_body" },
          ].map(({ key, label, dotClass }) => (
            <label key={key} className="type-checkbox" id={`type-toggle-${key}`}>
              <input
                type="checkbox"
                checked={objectTypes[key]}
                onChange={() => handleTypeToggle(key)}
              />
              <span className="checkbox-custom"></span>
              <span className={`type-dot ${dotClass}`}></span>
              <span className="type-label">{label}</span>
              <span className="type-count">
                {objectCounts[key] !== undefined ? objectCounts[key].toLocaleString() : "—"}
              </span>
            </label>
          ))}
        </div>
      </div>

      {/* Predict Button */}
      <button
        id="predict-btn"
        className={`predict-btn ${predicting ? "loading" : ""}`}
        onClick={onPredict}
        disabled={predicting}
      >
        {predicting ? "⏳ Analyzing Orbits..." : "🔬 Calculate 24h Risk"}
      </button>
    </div>
  );
}
