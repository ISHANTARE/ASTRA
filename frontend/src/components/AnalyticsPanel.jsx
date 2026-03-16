/**
 * Right-side analytics panel.
 * Displays prediction results (conjunction event table) and congestion chart.
 * Per doc 05: Right Analytics Panel with dynamic charts and Close Approach Prediction output.
 */
export default function AnalyticsPanel({
  predictionResult,
  congestionData,
  loading,
}) {
  const events = predictionResult?.events || [];
  const stats = predictionResult
    ? {
        analyzed: predictionResult.objects_analyzed,
        pairs: predictionResult.candidate_pairs,
        events: events.length,
      }
    : null;

  const riskClass = (classification) => {
    const map = {
      Extreme: "extreme",
      "High Risk": "high-risk",
      Conjunction: "conjunction",
      Safe: "safe",
    };
    return map[classification] || "safe";
  };

  const formatTime = (utc) => {
    if (!utc) return "—";
    const d = new Date(utc);
    return d.toLocaleTimeString("en-US", {
      hour: "2-digit",
      minute: "2-digit",
      hour12: false,
    });
  };

  // Find max count for congestion bar scaling
  const maxCongestion = congestionData?.bands
    ? Math.max(...congestionData.bands.map((b) => b.total_objects), 1)
    : 1;

  return (
    <div className="panel analytics-panel" id="analytics-panel">
      <div className="panel-title">
        <span className="icon">📊</span>
        Risk Analysis
      </div>

      {/* Stats Summary */}
      {stats && (
        <div className="stats-row">
          <div className="stat-card">
            <div className="stat-value">{stats.analyzed.toLocaleString()}</div>
            <div className="stat-label">Objects</div>
          </div>
          <div className="stat-card">
            <div className="stat-value">{stats.pairs.toLocaleString()}</div>
            <div className="stat-label">Pairs</div>
          </div>
          <div className="stat-card">
            <div className="stat-value">{stats.events}</div>
            <div className="stat-label">Events</div>
          </div>
        </div>
      )}

      {/* Computation time */}
      {predictionResult?.computation_time_seconds && (
        <div style={{ fontSize: "10px", color: "var(--text-muted)", marginBottom: "12px", fontFamily: "'JetBrains Mono', monospace" }}>
          ⏱ Computed in {predictionResult.computation_time_seconds}s
        </div>
      )}

      {/* Event Table */}
      {events.length > 0 ? (
        <div className="event-table-container">
          <div className="filter-label">Conjunction Events</div>
          <table className="event-table">
            <thead>
              <tr>
                <th>Objects</th>
                <th>Dist</th>
                <th>TCA</th>
                <th>V_rel</th>
                <th>Risk</th>
              </tr>
            </thead>
            <tbody>
              {events.slice(0, 20).map((evt, i) => (
                <tr key={i}>
                  <td style={{ maxWidth: "90px", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                    <div style={{ fontSize: "10px" }}>{evt.object_1}</div>
                    <div style={{ fontSize: "10px", color: "var(--text-muted)" }}>{evt.object_2}</div>
                  </td>
                  <td>{evt.closest_distance_km < 1 ? `${(evt.closest_distance_km * 1000).toFixed(0)}m` : `${evt.closest_distance_km.toFixed(2)}km`}</td>
                  <td>{formatTime(evt.time_of_closest_approach_utc)}</td>
                  <td>{evt.relative_velocity_km_s.toFixed(1)}</td>
                  <td>
                    <span className={`risk-badge ${riskClass(evt.risk_classification)}`}>
                      {evt.risk_classification}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          {events.length > 20 && (
            <div style={{ textAlign: "center", fontSize: "10px", color: "var(--text-muted)", marginTop: "8px" }}>
              +{events.length - 20} more events
            </div>
          )}
        </div>
      ) : !loading && predictionResult ? (
        <div className="empty-state">
          <div className="empty-icon">✅</div>
          <div>No conjunction events detected</div>
          <div style={{ marginTop: "4px" }}>All clear for the next 24 hours</div>
        </div>
      ) : !loading ? (
        <div className="empty-state">
          <div className="empty-icon">🔭</div>
          <div>Run a prediction analysis to see conjunction events</div>
        </div>
      ) : null}

      {/* Loading state */}
      {loading && (
        <div className="empty-state">
          <div className="loading-spinner" style={{ margin: "0 auto 12px" }}></div>
          <div>Propagating orbits & analyzing pairs...</div>
        </div>
      )}

      {/* Congestion Chart */}
      {congestionData?.bands && congestionData.bands.length > 0 && (
        <div className="congestion-section">
          <div className="filter-label">Orbital Congestion</div>
          <div className="congestion-bar-container">
            {congestionData.bands.slice(0, 12).map((band, i) => (
              <div key={i} className="congestion-bar-row">
                <span className="congestion-bar-label">
                  {band.altitude_min_km}-{band.altitude_max_km}
                </span>
                <div className="congestion-bar-track">
                  <div
                    className="congestion-bar-fill"
                    style={{
                      width: `${(band.total_objects / maxCongestion) * 100}%`,
                    }}
                  />
                </div>
                <span className="congestion-bar-count">
                  {band.total_objects.toLocaleString()}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
