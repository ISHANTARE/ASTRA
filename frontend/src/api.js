const API_BASE = "http://localhost:8000/api";

/**
 * Fetch orbital objects with optional filters.
 */
export async function fetchObjects({ region, type, limit = 50000 } = {}) {
  const params = new URLSearchParams();
  if (region && region !== "ALL") params.append("region", region);
  if (type) params.append("type", type);
  params.append("limit", limit.toString());

  const res = await fetch(`${API_BASE}/objects?${params}`);
  if (!res.ok) throw new Error(`Failed to fetch objects: ${res.statusText}`);
  return res.json();
}

/**
 * Fetch computed [x,y,z] positions at a specific time step.
 */
export async function fetchPositions({ timeStep = 0, region, type } = {}) {
  const params = new URLSearchParams();
  params.append("time_step", timeStep.toString());
  if (region && region !== "ALL") params.append("region", region);
  if (type) params.append("type", type);

  const res = await fetch(`${API_BASE}/positions?${params}`);
  if (!res.ok) throw new Error(`Failed to fetch positions: ${res.statusText}`);
  return res.json();
}

/**
 * Fetch single object details.
 */
export async function fetchObjectDetail(noradId) {
  const res = await fetch(`${API_BASE}/object/${noradId}`);
  if (!res.ok) throw new Error(`Failed to fetch object ${noradId}`);
  return res.json();
}

/**
 * Run close approach prediction analysis.
 */
export async function predictApproaches({
  region = "LEO",
  altitude_range = [400, 600],
  prediction_window_hours = 24,
  time_resolution_mins = 5,
} = {}) {
  const res = await fetch(`${API_BASE}/predict/approaches`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      region,
      altitude_range,
      prediction_window_hours,
      time_resolution_mins,
    }),
  });
  if (!res.ok) throw new Error(`Prediction failed: ${res.statusText}`);
  return res.json();
}

/**
 * Fetch congestion analytics.
 */
export async function fetchCongestion({ region, max_altitude_km = 2000 } = {}) {
  const params = new URLSearchParams();
  if (region) params.append("region", region);
  params.append("max_altitude_km", max_altitude_km.toString());

  const res = await fetch(`${API_BASE}/analytics/congestion?${params}`);
  if (!res.ok) throw new Error(`Failed to fetch congestion: ${res.statusText}`);
  return res.json();
}

/**
 * Trigger dataset refresh from CelesTrak.
 */
export async function refreshDataset() {
  const res = await fetch(`${API_BASE}/dataset/refresh`);
  if (!res.ok) throw new Error(`Dataset refresh failed: ${res.statusText}`);
  return res.json();
}
