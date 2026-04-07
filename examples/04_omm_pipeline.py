"""
Example 04: OMM Data Pipeline
==============================
Demonstrates how to ingest CCSDS Orbit Mean-Elements Message (OMM) data
from CelesTrak into the ASTRA-Core physics pipeline.

OMM is the modern, recommended orbital data standard. Unlike TLEs, OMM
records carry rich physical metadata:
  - RCS size category (SMALL / MEDIUM / LARGE) → approximate radar cross-section m²
  - Mass (kg) when available from the catalog
  - Drag coefficient x area / mass (for high-fidelity Cowell propagation)

This script:
  1. Fetches the GPS constellation in OMM JSON format from CelesTrak
  2. Inspects the expanded metadata unavailable in TLEs
  3. Filters satellites by altitude using the ASTRA debris pipeline
  4. Propagates a subset of OMM objects using the standard SGP4 engine
  5. Shows how to use OMM data in conjunction analysis
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import astra


def main():
    print("=" * 60)
    print("  ASTRA-Core Example 04: OMM Data Pipeline")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Fetch the GPS constellation in modern OMM format
    # ------------------------------------------------------------------
    print("\n[1] Fetching GPS constellation from CelesTrak in OMM format...")
    omms = astra.fetch_celestrak_group_omm("gps-ops")
    print(f"    Loaded {len(omms)} SatelliteOMM objects.")

    # ------------------------------------------------------------------
    # 2. Inspect OMM-exclusive physical metadata
    # ------------------------------------------------------------------
    print("\n[2] Inspecting OMM-exclusive physical metadata:")
    print(f"    {'Name':<26} {'NORAD':>7}  {'Type':<15}  {'RCS (m²)':>10}")
    print("    " + "-" * 64)
    for omm in omms[:8]:
        rcs = f"{omm.rcs_m2:.1f}" if omm.rcs_m2 is not None else "N/A"
        print(f"    {omm.name:<26} {omm.norad_id:>7}  {omm.object_type:<15}  {rcs:>10}")

    # ------------------------------------------------------------------
    # 3. Show unit-converted orbital elements (radians, rad/min)
    # ------------------------------------------------------------------
    first = omms[0]
    print(f"\n[3] Unit-converted orbital elements for {first.name}:")
    print(f"    Epoch JD         : {first.epoch_jd:.6f}")
    print(f"    Inclination      : {math.degrees(first.inclination_rad):.4f}°"
          f"  ({first.inclination_rad:.6f} rad)")
    print(f"    RAAN             : {math.degrees(first.raan_rad):.4f}°"
          f"  ({first.raan_rad:.6f} rad)")
    print(f"    Eccentricity     : {first.eccentricity:.7f}")
    print(f"    Mean Motion      : {first.mean_motion_rad_min:.8f} rad/min")
    print(f"    BSTAR            : {first.bstar:.6e}")

    # ------------------------------------------------------------------
    # 4. Filter by altitude using the debris catalog pipeline
    # ------------------------------------------------------------------
    print("\n[4] Filtering OMM objects to MEO (2000-35786 km)...")
    debris_objects = [astra.make_debris_object(omm) for omm in omms]
    meo_objects = astra.filter_altitude(debris_objects, min_km=2000, max_km=35786)
    print(f"    {len(meo_objects)} / {len(omms)} objects are in MEO.")

    # ------------------------------------------------------------------
    # 5. Propagate a small subset of OMM objects (SGP4 via same engine)
    # ------------------------------------------------------------------
    print("\n[5] Propagating first 5 OMM satellites over 1 hour (5-min steps)...")
    subset_omms = omms[:5]
    start_jd = subset_omms[0].epoch_jd
    times_jd = start_jd + np.arange(0, 60, 5) / 1440.0

    # propagate_many accepts SatelliteOMM exactly like SatelliteTLE
    trajectories = astra.propagate_many(subset_omms, times_jd)

    for omm in subset_omms:
        positions = trajectories[omm.norad_id]
        r_km = np.linalg.norm(positions, axis=1)
        alt_km = r_km - 6371.0
        print(f"    {omm.name:<26}  altitude range: "
              f"{alt_km.min():.1f} – {alt_km.max():.1f} km")

    # ------------------------------------------------------------------
    # 6. Summarise catalog metadata statistics
    # ------------------------------------------------------------------
    print("\n[6] Catalog statistics:")
    stats = astra.catalog_statistics(debris_objects)
    for key, val in stats.items():
        print(f"    {key:<30}: {val}")

    print("\n✓ OMM pipeline complete.\n")


if __name__ == "__main__":
    main()
