"""
Example 05: TLE vs OMM Pipeline Comparison
============================================
Demonstrates that ASTRA-Core's TLE and OMM ingestion pipelines produce
computationally equivalent orbital states for the same objects.

We fetch the GPS constellation from CelesTrak in *both* formats (TLE and
OMM JSON), propagate matched pairs through the same SGP4 engine, and
compute the 3-D position difference at every time step.

Expected result:
  The positional delta should be near-zero (< 1 km) throughout the entire
  propagation window, confirming that:
  1. Both parser pipelines (astra.tle and astra.omm) extract the same
     orbital elements correctly.
  2. All unit conversions (degrees→radians, rev/day→rad/min) applied
     inside astra.omm are correct.
  3. The physics engine receives bit-for-bit equivalent inputs regardless
     of which data format was used.

Note on small non-zero deltas (~0.001–0.01 km):
  Tiny differences can arise from Julian Date epoch rounding in the two
  parsers. The TLE parser uses the TLE epoch format (YYDDD.FRAC), while
  the OMM parser converts ISO-8601 epoch strings to Julian Date.
  Both are correct, but may differ at sub-millisecond precision.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import astra

GROUP = "gps-ops"      # GPS operational constellation — small, well-known
PROPAGATE_HOURS = 2.0  # Window to compare positions over
STEP_MINUTES = 5.0     # Time resolution


def _match_pairs(tles, omms):
    """Return a list of (SatelliteTLE, SatelliteOMM) matched by NORAD ID."""
    omm_map = {o.norad_id: o for o in omms}
    pairs = []
    for tle in tles:
        if tle.norad_id in omm_map:
            pairs.append((tle, omm_map[tle.norad_id]))
    return pairs


def main():
    print("=" * 60)
    print("  ASTRA-Core Example 05: TLE vs OMM Comparison")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Fetch the same constellation in both formats
    # ------------------------------------------------------------------
    print(f"\n[1] Fetching '{GROUP}' group from CelesTrak in TLE format...")
    tles = astra.fetch_celestrak_group(GROUP, format="tle")
    print(f"    Loaded {len(tles)} SatelliteTLE objects.")

    print(f"\n    Fetching '{GROUP}' group from CelesTrak in OMM format...")
    omms = astra.fetch_celestrak_group(GROUP, format="json")
    print(f"    Loaded {len(omms)} SatelliteOMM objects.")

    # ------------------------------------------------------------------
    # 2. Match objects by NORAD ID
    # ------------------------------------------------------------------
    pairs = _match_pairs(tles, omms)
    print(f"\n[2] Matched {len(pairs)} paired objects by NORAD ID.")
    if not pairs:
        print("    No matching NORAD IDs found — cannot compare. Exiting.")
        return

    # ------------------------------------------------------------------
    # 3. Build a shared time grid anchored to the first TLE's epoch
    # ------------------------------------------------------------------
    anchor_jd = pairs[0][0].epoch_jd
    n_steps = int(PROPAGATE_HOURS * 60 / STEP_MINUTES)
    times_jd = anchor_jd + np.arange(n_steps) * STEP_MINUTES / 1440.0
    print(f"\n[3] Propagation grid: {n_steps} steps of {STEP_MINUTES} min"
          f" over {PROPAGATE_HOURS} h from JD {anchor_jd:.4f}")

    # ------------------------------------------------------------------
    # 4. Propagate TLE and OMM lists separately
    # ------------------------------------------------------------------
    tle_list = [p[0] for p in pairs]
    omm_list = [p[1] for p in pairs]

    print("\n[4] Propagating TLE objects...")
    traj_tle = astra.propagate_many(tle_list, times_jd)

    print("    Propagating OMM objects...")
    traj_omm = astra.propagate_many(omm_list, times_jd)

    # ------------------------------------------------------------------
    # 5. Compute positional deltas per object
    # ------------------------------------------------------------------
    print("\n[5] Positional delta (TLE vs OMM) per satellite:")
    print(f"    {'Name':<26} {'NORAD':>7}  {'Mean Δ (km)':>12}  {'Max Δ (km)':>12}")
    print("    " + "-" * 66)

    all_deltas = []
    for tle, omm in pairs:
        pos_tle = traj_tle.get(tle.norad_id)
        pos_omm = traj_omm.get(omm.norad_id)

        if pos_tle is None or pos_omm is None:
            continue

        # 3-D Euclidean distance at each time step (km)
        diff = np.linalg.norm(pos_tle - pos_omm, axis=1)
        all_deltas.extend(diff)

        print(f"    {tle.name:<26} {tle.norad_id:>7}  "
              f"{diff.mean():>12.4f}  {diff.max():>12.4f}")

    # ------------------------------------------------------------------
    # 6. Overall summary
    # ------------------------------------------------------------------
    if all_deltas:
        arr = np.array(all_deltas)
        print(f"\n[6] Overall fleet summary across {len(pairs)} objects:")
        print(f"    Mean positional delta : {arr.mean():.6f} km")
        print(f"    Max  positional delta : {arr.max():.6f} km")
        print(f"    Std  positional delta : {arr.std():.6f} km")

        threshold_km = 1.0
        if arr.max() < threshold_km:
            print(f"\n✅ PASS — Max delta < {threshold_km} km. "
                  "TLE and OMM pipelines are computationally equivalent.")
        else:
            print(f"\n⚠️  WARN — Max delta ≥ {threshold_km} km. "
                  "Large differences may indicate an epoch mismatch between "
                  "CelesTrak's TLE and OMM data update cycles.")

    print()


if __name__ == "__main__":
    main()
