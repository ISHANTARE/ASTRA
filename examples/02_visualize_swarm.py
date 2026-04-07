"""
Example 02: Visualizing the Conjunction Swarm
This script showcases the built-in Plotly 3D visualizer.
It runs the same math as Example 01, but renders an interactive browser map.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import astra
import numpy as np

def main():
    print("--- ASTRA-Core: 3D Conjunction Visualization ---")
    
    # Limit to a smaller subset for faster 3D rendering in the browser
    catalog = astra.fetch_celestrak_active()
    objects = [astra.make_debris_object(tle) for tle in catalog]
    
    # Let's track just the Starlink constellation as a fun example
    starlinks = [obj for obj in objects if "STARLINK" in obj.tle.name.upper()]
    print(f"Tracking {len(starlinks)} Starlink satellites.")

    start_jd = starlinks[0].tle.epoch_jd
    time_steps_min = np.arange(0, 60, 2.0) # Next 1 hour
    times_jd = start_jd + time_steps_min / 1440.0
    
    tles = [obj.tle for obj in starlinks]
    print("Propagating orbits...")
    trajectories = astra.propagate_many(tles, times_jd)
    
    elements_map = {obj.tle.norad_id: obj for obj in starlinks}
    print("Detecting close approaches (< 10km)...")
    events = astra.find_conjunctions(
        trajectories, 
        times_jd, 
        elements_map, 
        threshold_km=10.0,
        coarse_threshold_km=100.0
    )
    
    print(f"Found {len(events)} close approaches.")
    fig = astra.plot_trajectories(trajectories, events, title="Starlink Constellation Risk Map")
    if os.environ.get("ASTRA_EXAMPLE_NO_BROWSER", "").lower() in ("1", "true", "yes"):
        out = Path(__file__).with_suffix(".html")
        fig.write_html(str(out))
        print(f"Skipping browser (ASTRA_EXAMPLE_NO_BROWSER). Wrote: {out}")
    else:
        print("Opening 3D visualizer in your browser...")
        fig.show()

if __name__ == "__main__":
    main()
