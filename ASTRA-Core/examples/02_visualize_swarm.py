"""
Example 02: Visualizing the Conjunction Swarm
This script showcases the built-in Plotly 3D visualizer.
It runs the same math as Example 01, but renders an interactive browser map.
"""

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
    trajectories = astra.propagate_many(tles, time_steps_min)
    
    elements_map = {obj.tle.norad_id: obj for obj in starlinks}
    print("Detecting close approaches (< 10km)...")
    events = astra.find_conjunctions(
        trajectories, 
        times_jd, 
        elements_map, 
        threshold_km=10.0
    )
    
    print(f"Found {len(events)} close approaches. Opening 3D visualizer in your browser...")
    fig = astra.plot_trajectories(trajectories, events, title="Starlink Constellation Risk Map")
    fig.show()

if __name__ == "__main__":
    main()
