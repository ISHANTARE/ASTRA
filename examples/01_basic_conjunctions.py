"""
Example 01: Basic Conjunction Analysis
This script demonstrates the core workflow of ASTRA-Core:
1. Fetching live satellite TLEs from CelesTrak.
2. Filtering the catalog to a specific subset (e.g., LEO).
3. Propagating the orbits forward in time.
4. Detecting potential collisions using the Sweep-and-Prune algorithm.
"""

import astra
import numpy as np

def main():
    print("--- ASTRA-Core: Basic Conjunction Analysis ---")
    
    # 1. Fetch live data
    print("Fetching active satellite catalog from CelesTrak...")
    catalog = astra.fetch_celestrak_active()
    print(f"Loaded {len(catalog)} active satellites.")

    # 2. Filter the catalog to keep only Low Earth Orbit (LEO) objects
    objects = [astra.make_debris_object(tle) for tle in catalog]
    leo_objects = astra.filter_altitude(objects, min_km=200, max_km=2000)
    print(f"Filtered down to {len(leo_objects)} LEO objects.")

    # 3. Define a time window (e.g., the next 2 hours, checked every 5 minutes)
    start_jd = leo_objects[0].tle.epoch_jd
    time_steps_min = np.arange(0, 120, 5.0)
    times_jd = start_jd + time_steps_min / 1440.0
    
    # 4. Propagate all orbits simultaneously using SGP4 arrays
    print(f"Propagating {len(leo_objects)} orbits over {len(time_steps_min)} time steps...")
    tles = [obj.tle for obj in leo_objects]
    trajectories = astra.propagate_many(tles, times_jd)
    
    # 5. Find conjunctions (miss distance < 5 km)
    print("Running Sweep-and-Prune spatial collision filter...")
    elements_map = {obj.tle.norad_id: obj for obj in leo_objects}
    events = astra.find_conjunctions(
        trajectories, 
        times_jd, 
        elements_map, 
        threshold_km=5.0,
        coarse_threshold_km=50.0
    )
    
    # 6. Output the results
    print(f"\nFound {len(events)} potential conjunction events!")
    for i, event in enumerate(events[:5]):
        print(f"\n[Event {i+1}]")
        print(f"  Target A: NORAD {event.object_a_id}")
        print(f"  Target B: NORAD {event.object_b_id}")
        print(f"  Time of Closest Approach (TCA JD): {event.tca_jd:.5f}")
        print(f"  Miss Distance: {event.miss_distance_km:.2f} km")
        print(f"  Risk Level: {event.risk_level}")
        print(f"  Collision Probability (Pc): {event.collision_probability:.2e}")

    if len(events) > 5:
        print(f"\n...and {len(events) - 5} more events hidden.")

if __name__ == "__main__":
    main()
