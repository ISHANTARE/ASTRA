"""
Example 03: Ground Station Visibility
This script predicts when a specific satellite (e.g., the ISS) 
will be visible from physical coordinates on Earth.
"""

import astra
from datetime import datetime, timezone
import numpy as np

def main():
    print("--- ASTRA-Core: Ground Station Observer Pass Prediction ---")
    
    # Example Location: Los Angeles, CA
    observer = astra.Observer(
        lat_deg=34.0522, 
        lon_deg=-118.2437, 
        alt_m=71.0
    )
    print(f"Observer Location: Latitude {observer.lat_deg}, Longitude {observer.lon_deg}")

    # Fetch ISS TLE
    print("Fetching active catalog to locate the ISS...")
    catalog = astra.fetch_celestrak_active()
    iss_tles = [t for t in catalog if "ISS" in t.name.upper() and "ZARYA" in t.name.upper()]
    if not iss_tles:
        print("Could not find ISS in catalog.")
        return
        
    iss_tle = iss_tles[0]
    print(f"Target Acquired: {iss_tle.name.strip()} (NORAD {iss_tle.norad_id})")

    # Define prediction window: Now up to 24 hours in the future
    now = datetime.now(timezone.utc)
    target_start_jd = astra.convert_time(now).jd
    
    # Check every minute for 24 hours
    time_steps_min = np.arange(0, 1440, 1.0)
    times_jd = target_start_jd + time_steps_min / 1440.0
    
    # Propagate the ISS
    trajectory_map = astra.propagate_many([iss_tle], time_steps_min)
    iss_positions_teme = trajectory_map[iss_tle.norad_id].positions_km
    
    # Calculate passes
    print(f"Calculating overhead passes for the next 24 hours...")
    passes = astra.passes_over_location(
        iss_positions_teme, 
        times_jd, 
        observer, 
        min_elevation_deg=10.0 # Only consider passes > 10 degrees above horizon
    )
    
    print(f"\nThe ISS will pass overhead {len(passes)} time(s) in the next 24 hours:")
    for i, p in enumerate(passes):
        print(f"\n[Pass {i+1}]")
        print(f"  AOS (Rise Time): {p.aos_time.isoformat()}")
        print(f"  TCA (Peak Time): {p.tca_time.isoformat()} at {p.max_elevation_deg:.1f}° elevation")
        print(f"  LOS (Set Time):  {p.los_time.isoformat()}")

if __name__ == "__main__":
    main()
