"""
Example 03: Ground Station Visibility
This script predicts when a specific satellite (e.g., the ISS) 
will be visible from physical coordinates on Earth.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import astra
from datetime import datetime, timezone
import numpy as np

def main():
    print("--- ASTRA-Core: Ground Station Observer Pass Prediction ---")
    
    # Example Location: Los Angeles, CA
    observer = astra.Observer(
        name="Los Angeles Station",
        latitude_deg=34.0522, 
        longitude_deg=-118.2437, 
        elevation_m=71.0
    )
    print(f"Observer Location: Latitude {observer.latitude_deg}, Longitude {observer.longitude_deg}")

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
    target_start_jd = float(astra.convert_time(now, "jd"))
    
    # Calculate passes
    print(f"Calculating overhead passes for the next 24 hours...")
    passes = astra.passes_over_location(
        satellite=iss_tle, 
        observer=observer, 
        t_start_jd=target_start_jd,
        t_end_jd=target_start_jd + 1.0, # 1 day
        step_minutes=1.0
    )
    
    print(f"\nThe ISS will pass overhead {len(passes)} time(s) in the next 24 hours:")
    for i, p in enumerate(passes):
        print(f"\n[Pass {i+1}]")
        print(f"  AOS (Rise Time JD): {p.aos_jd:.5f}")
        print(f"  TCA (Peak Time JD): {p.tca_jd:.5f} at {p.max_elevation_deg:.1f}° elevation")
        print(f"  LOS (Set Time JD):  {p.los_jd:.5f}")

if __name__ == "__main__":
    main()
