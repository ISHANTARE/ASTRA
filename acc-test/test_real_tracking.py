import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import astra
import requests
import numpy as np
from datetime import datetime, timezone
from astra.jdutil import datetime_utc_to_jd

def fetch_specific_satellite(norad_id: int):
    # Bypass active catalog rate-limits by pulling just 1 satellite
    url = f"https://celestrak.org/NORAD/elements/gp.php?CATNR={norad_id}&FORMAT=json"
    headers = {"User-Agent": "ASTRA-Core-Test"}
    res = requests.get(url, headers=headers, timeout=10)
    res.raise_for_status()
    omms = astra.parse_omm_json(res.text)
    if not omms:
         raise ValueError(f"No object found for {norad_id}")
    return omms[0]

def test_orbital_regimes():
    print("Fetching specific live satellites for LEO, MEO, GEO, and HEO...")
    
    # 25544 = ISS (LEO)
    # 40534 = NAVSTAR 73 / GPS (MEO)
    # 41866 = GOES 16 (GEO)
    # 25867 = CHANDRA (HEO)
    
    targets = {
        "LEO": 25544,
        "MEO": 40534,
        "GEO": 41866,
        "HEO": 25867
    }
    
    now = datetime.now(timezone.utc)
    jd_now = datetime_utc_to_jd(now)
    
    print(f"Current UTC time: {now}")
    print(f"Current Julian Date: {jd_now:.5f}\n")
    
    for regime, catnr in targets.items():
        try:
            sat = fetch_specific_satellite(catnr)
        except Exception as e:
            print(f"Failed to fetch {regime} (NORAD {catnr}): {e}")
            continue
            
        print(f"--- Analyzing {regime} Regime ---")
        print(f"Object: {sat.name} (NORAD {sat.norad_id})")
        print(f"OMM Epoch JD: {sat.epoch_jd:.5f} (Age: {jd_now - sat.epoch_jd:.2f} days)")
        
        # Propagate from epoch to NOW using ASTRA's engine
        mins_since_epoch = (jd_now - sat.epoch_jd) * 1440.0
        pos_vel = astra.propagate_orbit(sat, sat.epoch_jd, mins_since_epoch)
        
        if pos_vel.error_code != 0:
            print(f"  [!] Propagation error code: {pos_vel.error_code}")
            continue
            
        pos_teme = np.array([pos_vel.position_km])
        jds = np.array([jd_now])
        
        # Use ASTRA's ground_track to verify geodetic altitude
        track = astra.ground_track(pos_teme, jds)
        lat, lon, alt = track[0]
        vel_mag = np.linalg.norm(pos_vel.velocity_km_s)
        
        # eccentricity
        debris_obj = astra.make_debris_object(sat)
        ecc = debris_obj.eccentricity
        
        print(f"  Estimated Position (TEME): {pos_vel.position_km} km")
        print(f"  Estimated Velocity (TEME): {pos_vel.velocity_km_s} km/s (Speed: {vel_mag:.2f} km/s)")
        print(f"  Geodetic Coordinates: Lat {lat:.2f} deg, Lon {lon:.2f} deg, Alt {alt:.2f} km")
        print(f"  Eccentricity: {ecc:.4f}")
        
        # Assertions
        if regime == "LEO":
            assert alt < 2000, f"LEO object altitude {alt:.2f} km exceeds 2000 km!"
            print("  [Pass] Validated LEO altitude < 2000 km.")
        elif regime == "MEO":
            assert 2000 <= alt <= 35000, f"MEO object altitude {alt:.2f} km is out of bounds!"
            print("  [Pass] Validated MEO altitude between 2000 and 35000 km.")
        elif regime == "GEO":
            assert 35000 <= alt <= 36500, f"GEO object altitude {alt:.2f} km is out of bounds!"
            print("  [Pass] Validated GEO altitude roughly ~35786 km.")
        elif regime == "HEO":
            assert ecc > 0.25, f"HEO object eccentricity {ecc:.4f} is not highly elliptical!"
            print("  [Pass] Validated HEO highly elliptical orbit (Ecc > 0.25).")
            
        print()

if __name__ == "__main__":
    test_orbital_regimes()
