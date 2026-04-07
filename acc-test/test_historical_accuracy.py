import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import requests
import astra
import numpy as np
from datetime import datetime, timedelta, timezone
from astra.propagator import propagate_cowell, NumericalState, DragConfig

# Space-Track Credentials precisely as provided
USER = "ishan.tare2005@gmail.com"
PASS = "Ishanspacetrack!005"

def login_spacetrack():
    session = requests.Session()
    login_url = "https://www.space-track.org/ajaxauth/login"
    res = session.post(login_url, data={"identity": USER, "password": PASS}, timeout=10)
    if res.status_code != 200 or "Failed" in res.text:
         raise RuntimeError("Failed to authenticate to Space-Track.")
    return session

def fetch_historical_omm(session, norad_id: int, days_ago: int = 14):
    """Fetch the historic TLE closest to exactly N days ago."""
    now = datetime.now(timezone.utc)
    target_dt = now - timedelta(days=days_ago)
    
    # We query a 2-day window around the target to ensure we capture at least one TLE
    start_str = (target_dt - timedelta(days=1)).strftime('%Y-%m-%d')
    end_str = (target_dt + timedelta(days=1)).strftime('%Y-%m-%d')
    
    url = f"https://www.space-track.org/basicspacedata/query/class/gp_history/NORAD_CAT_ID/{norad_id}/EPOCH/{start_str}--{end_str}/FORMAT/json/LIMIT/1"
    res = session.get(url, timeout=15)
    res.raise_for_status()
    omms = astra.parse_omm_json(res.text)
    if not omms:
        raise ValueError(f"No historical OMM found for {norad_id} around {target_dt}")
    return omms[0]

def fetch_live_omm(session, norad_id: int):
    """Fetch the absolute latest current TLE."""
    url = f"https://www.space-track.org/basicspacedata/query/class/gp/NORAD_CAT_ID/{norad_id}/FORMAT/json/LIMIT/1"
    res = session.get(url, timeout=15)
    res.raise_for_status()
    omms = astra.parse_omm_json(res.text)
    if not omms:
         raise ValueError(f"No live OMM found for {norad_id}")
    return omms[0]

def test_historical_accuracy():
    print("==========================================================================")
    print(" TRUE 14-DAY FORECAST BENCHMARK: ASTRA VS LIVE OBSERVED DATA ")
    print("==========================================================================\n")

    print("[*] Logging into Space-Track (ishan.tare2005@gmail.com)...")
    session = login_spacetrack()
    print("[*] Authentication Successful.\n")

    targets = {"LEO (ISS)": 25544, "MEO (GPS)": 40534}

    for regime, norad_id in targets.items():
        print(f"--- Benchmarking {regime} (NORAD {norad_id}) ---")
        try:
            # 1. Fetch exactly 14-day old orbital data (Historical Truth)
            old_sat = fetch_historical_omm(session, norad_id, days_ago=14)
            print(f"  [1] Fetched 14-day OLD Data (Epoch: JD {old_sat.epoch_jd:.4f})")

            # 2. Fetch the current, Live data (Live Observed Truth)
            live_sat = fetch_live_omm(session, norad_id)
            print(f"  [2] Fetched LIVE Current Data (Epoch: JD {live_sat.epoch_jd:.4f})")
            
            # 3. Use ASTRA's Numerical Integrator to forecast the EXACT path from 
            #    the Old Epoch forward through time to the EXACT Live Epoch
            
            pos_vel_0 = astra.propagate_orbit(old_sat, old_sat.epoch_jd, 0.0)
            
            area_m2 = getattr(old_sat, 'rcs_m2', 10.0) or 10.0
            mass_kg = getattr(old_sat, 'mass_kg', 1000.0) or 1000.0
            
            state0 = NumericalState(
                t_jd=old_sat.epoch_jd,
                position_km=pos_vel_0.position_km.copy(),
                velocity_km_s=pos_vel_0.velocity_km_s.copy(),
                mass_kg=mass_kg
            )
            drag_cfg = DragConfig(cd=2.2, area_m2=area_m2, mass_kg=mass_kg, include_srp=True)
            
            mins_between_epochs = (live_sat.epoch_jd - old_sat.epoch_jd) * 1440.0
            duration_s = mins_between_epochs * 60.0
            
            print(f"  [3] Firing ASTRA High-Fidelity Engine to forecast +{duration_s/86400.0:.2f} days forward...")
            cowell_states = propagate_cowell(
                state0, 
                duration_s=duration_s, 
                dt_out=duration_s, # just one output step exactly mapping to the live epoch
                drag_config=drag_cfg,
                include_third_body=True,
                use_de=True,
                use_empirical_drag=False
            )
            
            predicted_state = cowell_states[-1]

            # 4. Extract the literal observed coordinates from the Live TLE
            live_position = astra.propagate_orbit(live_sat, live_sat.epoch_jd, 0.0)

            # 5. Measure how close ASTRA got to reality!
            diff_km = np.linalg.norm(live_position.position_km - predicted_state.position_km)

            print(f"  [4] Results:")
            print(f"      -> LIVE Observed True Position Today:   {live_position.position_km} km")
            print(f"      -> ASTRA's Extrapolated Forecast:       {predicted_state.position_km} km")
            print(f"      ---------------------------------------------------------")
            print(f"      => ASTRA Accuracy vs REALITY: Discrepancy of {diff_km:.2f} km\n")
            
        except Exception as e:
            print(f"  Failed: {e}\n")

if __name__ == "__main__":
    test_historical_accuracy()
