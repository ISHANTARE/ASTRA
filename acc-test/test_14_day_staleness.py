import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import astra
import requests
import numpy as np
from astra.propagator import propagate_cowell, NumericalState, DragConfig
from datetime import datetime, timezone

def fetch_specific_satellite(norad_id: int):
    url = f"https://celestrak.org/NORAD/elements/gp.php?CATNR={norad_id}&FORMAT=json"
    headers = {"User-Agent": "ASTRA-Core-Test"}
    res = requests.get(url, headers=headers, timeout=10)
    res.raise_for_status()
    omms = astra.parse_omm_json(res.text)
    if not omms:
         raise ValueError(f"No object found for {norad_id}")
    return omms[0]

def test_14_day_staleness():
    print("==========================================================================")
    print(" 14-DAY STALENESS BENCHMARK: ASTRA Propagation vs LIVE Data ")
    print("==========================================================================")
    
    # We will benchmark the ISS (stiffest drag test) and GPS (no drag, stable)
    targets = {"LEO (ISS)": 25544, "MEO (GPS)": 40534}
    
    for regime, norad_id in targets.items():
        print(f"\n--- Benchmarking {regime} (NORAD {norad_id}) ---")
        try:
            # Step 1: Get the 'Live' TLE from Today
            live_sat = fetch_specific_satellite(norad_id)
            print(f"  [1] Fetched LIVE Data for '{live_sat.name}' (Epoch: {live_sat.epoch_jd:.4f})")
            
            # Step 2: Since we cannot download a literal 14-day-old TLE without an authenticated 
            # Space-Track account, we can simulate what a radar station would have observed 
            # 14 days ago by back-propagating the current orbit exactly 14 days into the past.
            days_ago = 14
            mins_ago = -days_ago * 1440.0
            
            past_state = astra.propagate_orbit(live_sat, live_sat.epoch_jd, mins_ago)
            if past_state.error_code != 0:
                print("      Failed to back-propagate past state.")
                continue
                
            print(f"  [2] Reconstructed satellite's physical state from exactly {days_ago} days ago.")
            
            # Step 3: We now 'pretend' we are a radar operator 14 days ago. We drop this 14-day-old
            # state into ASTRA's high-fidelity predictive physics engine (Cowell RK87) and ask it 
            # to predict where the satellite will be "TODAY", factoring in 14 days of atmospheric drag, 
            # lunar gravity, and solar radiation pressure.
            
            area_m2 = getattr(live_sat, 'rcs_m2', 10.0) or 10.0
            mass_kg = getattr(live_sat, 'mass_kg', 1000.0) or 1000.0
            
            state_14days_ago = NumericalState(
                t_jd=past_state.t_jd,
                position_km=past_state.position_km.copy(),
                velocity_km_s=past_state.velocity_km_s.copy(),
                mass_kg=mass_kg
            )
            
            drag_cfg = DragConfig(
                cd=2.2, area_m2=area_m2, mass_kg=mass_kg, include_srp=True
            )
            
            print(f"  [3] Firing ASTRA High-Fidelity Physics Engine to forecast +{days_ago} days forward...")
            cowell_states = propagate_cowell(
                state_14days_ago, 
                duration_s=days_ago * 86400.0, 
                dt_out=days_ago * 86400.0, # Just output the exact final state
                drag_config=drag_cfg,
                include_third_body=True,
                use_de=True,
                use_empirical_drag=False
            )
            
            predicted_state_today = cowell_states[-1]
            
            # Step 4: Compare ASTRA's 14-day prediction against the LIVE TLE's actual position today.
            live_position_today = astra.propagate_orbit(live_sat, live_sat.epoch_jd, 0.0)
            
            diff_km = np.linalg.norm(live_position_today.position_km - predicted_state_today.position_km)
            
            print(f"  [4] Results:")
            print(f"      -> LIVE True Position Today:   {live_position_today.position_km} km")
            print(f"      -> ASTRA's 14-Day Prediction:  {predicted_state_today.position_km} km")
            print(f"      ---------------------------------------------------------")
            print(f"      => ASTRA Accuracy vs LIVE Data: Discrepancy of {diff_km:.2f} km")
            
            if "LEO" in regime:
                print("      (Note: 14 days of atmospheric drag is heavily unpredictable due to space weather.")
                print("       A discrepancy of < 500km over half a month for LEO is considered excellent predictability).")
            else:
                print("      (Note: MEO experiences no drag. High precision is expected).")
                
        except Exception as e:
            print(f"  Failed: {e}")

if __name__ == "__main__":
    test_14_day_staleness()
