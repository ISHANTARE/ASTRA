import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import astra
import requests
import numpy as np
from astra.propagator import propagate_cowell, NumericalState, DragConfig

def fetch_specific_satellite(norad_id: int):
    # Fetch actual empirical data directly to avoid bulk rate limits
    url = f"https://celestrak.org/NORAD/elements/gp.php?CATNR={norad_id}&FORMAT=json"
    headers = {"User-Agent": "ASTRA-Core-Benchmark"}
    res = requests.get(url, headers=headers, timeout=10)
    res.raise_for_status()
    omms = astra.parse_omm_json(res.text)
    if not omms:
         raise ValueError(f"No object found for {norad_id}")
    return omms[0]

def run_benchmark(sat, days=7):
    print(f"\nBenchmarking {sat.name} (NORAD {sat.norad_id}) for {days} days")
    
    # Extract structural metadata available in OMM format but missing in TLEs
    # ensuring the Cowell forces (drag/SRP) execute as physically accurately as possible
    area_m2 = getattr(sat, 'rcs_m2', 10.0) or 10.0
    mass_kg = getattr(sat, 'mass_kg', 1000.0) or 1000.0
    
    # Get initial SGP4 state exactly at epoch
    pos_vel_0 = astra.propagate_orbit(sat, sat.epoch_jd, 0.0)
    if pos_vel_0.error_code != 0:
        print("SGP4 failed to initialize at epoch.")
        return
        
    state0 = NumericalState(
        t_jd=sat.epoch_jd,
        position_km=pos_vel_0.position_km.copy(),
        velocity_km_s=pos_vel_0.velocity_km_s.copy(),
        mass_kg=mass_kg
    )
    
    # Setup high-fidelity dynamics parameters
    drag_cfg = DragConfig(
        cd=2.2,
        area_m2=area_m2,
        mass_kg=mass_kg,
        include_srp=True
    )
    
    duration_s = days * 86400.0
    dt_out = 86400.0 # Output a grid point every 1 full day
    
    # 1. Run Cowell's numerical integrator (Truth baseline)
    # ASTRA's 6-DOF Cowell uses heavily optimized Numba JIT kernels.
    print("  [>] Running Cowell Integrator (High-Fidelity Runge-Kutta 8/7)...")
    try:
        cowell_states = propagate_cowell(
            state0, duration_s, dt_out,
            drag_config=drag_cfg,
            include_third_body=True,
            use_de=True,              # Will gracefully fallback to analytical if Skyfield isn't loaded
            use_empirical_drag=False  # Avoid dynamic space-weather lookup to enforce deterministic test
        )
    except Exception as e:
        print(f"  Cowell numeric propagation failed: {e}")
        return
        
    # 2. Compare against Analytical SGP4
    print("  [>] Cross-validating against rapid SGP4 analytical propagation...")
    print("\n  Day | SGP4 Drift Error vs Cowell Truth (km)")
    print("  --------------------------------------")
    
    max_error = 0.0
    
    for i, c_state in enumerate(cowell_states):
        day = i
        # SGP4 propagation exactly matched to the Cowell time output grid
        mins_since_epoch = (c_state.t_jd - sat.epoch_jd) * 1440.0
        sgp4_pv = astra.propagate_orbit(sat, sat.epoch_jd, mins_since_epoch)
        if sgp4_pv.error_code != 0:
            print(f"  {day:2d}  | ERROR {sgp4_pv.error_code}")
            continue
            
        diff_km = np.linalg.norm(sgp4_pv.position_km - c_state.position_km)
        print(f"  {day:2d}  | {diff_km:>10.2f} km")
        max_error = max(max_error, diff_km)
        
    print(f"  --------------------------------------")
    print(f"  Maximum divergence over {days} days: {max_error:.2f} km")

def main():
    print("==========================================================================")
    print(" ASTRA-Core Long-Duration Accuracy Benchmark (Analytical vs Numerical) ")
    print("==========================================================================")
    print(" This script illustrates the inherent mathematical drift of standard")
    print(" SGP4 algorithms (the analytical industry baseline) compared to")
    print(" ASTRA's production Cowell RK87 numerical physics engine mapping.")
    print("==========================================================================\n")
    
    # 1. Singular Object Long Duration (LEO drifts fast due to drag!)
    # We test the ISS for 7 days
    try:
        iss = fetch_specific_satellite(25544)
        run_benchmark(iss, days=7)
    except Exception as e:
        print(f"ISS bench failed: {e}")
        
    # 2. Multiple Objects (MEO and GEO are much more stable)
    # We test GPS and GOES over 14 full days
    try:
        gps = fetch_specific_satellite(40534)
        run_benchmark(gps, days=14)
    except Exception as e:
        print(f"GPS bench failed: {e}")

    try:
        goes = fetch_specific_satellite(41866)
        run_benchmark(goes, days=14)
    except Exception as e:
        print(f"GOES bench failed: {e}")
        
if __name__ == "__main__":
    main()
