import numpy as np
from datetime import datetime, timezone
import astra
import astra.omm
from astra.spacebook import fetch_xp_tle_catalog, fetch_synthetic_covariance_stk, get_norad_guid
from astra.ocm import parse_stk_ephemeris
from astra.orbit import propagate_trajectory
import warnings

# Suppress the numpy datetime warnings
warnings.filterwarnings('ignore', category=UserWarning, module='astra.jdutil')

def run_benchmark():
    # Attempt to use specific satellites if they exist in the catalog
    target_ids = [
        25544,  # ISS (LEO)
        39215,  # Landsat 8 (LEO)
        43226,  # Starlink (LEO)
        43206,  # GPS III (MEO)
        43224,  # GOES 17 (GEO)
    ]
    
    # First, load the XP-TLE catalog
    print("Loading Spacebook XP-TLE catalog...")
    xp_catalog = fetch_xp_tle_catalog()
    xp_map = {int(tle.norad_id): tle for tle in xp_catalog}
    
    # Also load standard TLEs as fallback for MEO/GEO
    print("Loading Spacebook Standard TLE backup...")
    try:
        from astra.spacebook import fetch_tle_catalog
        std_catalog = fetch_tle_catalog()
        std_map = {int(tle.norad_id): tle for tle in std_catalog}
    except:
        std_map = {}
    
    print("\n--- ASTRA-CORE vs COMSPOC REFERENCE EPHEMERIS BENCHMARK ---")
    print(f"{'Satellite':<20} | {'Regime':<6} | {'Span (hr)':<9} | {'Mean Err':<9} | {'Med Err':<9} | {'95th Pctl':<9} | {'Max Err':<9}")
    print("-" * 90)
    
    results = []
    
    for nid in target_ids:
        regime = "LEO" if nid in (25544, 39215, 43226) else ("MEO" if nid == 43206 else "GEO")
        name = xp_map[nid].name if nid in xp_map else f"NORAD {nid}"
        
        if nid in xp_map:
            tle = xp_map[nid]
        elif nid in std_map:
            tle = std_map[nid]
            name += " (Std TLE)"
        else:
            print(f"{name + ' ('+str(nid)+')':<20} | {regime:<6} | {'N/A (No TLE)':<14} | {'-':<16} | {'-':<15}")
            continue
            
        # Convert to OMM to use ASTRA RK87 numerical propagator
        from astra.omm import xptle_to_satellite_omm
        omm = xptle_to_satellite_omm([tle])[0]
        
        try:
            guid = get_norad_guid(nid)
            stk_text = fetch_synthetic_covariance_stk(nid)
            ref_states = parse_stk_ephemeris(stk_text)
        except Exception as e:
            print(f"{name + ' ('+str(nid)+')':<20} | {regime:<6} | {'N/A (No STK data)':<14} | {'-':<16} | {'-':<15}")
            continue
            
        # Get target times from STK
        # Limit to the first 48 hours for practical benchmark speed
        target_jds = []
        ref_positions = []
        
        start_jd = ref_states[0].t_jd
        for state in ref_states:
            if (state.t_jd - start_jd) * 24.0 <= 48.0:
                target_jds.append(state.t_jd)
                ref_positions.append(state.position_km)
                
        target_jds = np.array(target_jds)
        ref_positions = np.array(ref_positions)
        
        span_hr = (target_jds[-1] - target_jds[0]) * 24.0
        
        # Propagate using ASTRA
        try:
            prop_times, prop_pos, prop_vel = astra.orbit.propagate_trajectory(omm, target_jds[0], target_jds[-1], step_minutes=3.0)
            
            # Use spline interpolation to get exact time matches against the STK ephemeris
            from scipy.interpolate import CubicSpline
            spline = CubicSpline(prop_times, prop_pos, bc_type='natural')
            astra_pos = spline(target_jds)
            
            # Compute errors
            errors_km = np.linalg.norm(astra_pos - ref_positions, axis=1)
            mean_err = np.mean(errors_km)
            med_err = np.median(errors_km)
            p95_err = np.percentile(errors_km, 95)
            max_err = np.max(errors_km)
            
            print(f"{name[:18]:<20} | {regime:<6} | {span_hr:>9.1f} | {mean_err:>9.3f} | {med_err:>9.3f} | {p95_err:>9.3f} | {max_err:>9.3f}")
            results.append((mean_err, med_err, p95_err, max_err))
            
        except Exception as e:
            print(f"{name[:18]:<20} | {regime:<6} | {'Prop Fail':<9} | {'-':<9} | {'-':<9} | {'-':<9} | {'-':<9}")
            import traceback
            traceback.print_exc()

    print("-" * 90)
    if results:
        overall_mean = np.mean([r[0] for r in results])
        print(f"Overall Mean 3D Error: {overall_mean:.3f} km")

if __name__ == '__main__':
    run_benchmark()
