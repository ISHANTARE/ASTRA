import time
import numpy as np
import pytest
from astra.spatial_index import SpatialIndex
from astra.conjunction import find_conjunctions
from astra.models import DebrisObject, SatelliteTLE

def generate_synthetic_trajectories(n_objects=1000, n_steps=100):
    """Generate random trajectories for benchmarking."""
    times_jd = np.linspace(2460000.0, 2460000.0 + n_steps/1440.0, n_steps)
    trajectories = {}
    elements_map = {}
    
    for i in range(n_objects):
        nid = f"{i:05d}"
        # Start at random LEO position
        r0 = (6378.137 + 500.0 + np.random.rand()*500.0)
        theta = np.random.rand() * 2 * np.pi
        phi = np.random.rand() * np.pi
        pos0 = np.array([
            r0 * np.sin(phi) * np.cos(theta),
            r0 * np.sin(phi) * np.sin(theta),
            r0 * np.cos(phi)
        ])
        
        # Simple linear drift (unphysical but good for spatial bench)
        v = (np.random.rand(3) - 0.5) * 0.1 # 100 m/s drift
        traj = np.array([pos0 + v * t * 86400.0 for t in (times_jd - times_jd[0])])
        
        trajectories[nid] = traj
        
        # Dummy element
        dummy_tle = SatelliteTLE(
            norad_id=nid, name=f"SAT-{nid}", line1="", line2="",
            epoch_jd=times_jd[0], object_type="DEBRIS"
        )
        elements_map[nid] = DebrisObject(
            source=dummy_tle, altitude_km=500.0, inclination_deg=0.0,
            period_minutes=90.0, raan_deg=0.0, eccentricity=0.0,
            apogee_km=500.0, perigee_km=500.0, object_class="LEO"
        )
        
    return trajectories, times_jd, elements_map

def benchmark_spatial_index():
    N = 2000
    T = 100
    print(f"Generating synthetic data for {N} objects, {T} steps...")
    trajectories, times_jd, elements_map = generate_synthetic_trajectories(N, T)
    
    # 1. Benchmark Rebuild-Per-Step (Simulated Old Way)
    print("Benchmarking Old Way (Simulated)...")
    start = time.time()
    old_pairs = set()
    for t_idx in range(T):
        idx = SpatialIndex()
        current_positions = {nid: traj[t_idx] for nid, traj in trajectories.items()}
        idx.rebuild(current_positions)
        for a, b in idx.query_pairs(threshold_km=50.0):
            old_pairs.add((min(a, b), max(a, b)))
    old_time = time.time() - start
    print(f"Old Way Time: {old_time:.4f} s")
    
    # 2. Benchmark Unified Trajectory Index (New Way)
    print("Benchmarking New Way (Unified Trajectory-AABB)...")
    start = time.time()
    idx = SpatialIndex()
    idx.rebuild_for_trajectories(trajectories)
    new_pairs = set(idx.query_pairs(threshold_km=50.0))
    new_time = time.time() - start
    print(f"New Way Time: {new_time:.4f} s")
    
    speedup = old_time / new_time if new_time > 0 else 0
    print(f"Speedup: {speedup:.1f}x")
    
    # 3. Accuracy Check
    # New way should be a superset of the old way (more conservative)
    missed = old_pairs - new_pairs
    print(f"Old pairs: {len(old_pairs)}, New pairs: {len(new_pairs)}")
    if missed:
        print(f"ERROR: New method missed {len(missed)} pairs!")
        # Print a few missed pairs to debug
        for a, b in list(missed)[:5]:
            print(f"  Missed: {a}, {b}")
    else:
        print("Accuracy check PASSED (No pairs missed).")
    
    assert not missed
    assert speedup > 2.0

if __name__ == "__main__":
    benchmark_spatial_index()
