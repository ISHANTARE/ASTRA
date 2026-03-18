# 08 — ASTRA Core: Performance Constraints

---

## 1. Scale Requirements

ASTRA Core MUST operate correctly and within acceptable time bounds at the following catalog sizes:

| Scale | Objects | Context |
|---|---|---|
| Minimum viable | 1,000 | Development testing |
| Standard operation | 10,000 | Typical operational catalog |
| Maximum target | 50,000 | Full space debris catalog (CelesTrak + Space-Track) |

---

## 2. Performance Targets

| Operation | Input Scale | Target Time |
|---|---|---|
| `load_tle_catalog()` | 50,000 TLE triplets | < 10 seconds |
| `filter_altitude()` | 50,000 objects | < 100 ms |
| `filter_region()` | 50,000 objects | < 100 ms |
| `filter_time_window()` | 50,000 objects | < 50 ms |
| `propagate_many()` | 1,000 survivors × 288 steps | < 5 seconds |
| `propagate_many()` | 5,000 survivors × 288 steps | < 25 seconds |
| `find_conjunctions()` | 1,000 trajectories | < 10 seconds |
| `find_conjunctions()` | 5,000 trajectories | < 60 seconds |
| `passes_over_location()` | 1 satellite, 24 hours | < 1 second |

---

## 3. Vectorization Strategy

### Mandatory Vectorization Points

Every operation over `T` timesteps or `N` objects MUST use NumPy vectorization:

```python
# FORBIDDEN — Python loop over timesteps:
distances = []
for t in range(288):
    d = np.sqrt(np.sum((pos_a[t] - pos_b[t])**2))
    distances.append(d)

# REQUIRED — Vectorized:
distances = np.linalg.norm(pos_a - pos_b, axis=1)  # shape (T,)
```

```python
# FORBIDDEN — Python loop in propagation:
positions = []
for t_min in time_steps:
    _, pos, vel = satrec.sgp4(epoch_jd + t_min/1440, 0.0)
    positions.append(pos)

# REQUIRED — Vectorized sgp4_array:
e, r, v = sgp4_array(satrec, jd_ints, jd_fracs)
# r is already shape (T, 3)
```

### NumPy Operations That MUST Be Used

| Operation | NumPy Call | Notes |
|---|---|---|
| 3D distance (T steps) | `np.linalg.norm(diff, axis=1)` | axis=1 for row-wise norm |
| Minimum distance | `np.min(distances)` | Scalar |
| Index of minimum | `np.argmin(distances)` | For TCA detection |
| Boolean masking | `positions[error_mask] = np.nan` | Mark bad propagations |
| Array of time steps | `np.arange(start, end, step)` | Never use range() |
| Stacking results | `np.stack([x, y, z], axis=1)` | Build (T, 3) from components |

---

## 4. Memory Considerations

### Trajectory Array Size Estimates

```
Per object trajectory:
  T × 3 float64 = 288 × 3 × 8 bytes = 6,912 bytes ≈ 6.75 KB per object

For 1,000 survivors:
  1,000 × 6,912 = 6.75 MB

For 5,000 survivors:
  5,000 × 6,912 = 33.75 MB

For 50,000 objects (all propagated — NEVER do this):
  50,000 × 6,912 = 330 MB — THIS IS WHY WE FILTER FIRST
```

### Memory Rules

1. **Never propagate the full catalog**: filtering must reduce to survivors before `propagate_many()` is called
2. **Store trajectories in dict**: `dict[str, np.ndarray]` is efficient and allows O(1) lookup by NORAD ID
3. **Release trajectory memory after use**: Don't hold `TrajectoryMap` references longer than needed
4. **Avoid copying arrays**: Pass array views, not copies, to conjunction detection functions
5. **NaN masking, not deletion**: Use `np.nan` rows in trajectory arrays to mark failed propagations; don't delete rows (preserves time alignment)

---

## 5. Trajectory Precomputation Requirement

**This is the single most important performance rule in ASTRA Core.**

```
MANDATORY: Propagation must occur ONCE per object BEFORE any pairwise analysis.

FORBIDDEN ANTI-PATTERN:
  for object_a in objects:           # O(N)
      for object_b in objects:       # O(N)
          for t in time_steps:       # O(T)
              pos_a = propagate(object_a, t)   # Propagation INSIDE the loop
              pos_b = propagate(object_b, t)
              distance = compute_distance(pos_a, pos_b)

  Total propagation calls: N × N × T = 1,000 × 1,000 × 288 = 288,000,000 CALLS
  This is completely infeasible.

REQUIRED PATTERN:
  trajectories = propagate_many(objects, time_steps)   # Propagate ONCE
  # Total propagation calls: N × T = 1,000 × 288 = 288,000 calls
  
  for pair in candidate_pairs:
      distances = np.linalg.norm(                      # No propagation here
          trajectories[pair.a] - trajectories[pair.b],
          axis=1
      )
```

---

## 6. Batching Strategy

For very large catalogs (> 10,000 survivors) that cannot fit all trajectories in memory simultaneously:

```python
BATCH_SIZE = 500  # objects per batch

def propagate_and_analyze_batched(
    survivors: list[DebrisObject],
    time_steps: np.ndarray,
    threshold_km: float,
) -> list[ConjunctionEvent]:
    
    all_events = []
    batches = chunk(survivors, BATCH_SIZE)
    
    for batch_a in batches:
        traj_a = propagate_many(batch_a, time_steps)
        
        for batch_b in batches:
            traj_b = propagate_many(batch_b, time_steps)
            
            # Analyze this block of pairs:
            events = find_conjunctions_between(traj_a, traj_b, threshold_km)
            all_events.extend(events)
        
        # Release traj_a memory
        del traj_a
    
    return all_events
```

**Note:** Within the same batch, each object is still propagated only once. The `propagate_many` call in the inner loop recomputes trajectories for `batch_b` objects — for large `N` this is unavoidable without holding all `N` trajectories in RAM simultaneously. This is a RAM vs. time trade-off.

---

## 7. Lazy Computation Principle

ASTRA Core applies lazy evaluation throughout:

| Principle | Implementation |
|---|---|
| Filter before compute | Never propagate before all filter stages complete |
| Short-circuit empty results | If any filter stage yields 0 survivors, return immediately |
| Conjunction threshold early-exit | `np.min(distances) > threshold_km` → skip pair immediately |
| Coarse filter before fine | Orbital element checks before trajectory distance computation |

---

## 8. Anti-Patterns Reference Table

| Anti-Pattern | Why It's Forbidden | Correct Alternative |
|---|---|---|
| Full catalog propagation | 50,000 × 288 = 14.4M SGP4 calls | Filter first, propagate survivors |
| Propagation inside pairwise loop | N² × T calls (billions at scale) | Precompute trajectories before loops |
| Python loop over timesteps | 288× slower than vectorized | Use `sgp4_array()` + NumPy |
| O(n²) pair loop without filtering | 1.25 billion pairs at N=50K | Apply orbital element coarse filter first |
| `datetime.now()` inside functions | Breaks determinism and testability | Pass time as explicit parameter |
| Mutable module-level state | Breaks parallelism and testing | Pure functions only |
| Redundant `sgp4_array` calls | Recalculating same object | Store in `TrajectoryMap` and reuse |

---

## 9. Machine-Level Optimization Notes

These are implemented automatically by the library stack — ASTRA Core does not need to implement these manually:

- **BLAS/LAPACK**: NumPy operations over large arrays use multi-threaded BLAS automatically on most platforms
- **C extension for SGP4**: `sgp4.api.Satrec` is a C extension (not pure Python) — 10–100× faster than `sgp4.io.twoline2rv()`
- **Memory layout**: `np.ndarray` arrays use C-contiguous row-major layout; `axis=1` operations are cache-friendly on `(T, 3)` arrays
