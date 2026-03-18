# 06 — ASTRA Core: Algorithm Design

---

## Overview

This document provides step-by-step algorithm specifications for every major computational workflow in ASTRA Core. These descriptions are authoritative and must be followed exactly during implementation.

---

## Algorithm 1: Orbit Propagation Workflow

### 1.1 Single-Object Propagation

**Goal:** Compute position and velocity of one satellite at one time instant.

```
INPUTS:
  - satellite: SatelliteTLE
  - t_since_minutes: float (minutes after satellite's epoch)

ALGORITHM:
  Step 1: Initialize sgp4 satrec object
          satrec = sgp4.io.twoline2rv(satellite.line1, satellite.line2, whichconst)
  
  Step 2: Compute Julian Date of target time
          t_jd = satellite.epoch_jd + t_since_minutes / 1440.0
          jd_int = floor(t_jd)
          jd_frac = t_jd - jd_int
  
  Step 3: Call sgp4 propagator
          error_code, position, velocity = satrec.sgp4(jd_int, jd_frac)
  
  Step 4: Validate error code
          If error_code == 0: return OrbitalState with position/velocity
          If error_code > 0: return OrbitalState with error_code set (NaN position)
  
  Step 5: Package result
          return OrbitalState(
              norad_id=satellite.norad_id,
              t_jd=t_jd,
              position_km=np.array(position),    # TEME, km
              velocity_km_s=np.array(velocity),  # TEME, km/s
              error_code=error_code
          )

OUTPUT:
  - OrbitalState (TEME frame, km, km/s)
```

### 1.2 Batch Propagation Workflow

**Goal:** Propagate N satellites over T timesteps efficiently.

```
INPUTS:
  - satellites: list of M SatelliteTLE
  - time_steps: np.ndarray shape (T,), minutes since each satellite's epoch

ALGORITHM:
  Step 1: Initialize output dict
          trajectories = {}

  Step 2: For each satellite in satellites:
  
    Step 2a: Build sgp4 satrec
             satrec = twoline2rv(sat.line1, sat.line2)
    
    Step 2b: Build Julian Date arrays
             jd_array = sat.epoch_jd + time_steps / 1440.0
             jd_int_array = floor(jd_array)
             jd_frac_array = jd_array - jd_int_array
    
    Step 2c: Call vectorized propagator
             error_codes, positions, velocities = sgp4_array(
                 satrec, jd_int_array, jd_frac_array
             )
             # positions: shape (T, 3), km, TEME
    
    Step 2d: Handle errors
             error_mask = (error_codes != 0)
             positions[error_mask] = np.nan
    
    Step 2e: Store trajectory
             trajectories[sat.norad_id] = positions  # shape (T, 3)
  
  Step 3: Return trajectories  # dict[str, ndarray(T,3)]

CRITICAL RULE:
  sgp4_array() MUST be used—never iterate over time_steps in Python.
  Propagation happens OUTSIDE any pairwise comparison loop.

OUTPUT:
  - TrajectoryMap: dict[norad_id → ndarray shape (T, 3)]
```

---

## Algorithm 2: Multi-Stage Filtering Pipeline

**Goal:** Reduce a catalog of N objects to a manageable subset before any propagation is performed. Each stage eliminates objects using only pre-derived TLE fields.

```
INPUTS:
  - catalog: list[DebrisObject]  (N objects, N may be 50,000)
  - config: FilterConfig

FULL PIPELINE (apply in order):

  ┌─────────────────────────────────────────────────────────┐
  │ Stage 1: ALTITUDE FILTER                                │
  │          O(N), no propagation                           │
  │          Eliminates: objects in wrong orbital regime    │
  ├─────────────────────────────────────────────────────────┤
  │ Stage 2: REGION FILTER                                  │
  │          O(N), no propagation                           │
  │          Eliminates: orbital inclinations incompatible  │
  │          with the target latitude range                 │
  ├─────────────────────────────────────────────────────────┤
  │ Stage 3: TIME WINDOW / TLE AGE FILTER                   │
  │          O(N), no propagation                           │
  │          Eliminates: objects with stale TLEs            │
  └─────────────────────────────────────────────────────────┘
          │
          ▼
  [Survivors: propagate these only]
          │
          ▼
  Step 4: propagate_many(survivors, time_steps)
          → TrajectoryMap
          │
          ▼
  Step 5: (Optional) Pass to conjunction detection

ALGORITHM DETAIL:

  Stage 1 — Altitude Filter:
    FOR each obj in catalog:
      IF NOT (config.min_altitude_km <= obj.altitude_km <= config.max_altitude_km):
        ELIMINATE obj
    Expected reduction: 60–80% of catalog

  Stage 2 — Region Filter (if config.lat_min_deg/lat_max_deg set):
    FOR each obj in altitude_survivors:
      accessible_lats = [-obj.inclination_deg, +obj.inclination_deg]
      requested_lats  = [config.lat_min_deg, config.lat_max_deg]
      IF NOT latitude_ranges_overlap(accessible_lats, requested_lats):
        ELIMINATE obj
    Expected reduction: additional 30–50% of survivors

  Stage 3 — TLE Age Filter:
    FOR each obj in region_survivors:
      age_days = config.t_start_jd - obj.tle.epoch_jd
      threshold = 7.0 if obj.altitude_km < 2000 else 14.0
      IF age_days > threshold:
        ELIMINATE obj
    Expected reduction: typically < 5% (data is usually fresh)

  Step 4 — Propagation of survivors only:
    survivors = stage_3_results
    time_steps = build_time_steps(config.t_start_jd, config.t_end_jd, step=5.0)
    trajectories = propagate_many(survivors, time_steps)

OUTPUT:
  - TrajectoryMap (survivors only)

PERFORMANCE IMPACT:
  - Catalog of 50,000 → after Stage 1: ~5,000
  - After Stage 2: ~1,000
  - After Stage 3: ~950
  - Propagation cost: 950 × 288 steps instead of 50,000 × 288
  - Reduction factor: ~52x fewer propagation calls
```

---

## Algorithm 3: Geospatial Region Filter

**Goal:** Filter orbital objects by geographic region using inclination-based approximation (no propagation required).

```
BACKGROUND:
  A satellite's inclination determines the maximum latitude it can reach.
  A satellite with inclination I passes over all latitudes in [-I, +I].
  Therefore, a satellite CANNOT overfly a latitude band entirely outside this range.

ALGORITHM:
  INPUT:
    - obj: DebrisObject
    - lat_min_deg, lat_max_deg: target latitude bounds (deg)
    - lon_min_deg, lon_max_deg: target longitude bounds (deg)

  Step 1: Latitude pre-check
          If lat_max_deg < -obj.inclination_deg:
            → reject (region is too far south for this orbit)
          If lat_min_deg > obj.inclination_deg:
            → reject (region is too far north for this orbit)

  Step 2: Polar orbit special case
          If obj.inclination_deg > 80.0:
            → accept (polar orbits cover all longitudes, latitude bands broadly)
  
  Step 3: Near-equatorial orbit
          If obj.inclination_deg < 10.0 and lat_min_deg > 15.0:
            → reject (equatorial orbit rarely reaches high latitudes)
  
  Step 4: General case
          → accept (orbit passes through compatible latitude range)
          (longitude filtering is not applied because orbital precession
          causes orbits to drift in longitude over time)

  OUTPUT: bool (True = keep, False = reject)

NOTE:
  This filter is CONSERVATIVE. It may pass objects that do not
  actually intersect the region during the analysis window.
  False positives are acceptable; false negatives are NOT.
  Downstream propagation provides exact positions.
```

---

## Algorithm 4: Conjunction Detection (Optimized)

**Goal:** Detect all object pairs with miss distance ≤ threshold_km across the analysis window, without O(n²) brute force.

**Given:** N surviving objects with precomputed trajectories. At scale: N = 100–5,000.

```
INPUTS:
  - trajectories: TrajectoryMap  {norad_id → ndarray(T,3)}
  - times_jd: ndarray(T,)
  - threshold_km: float = 5.0
  - coarse_threshold_km: float = 50.0

FULL ALGORITHM:

  ═══ PHASE 1: ORBITAL ELEMENT COARSE FILTER ═══
  
  Purpose: Eliminate pairs that physically cannot approach each other
           based on orbital geometry.
  
  For each pair (A, B):
  
    Step 1a: Compute semi-major axis difference:
             Δa = |a_A - a_B| in km
             (derived from mean_motion: a = (μ / n²)^(1/3))
    
    Step 1b: If Δa > 2 × coarse_threshold_km:
               → REJECT pair (orbits too far apart in altitude)
    
    Step 1c: Compute inclination difference:
             Δi = |incl_A - incl_B| in degrees
             
    Step 1d: Compute minimum angular separation:
             If Δi > 30°: minimum orbit plane separation is large
             Compute chord distance between orbit planes:
             d_min = a_A × sin(Δi/2) × 2
             If d_min > coarse_threshold_km → REJECT
  
  Expected rejection: 85–95% of all pairs
  
  ═══ PHASE 2: SPATIAL GRID BUCKETING ═══
  
  Purpose: Further reduce candidates using 3D space partitioning.
  This operates on the TRAJECTORY ARRAY, not orbital elements.
  
  Step 2a: Choose grid resolution:
           - Altitude cells: 100 km bands
           - Latitude cells: 10-degree bands
           - Longitude cells: 10-degree bands
  
  Step 2b: For each timestep t in T:
           Build a dictionary: grid_cell → list of norad_ids
           For each object at time t:
             position → altitude, lat, lon
             bucket_key = (alt_band, lat_band, lon_band)
             grid[t][bucket_key].append(norad_id)
  
  Step 2c: Candidate pair generation:
           For each timestep t:
             For each grid cell C:
               For each adjacent cell C' (including C itself, 3x3x3 cube):
                 For each pair (A in C, B in C'):
                   candidate_pairs.add((min(A,B), max(A,B)))
  
  Expected additional rejection: 50–70% of PHASE 1 survivors
  
  ═══ PHASE 3: FINE-GRAINED DISTANCE COMPUTATION ═══
  
  For each candidate pair (A, B):
  
    Step 3a: Extract trajectories:
             traj_A = trajectories[A]  # shape (T, 3)
             traj_B = trajectories[B]  # shape (T, 3)
    
    Step 3b: Compute distances (VECTORIZED):
             distances = np.linalg.norm(traj_A - traj_B, axis=1)  # shape (T,)
    
    Step 3c: Quick reject:
             If np.min(distances) > threshold_km: SKIP
    
    Step 3d: Find TCA:
             t_idx = np.argmin(distances)
             min_dist = distances[t_idx]
             tca_jd = times_jd[t_idx]
  
  ═══ PHASE 4: EVENT CONSTRUCTION ═══
  
  For each qualifying pair (min_dist ≤ threshold_km):
  
    Step 4a: Extract positions at TCA
             pos_A = trajectories[A][t_idx]
             pos_B = trajectories[B][t_idx]
    
    Step 4b: Compute relative velocity (using finite difference on trajectory):
             vel_A ≈ (trajectories[A][t_idx+1] - trajectories[A][t_idx-1]) / (2 * step_seconds)
             vel_B ≈ (trajectories[B][t_idx+1] - trajectories[B][t_idx-1]) / (2 * step_seconds)
             rel_vel = np.linalg.norm(vel_A - vel_B)
    
    Step 4c: Compute collision probability:
             P_c = collision_probability(min_dist, rel_vel)
    
    Step 4d: Classify risk level:
             risk = classify_risk(min_dist, P_c)
    
    Step 4e: Build ConjunctionEvent

OUTPUT:
  - list[ConjunctionEvent] sorted by miss_distance_km ascending

PERFORMANCE ANALYSIS:
  N = 1,000 objects, T = 288 timesteps
  Total pairs without filtering: N²/2 = 499,500
  After Phase 1 rejection (93%): ~35,000 pairs
  After Phase 2 rejection (60%): ~14,000 pairs
  Phase 3 vectorized distance: 14,000 × 288 = 4M operations (milliseconds in NumPy)
  
  FORBIDDEN: A naive triple loop over (A, B, t) would be:
  1,000 × 1,000 × 288 = 288 million operations — impractical.
```

---

## Algorithm 5: Visibility Calculation (Topocentric Transform)

**Goal:** Compute satellite elevation/azimuth as seen from a ground observer, then identify pass intervals.

```
INPUTS:
  - satellite: SatelliteTLE
  - observer: Observer
  - t_start_jd, t_end_jd, step_minutes

ALGORITHM:

  Step 1: Build skyfield time scale
          ts = skyfield.api.load.timescale()
          t_array = ts.from_jd(times_jd)   # skyfield Time object

  Step 2: Build skyfield EarthSatellite
          sat = EarthSatellite(satellite.line1, satellite.line2, satellite.name, ts)

  Step 3: Build skyfield observer location
          obs_loc = wgs84.latlon(
              observer.latitude_deg,
              observer.longitude_deg,
              observer.elevation_m
          )

  Step 4: Compute topocentric difference
          difference = sat - obs_loc
          topocentric = difference.at(t_array)

  Step 5: Convert to Alt/Az
          alt, az, dist = topocentric.altaz()
          elevation_array = alt.degrees    # shape (T,)
          azimuth_array = az.degrees       # shape (T,)

  Step 6: Pass detection
          above_horizon = elevation_array >= observer.min_elevation_deg
          
          # Detect contiguous True segments:
          transitions = np.diff(above_horizon.astype(int))
          rise_indices = np.where(transitions == 1)[0] + 1   # 0→1
          set_indices  = np.where(transitions == -1)[0] + 1  # 1→0

  Step 7: For each pass interval [rise_idx, set_idx]:
          aos_jd = times_jd[rise_idx]
          los_jd = times_jd[set_idx]
          pass_elevations = elevation_array[rise_idx:set_idx]
          tca_idx = np.argmax(pass_elevations) + rise_idx
          tca_jd = times_jd[tca_idx]
          max_el = elevation_array[tca_idx]
          aos_az = azimuth_array[rise_idx]
          los_az = azimuth_array[set_idx]
          duration_s = (los_jd - aos_jd) * 86400.0
          
          yield PassEvent(...)

OUTPUT:
  - list[PassEvent]

COORDINATE NOTES:
  - skyfield internally handles TEME → GCRS → ITRS → Topo correctly
  - Never manually implement the TEME→ITRS matrix rotation
  - The 'difference.at()' call handles all frame conversions
```

---

## Algorithm 6: TLE Epoch to Julian Date Conversion

**Goal:** Convert the two-field TLE epoch (year+day-of-year) to a Julian Date.

```
TLE SOURCE:
  Line 1, columns 19-32: "YYddddddddddd"
  YY = last 2 digits of year
  ddddddddddd = day of year with fractional part

ALGORITHM:
  Step 1: Extract epoch string from line1[18:32]
  Step 2: Parse YY: epoch_str[:2] → int
          If YY >= 57: full_year = 1900 + YY  (Sputnik era: 1957+)
          If YY < 57:  full_year = 2000 + YY
  Step 3: Parse day_of_year (with fraction): float(epoch_str[2:])
  Step 4: Build datetime:
          base = datetime(full_year, 1, 1, tzinfo=utc)
          epoch_dt = base + timedelta(days=day_of_year - 1.0)
  Step 5: Convert to Julian Date:
          jd = 2451545.0 + (epoch_dt - J2000_EPOCH).total_seconds() / 86400.0
          where J2000_EPOCH = datetime(2000, 1, 1, 12, 0, 0, tzinfo=utc)
  
OUTPUT:
  - epoch_jd: float (Julian Date)
```

---

## Algorithm 7: TLE Checksum Validation

```
RULE: The last digit of each TLE line (column 69) is a checksum.
      Letters and spaces count as 0. Digits count as their value. '-' counts as 1.

ALGORITHM:
  Step 1: Extract data portion: line[0:68]  (first 68 characters)
  Step 2: For each character c in data:
          If c.isdigit(): add int(c) to total
          If c == '-':    add 1 to total
          else:           add 0
  Step 3: checksum = total % 10
  Step 4: expected = int(line[68])
  Step 5: If checksum != expected: raise InvalidTLEError

OUTPUT:
  - bool (True = valid checksum)
```
