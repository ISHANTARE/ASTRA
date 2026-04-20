# KNOWMORE: The science and math behind ASTRA-Core

ASTRA-Core computes orbits, screens conjunctions, and estimates collision risk. This guide explains **what the library is doing conceptually**—TLEs, OMM, frames, screening, P_c, and Cowell forces—and **where simplified models stop matching reality**.

For install steps, API tables, and copy-paste workflows, use [README.md](./README.md). For the full API, see [Read the Docs](https://astra-core.readthedocs.io/en/latest/).

---

## 1. What is a TLE?

A **Two-Line Element (TLE)** set is the classic NORAD / US Space Force format for **mean orbital elements at an epoch**: inclination, eccentricity, RAAN, argument of perigee, mean anomaly, mean motion, etc. A TLE is **not** a Cartesian state—you need **SGP4** to turn (TLE, time) into position and velocity (typically in **TEME**).

TLEs have no mass, area, or ballistic coefficient fields, and the two-digit year convention creates **ambiguity around 2057**—for long horizons, **OMM** is usually clearer.

---

## 2. Orbital propagation (SGP4)

Earth is not a point mass: **J₂** flattening, drag in LEO, and deep-space perturbations are folded into **SGP4**. ASTRA uses the standard **sgp4** Python library and can propagate **many** satellites over the **same** time grid (`propagate_many`) efficiently.

**UT1:** For consistent Earth rotation with the elset, ASTRA can apply **UT1−UTC** from a managed Skyfield timescale when data are available. In the default **relaxed** mode, some failures fall back with warnings; **strict** mode raises `EphemerisError` so you are not silently off by ~1 s class effects at the equator.

---

## 3. Coordinate systems

You will see these frames in the pipeline:

* **TEME (True Equator, Mean Equinox)** — Native output of SGP4; used heavily for screening.
* **ECI / GCRS** — Inertial directions for Newtonian mechanics and some covariance conventions.
* **ITRS / ECEF** — Fixed to Earth; bridge to latitude/longitude/height.
* **Topocentric (ENU)** — East–North–Up at an observer; used for azimuth/elevation and passes.

**Practical note:** Some code paths label inputs as **UTC Julian Date** while Skyfield may use **TT** internally for rotations. The library documents this at important entry points—do not mix scales without reading the docstring for that function.

---

## 4. Conjunction analysis (finding close approaches)

Checking every pair of objects at every timestep is **O(n²)** in objects × time samples. ASTRA:

1. Builds a **3D spatial index** (`scipy.spatial.cKDTree`) over positions at each coarse step to keep only pairs within a **large** distance threshold, yielding ~14.8x runtime speedups.
2. Refines **time of closest approach (TCA)** with **cubic splines** on the time axis, using a dense 1-second scan over the bracketed interval.
3. Optionally uses **SGP4 velocities** (`vel_map`) at TCA instead of differentiating noisy position splines—important for eccentric LEO.
4. Seamlessly performs rigorous **TEME to ECEF** coordinate adjustments using live Spacebook **Earth Orientation Parameters (EOP)** (polar motion + UT1−UTC).

**Effective radius:** When **OMM** supplies RCS or object dimensions are attached, the code uses a **dynamic** collision radius instead of a generic 5-metre default. The priority order is: explicit dimensions → OMM RCS → `DebrisObject.radius_m` → 5 m fallback.

---

## 5. Covariance and probability of collision (P_c)

A miss distance alone is not enough—you need **uncertainty**. ASTRA supports three paths:

* **Encounter-plane (Chan / Foster-type) methods** — fast; assumes **nearly straight-line** relative motion for the analytical shortcut. Near **direct hits** or co-orbital trajectories, an exact **`scipy.integrate.dblquad` 2D Gaussian integration** over the collision disk dynamically replaces the point approximation.
* **6×6 Monte Carlo** — samples the combined Gaussian uncertainty traversing continuous curvilinear paths, resolving exact collision counts inside a hard sphere; requires proper **6×6** covariances (e.g. from CDMs or Spacebook `SynCoPate`).
* **STM covariance propagation** — `propagate_covariance_stm(cov0, Phi)` propagates an initial **6×6** covariance forward using the State Transition Matrix. The force-model Jacobian includes analytical **J₂ partial derivatives** (Montenbruck & Gill §3.2.4) and a co-rotating drag correction so variance is not lost.

**`estimate_covariance()`** grows a **diagonal RTN** heuristic from time since epoch and solar flux—it is **not** a substitute for orbit-determination covariances. For operational thresholds, prefer **CDM** inputs. **Strict mode** can forbid the heuristic path.

**CDM XML** is parsed with **defusedxml** to reduce classic XML abuse risks.

---

## 6. Cowell propagation, maneuvers, and environmental models

**Cowell** integrates $\dot{r} = v$, $\dot{v} = a$ in inertial space with:

* **Gravity:** Two-body + **J₂, J₃, J₄** (WGS-84).
* **Drag:** NRLMSISE-00 atmospheric density (F10.7, Ap) when space weather is loaded. The Numba-JIT path calls `_nrlmsise00_density_njit` directly; the Python path delegates to `atmospheric_density_empirical`. Both are calibrated to ρ(400 km, F10.7=150, Ap=15) ≈ 3.7 × 10⁻¹² kg/m³.
* **Third body:** Point-mass **Sun** and **Moon** using high-precision **JPL DE421** positions when the ephemeris is available. Positions are pre-fitted into **piecewise 7-day, 20-node Chebyshev polynomial splines** (`_eval_cheb_3d_njit`) and queried entirely in memory inside the JIT loop to avoid I/O bottlenecks.
* **SRP:** Cannonball model, flux from 1 AU, with a high-fidelity **conical Earth umbra/penumbra** model (`_srp_illumination_factor_njit`) that computes fractional solar illumination continuously across the penumbra via a circle-circle intersection formula. The canonical `DragConfig` field is `srp_conical_shadow`; the old name `srp_cylindrical_shadow` is **deprecated** and emits a `DeprecationWarning`.
* **Finite burns:** Thrust in **VNB** or **RTN** frames, with **mass depletion** (Tsiolkovsky-style $\dot{m} = -F / (I_{sp} \cdot g_0)$).
* **Segmented orchestration:** The propagator slices the time span at engine ignition/cutoff boundaries so the integrator never steps across a force-model discontinuity.
* **Integrator:** `scipy.integrate.solve_ivp` with `method='DOP853'` (8th-order Dormand–Prince). Coast arcs use `rtol=atol=1e-8`; powered arcs tighten to `rtol=atol=1e-12`.

The fast path uses **Numba** (`@njit(fastmath=True, cache=True)`). Tiny numerical differences vs the pure-Python acceleration path are normal—compare **integrated trajectories**, not bitwise forces.

**Caches:** DE421, IERS, and space-weather files download to `~/.astra/data/` (override with **`ASTRA_DATA_DIR`**). Plan ahead for **air-gapped** machines.

---

## 7. Data Source Priority Hierarchy

ASTRA-Core implements a fail-safe data hierarchy to ensure the best available observations are used before falling back on models. This hierarchy is controlled by strict mode and environment variables (e.g. `ASTRA_SPACEBOOK_ENABLED="true"`).

1. **Spacebook / COMSPOC (Primary)** — High-fidelity observational data.
   - *Ephemeris & Uncertainty:* Synthetic Covariance files containing 6×6 observational covariance and precise numerical reference states mapped accurately over time.
   - *Space Weather:* Live daily observations retrieved from COMSPOC endpoints (6-hour cache TTL; background refresh daemon).
   - *EOP:* Live IERS Earth Orientation Parameters (24-hour cache TTL).
   - *Orbits:* XP-TLEs with precision parameters, converted to `SatelliteOMM` via `xptle_to_satellite_omm`.

2. **CelesTrak / Space-Track (Secondary)** — The classic catalog fallback.
   - *Orbits:* Standard TLEs / OMM elements.
   - *Space Weather:* `SW-All.csv` downloaded and cached locally (24-hour TTL; background refresh after 48 hours since last success).

3. **Synthetic Defaults (Last Resort)** — Empirical models.
   - *Covariance:* Time-based degradation models applying generic drag/SRP variances (`estimate_covariance()`). NOTE: Forbidden in **strict mode** without explicit configuration.
   - *Space Weather:* (F10.7=150, F10.7_adj=150, Ap=15). Forbidden in **strict mode**.

---

## 8. OMM: the modern orbital data standard

OMM (CCSDS) expresses the same *mean-element* information as TLE but with **named JSON fields**—including **mass**, **RCS size**, and **C_D A/m** when the provider supplies them.

### Why it matters for physics

| Need | TLE | OMM |
|------|-----|-----|
| Ballistic coefficient for drag | External table | Often in message |
| Collision cross-section | Guessed default | RCS / metadata |
| Mass for Cowell | External table | Often in message |

### How ASTRA unifies formats

Both `SatelliteTLE` and `SatelliteOMM` are accepted wherever the type hint says **`SatelliteState`**. Internally, `_build_satrec` either calls **TLE parsing** or **sgp4init** from OMM numbers with correct **unit conversions** (degrees → radians, rev/day → rad/min, ISO epoch → Julian Date). Spacebook's **XP-TLEs** are seamlessly converted directly into `SatelliteOMM` types via `xptle_to_satellite_omm`.

```
   Spacebook (XP-TLE) ───┐
   CelesTrak (TLE)  ─────┤
   CelesTrak (OMM)  ─────┤
   Space-Track      ─────┤──► make_debris_object() ──► DebrisObject
   Local JSON       ─────┘              │
                                        ▼
                              propagate_*(), find_conjunctions(),
                              filter_altitude(), …
```

---

## 9. What the library does *not* guarantee

1. **Ephemeris span:** Bundled **DE421** is nominally **~1900–2050**. Outside that, use another kernel (e.g. DE440, ~100 MB, covers 1549–2650) and validate.
2. **Atmosphere:** Implements NRLMSISE-00, but is not a dedicated re-entry analysis tool. Density below 100 km and above 1 500 km is returned as 0.
3. **SRP:** Planar circle-circle intersection geometry. For ultra-precise SRP-limited precision orbit determination replace with exact spherical-cap intersection (Vallado 2013, Algorithm 34).
4. **P_c:** Only as good as the **covariances** you pass in.
5. **Network providers:** Respect CelesTrak, Space-Track, and Spacebook **rate limits** and terms—cache catalogs for production.
6. **Certification:** Automated tests check consistency and regressions; they do not replace **your** independent validation if your process requires it.

---

## 10. Strict vs relaxed mode

**`astra.config.ASTRA_STRICT_MODE`** (or **`set_strict_mode(True)`** for thread-safe updates):

* **Relaxed (default):** Missing optional data may produce **warnings** and **fallbacks** (e.g. simplified Sun/Moon if DE421 cannot load; synthetic covariance when CDM is absent).
* **Strict:** Many of those situations **raise** (`EphemerisError`, `SpaceWeatherError`, `PropagationError`, etc.) so pipelines do not silently continue with degraded physics.

On import, ASTRA prints a one-line **stderr** banner about the mode—suppress it with `ASTRA_NO_BANNER=1` if it is noisy in production log aggregation.

---

## 11. JIT warm-up

```python
astra.warmup()
```

Pre-compiles the Numba Cowell kernel and the `find_conjunctions` SpatialIndex path to eliminate cold-start latency. Call once at worker startup before processing real data. The warm-up runs a trivial 1-second propagation and a do-nothing conjunction check on synthetic objects.

---

## 12. Optional visualization

**Plotly** is optional (`pip install "astra-core-engine[viz]"`). The name `plot_trajectories` is loaded **lazily** from `astra.plot` so headless servers do not need a plotting stack. Importing it without Plotly installed raises a clear `ImportError` pointing to the `viz` extra.

---

## Further reading

* Vallado, *Fundamentals of Astrodynamics and Applications* — SGP4 and perturbations.
* Montenbruck & Gill, *Satellite Orbits* — Cowell integrator, force models.
* Park et al., JPL **DE440/DE441** — planetary ephemeris context.
* Foster, Chan, Alfano — collision probability and encounter geometry.
* Picone et al. (2002), *NRLMSISE-00* — empirical atmospheric model.
