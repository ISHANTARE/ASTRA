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

1. Builds a **3D spatial index** (`scipy.spatial.cKDTree`) over positions at each coarse step to keep only pairs within a **large** distance threshold.
2. Refines **time of closest approach (TCA)** with **cubic splines** on the time axis.
3. Optionally uses **SGP4 velocities** (`vel_map`) at TCA instead of differentiating noisy position splines—important for eccentric LEO.

**Effective radius:** When **OMM** supplies RCS or you attach dimensions, the code can use a **dynamic** collision radius instead of a generic default.

---

## 5. Covariance and probability of collision (P_c)

A miss distance alone is not enough—you need **uncertainty**. ASTRA supports:

* **Encounter-plane (Chan / Foster–type) methods** — fast; assumes **nearly straight-line** relative motion for the analytical shortcut. Near **direct hits**, a **2D quadrature** over the collision disk can be more appropriate when only 3×3 covariances exist.
* **6×6 Monte Carlo** — samples the combined Gaussian uncertainty and counts hits inside a hard sphere; requires proper **6×6** covariances (e.g. from CDMs).

**`estimate_covariance()`** grows a **diagonal RTN** heuristic from time since epoch and solar flux—it is **not** substitute for orbit-determination covariances. For operational thresholds, prefer **CDM** inputs. **Strict mode** can forbid the heuristic path.

**CDM XML** is parsed with **defusedxml** to reduce classic XML abuse risks.

---

## 6. Cowell propagation, maneuvers, and environmental models

**Cowell** integrates $\dot{r} = v$, $\dot{v} = a$ in inertial space with:

* **Gravity:** Two-body + **J₂, J₃, J₄** (WGS-84).
* **Drag:** Empirical density (F10.7, Ap) when space weather is loaded; co-rotating atmosphere.
* **Third body:** Point-mass **Sun** and **Moon** using **JPL DE421** positions when the ephemeris is available.
* **SRP:** Cannonball model, flux from 1 AU, optional **cylindrical Earth umbra** (full shadow vs full sun; **no penumbra**).
* **Finite burns:** Thrust in **VNB** or **RTN**, with **mass depletion** (Tsiolkovsky-style $\dot{m}$).

The fast path uses **Numba** (`fastmath=True`). Tiny numerical differences vs the pure-Python acceleration are normal—compare **integrated trajectories**, not bitwise forces.

**Caches:** DE421, IERS, and space-weather files download to a user cache (override with **`ASTRA_DATA_DIR`**). Plan ahead for **air-gapped** machines.

---

## 7. Data Source Priority Hierarchy

ASTRA-Core implements a fail-safe data hierarchy to ensure the best available observations are used before falling back on models. This hierarchy is controlled by strict mode and environment variables (e.g. `ASTRA_SPACEBOOK_ENABLED="true"`).

1. **Spacebook / COMSPOC (Primary)** — High-fidelity observational data.
   - *Ephemeris & Uncertainty:* Synthetic Covariance files containing 6x6 observational covariance and precise numerical reference states mapped accurately over time.
   - *Space Weather:* Live daily observations retrieved from COMSPOC endpoints.
   - *Orbits:* XP-TLEs with precision parameters.

2. **CelesTrak / Space-Track (Secondary)** — The classic catalog fallback.
   - *Orbits:* Standard TLEs / OMM elements.
   - *Space Weather:* CelesTrak SW files downloaded and cached locally.

3. **Synthetic Defaults (Last Resort)** — Empirical models.
   - *Covariance:* Time-based degradation models applying generic drag/SRP variances (`estimate_covariance()`). NOTE: Forbidden in **strict mode** without explicit configuration.

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

Both `SatelliteTLE` and `SatelliteOMM` are accepted wherever the type hint says **`SatelliteState`**. Internally, `_build_satrec` either calls **TLE parsing** or **sgp4init** from OMM numbers with correct **unit conversions** (degrees → radians, rev/day → rad/min, ISO epoch → Julian Date). Spacebook's **XP-TLEs** are seamlessly converted directly into `SatelliteOMM` types ensuring parameter compliance.

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

1. **Ephemeris span:** Bundled **DE421** is nominally **~1900–2050**. Outside that, use another kernel and validate.
2. **Atmosphere:** Not NRLMSISE; not a full re-entry tool.
3. **SRP:** Simplified geometry—no penumbra, no detailed spacecraft bus model.
4. **P_c:** Only as good as the **covariances** you pass in.
5. **Network providers:** Respect CelesTrak, Space-Track, and Spacebook **rate limits** and terms—cache catalogs for production.
6. **Certification:** Automated tests check consistency and regressions; they do not replace **your** independent validation if your process requires it.

---

## 10. Strict vs relaxed mode

**`astra.config.ASTRA_STRICT_MODE`** (or **`set_strict_mode(True)`** for thread-safe updates):

* **Relaxed (default):** Missing optional data may produce **warnings** and **fallbacks** (e.g. simplified Sun/Moon if DE421 cannot load).
* **Strict:** Many of those situations **raise** (`EphemerisError`, `SpaceWeatherError`, etc.) so pipelines do not silently continue with degraded physics.

On import, ASTRA prints a one-line **stderr** banner about the mode—filter it in log aggregation if it is noisy.

---

## 10. Optional visualization

**Plotly** is optional (`pip install "astra-core-engine[viz]"`). The name `plot_trajectories` is loaded **lazily** from `astra.plot` so headless servers do not need a plotting stack.

---

## Further reading

* Vallado, *Fundamentals of Astrodynamics and Applications* — SGP4 and perturbations.
* Park et al., JPL **DE440/DE441** — planetary ephemeris context.
* Foster, Chan, Alfano — collision probability and encounter geometry.
