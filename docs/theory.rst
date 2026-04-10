Theoretical Framework
=====================

This page sketches SSA concepts as reflected in the code. For **operational
limitations** (DE421 span, covariance/P\ :sub:`c` caveats, strict mode), see
:doc:`limitations` and the repository **KNOWMORE.md**.

ASTRA-Core is a high-performance Python mathematical engine that calculates orbits and predicts satellite collisions. This document outlines the fundamental concepts of Space Situational Awareness (SSA), Orbital Mechanics, and Collision Probability as implemented deeply within the library.

1. Two-Line Elements (TLE)
--------------------------
A **Two-Line Element (TLE)** set is the standard data format used by NORAD and the US Space Force to describe the orbit of a satellite at a specific point in time (the **Epoch**). It contains parameters like:

- **Inclination**: The tilt of the orbit relative to the Earth's equator.
- **Eccentricity**: How elliptical (oval) versus circular the orbit is.
- **Mean Anomaly**: Exactly where the satellite is along its orbit oval at the time of the epoch.

2. Orbital Propagation (SGP4)
-----------------------------
A TLE is useless without an orbital mathematical model. You cannot simply use high-school physics (Kepler's laws) to predict where a satellite will be tomorrow. Earth isn't a perfect sphere (it bulges at the equator, known as the $J_2$ perturbation), the Moon and Sun exert gravitational pull, and Low Earth Orbit (LEO) has atmospheric drag.

**SGP4 (Simplified General Perturbations #4)** is a mathematical model that accounts for these perturbations. You feed SGP4 a TLE and a target time, and it outputs the $(x,y,z)$ position and $(v_x,v_y,v_z)$ velocity of the satellite at that exact moment. ASTRA-Core vectorizes this model to propagate tens of thousands of objects at once across massive time arrays.

3. Coordinate Systems
---------------------
Tracking a satellite requires rigorous conversion between reference frames. ASTRA-Core handles transformations between:

- **TEME (True Equator, Mean Equinox)**: The native output frame of the SGP4 algorithm.
- **ECI (Earth-Centered Inertial)**: A fixed coordinate system anchored relative to the distant stars. Crucial for physics math where Newton's laws apply.
- **ECEF (Earth-Centered, Earth-Fixed)**: A coordinate system that rotates *with* the Earth. Needed to track satellites over a geographic map (Latitude/Longitude/Altitude).
- **Topocentric (AER - Azimuth, Elevation, Range)**: The coordinate system from the perspective of an observer standing on the ground looking up at the sky.

4. Conjunction Analysis (Finding Collisions)
--------------------------------------------
When you have 30,000 active objects in space, checking every pair for a collision at every single second over a week is computationally impossible ($O(N^2)$ complexity). 

How ASTRA-Core solves this efficiently:

1. **cKDTree Spatial Partitioning**: We map satellite trajectories into a highly-optimized C++ ``scipy.spatial.cKDTree`` structure. By querying spatial overlap across discrete integration intervals natively in C, we instantly discard 99.9% of safely passing configurations in $O(N \log N)$ time natively bypassing the Python Global Interpreter Lock (GIL), resulting in ~14.8x operational speedups.
2. **Dynamic Attitude Cross-Sections**: For surviving "close calls", we compute the exact TCA (Time of Closest Approach) via sub-second cubic splines. Based on the satellite's specific hardware pointing mode (e.g. Nadir Earth-pointing), we dynamically rotate its geometric faces to calculate the exact projected surface area slicing through the probability field.
3. **High-Precision Ephemeris Integration**: Leverages Spacebook EOP definitions to transform states rigorously, ensuring coordinates align seamlessly.

5. Covariance & State Transition Matrix
---------------------------------------
A miss distance of 500 meters doesn't mean much on its own. We only have *statistical estimations* of a satellite's state. ASTRA maps this uncertainty fundamentally:

- **6x6 State Propagation**: We integrate a 6x6 State Transition Matrix alongside the Cowell force model's numerical Jacobian. This correctly ties initial velocity variance into exploding positional uncertainty over time perfectly, including exact Jacobian corrections for co-rotating atmospheric drag.
- **Spacebook Synthetic Covariance**: Direct integration of Spacebook's ``SynCoPate`` STK ephemeris ensures flight-grade observational covariance is injected natively into probability calculations.
- **2D Quadrature & 6D Monte Carlo**: To calculate actual impact likelihood across all geometric extremes:
  - For very close co-orbital scenarios without full state data, ASTRA utilizes rigorous ``scipy.integrate.dblquad`` Exact 2D Gaussian integrations across the hard-body disk, ensuring point approximations (Chan/Foster) never degrade risk scores.
  - For complex probabilistic sweeps where full 6x6 covariances exist, we perform heavy 6-DOF Monte Carlo continuous-path sampling to resolve exact collision likelihoods without imposing rectilinear mapping limitations.
- **Impact Probability**: The final ratio yields the true Probability of Collision ($P_c$), enabling mission control centers to plot robust avoidance regimens.

6. Active Collision Avoidance Maneuvers
---------------------------------------
Static un-thrusting satellites are rarely the focus of highly critical events. When risk hits a certain threshold, operators must plot **Collision Avoidance (COLA) Maneuvers**, fundamentally altering the numerical propagation chain.

ASTRA-Core uses a **7-DOF Variable Mass Cowell Integrator** for this step:

1. **Attitude-Steered Burns**: Instead of a simple delta-V impulse, we define exact burn durations and engine thrust metrics. At every micro-step of propagation, the physics engine dynamically rotates the spacecraft vector back into the absolute Space frame dynamically.
2. **Numba JIT Accelerated Numerical Integration**: We JIT-compile the 7-DOF core Cowell differential equations (``@njit(fastmath=True)``) directly to machine code. This allows the integrator to dynamically resolve exact Lunar/Solar ephemerides interpolations and $J_2-J_4$ harmonics millions of times per orbit at near-C speeds.
3. **Space Weather & Celestial Truth**: During precision COLA verification, approximations are dumped. The integration loop inherently scales its atmospheric density drag based on **Live F10.7 Solar Flux** fed automatically from the internet and replaces the position of the Sun/Moon with highly rigorous sub-arcsecond **NASA JPL Ephemerides (DE421)**.
