# 🔭 KNOWMORE: The Science & Math behind ASTRA-Core

ASTRA-Core is a high-performance Python mathematical engine that calculates orbits and predicts satellite collisions. But what exactly is happening under the hood?

This document is designed to teach you the fundamental concepts of Space Situational Awareness (SSA), Orbital Mechanics, and Collision Probability as implemented in this library.

## 1. What is a TLE?
A **Two-Line Element (TLE)** set is the standard data format used by NORAD and the US Space Force to describe the orbit of a satellite at a specific point in time (the **Epoch**). It contains parameters like:
- **Inclination**: The tilt of the orbit relative to the Earth's equator.
- **Eccentricity**: How elliptical (oval) versus circular the orbit is.
- **Mean Anomaly**: Exactly where the satellite is along its orbit oval at the time of the epoch.

## 2. Orbital Propagation (SGP4)
A TLE is useless without an orbital mathematical model. You cannot simply use high-school physics (Kepler's laws) to predict where a satellite will be tomorrow. Earth isn't a perfect sphere (it bulges at the equator, known as the **$\text{J}_2$ perturbation**), the Moon and Sun exert gravitational pull, and Low Earth Orbit (LEO) has atmospheric drag.

**SGP4 (Simplified General Perturbations #4)** is a mathematical model that accounts for these perturbations. You feed SGP4 a TLE and a target time, and it outputs the $(x,y,z)$ position and $(v_x,v_y,v_z)$ velocity of the satellite at that exact moment. ASTRA-Core vectorizes this model to propagate tens of thousands of objects at once across massive time arrays.

## 3. Coordinate Systems
Tracking a satellite requires rigorous conversion between reference frames. ASTRA-Core handles transformations between:
- **TEME (True Equator, Mean Equinox)**: The native output frame of the SGP4 algorithm.
- **ECI (Earth-Centered Inertial)**: A fixed coordinate system anchored relative to the distant stars. Crucial for physics math where Newton's laws apply.
- **ECEF (Earth-Centered, Earth-Fixed)**: A coordinate system that rotates *with* the Earth. Needed to track satellites over a geographic map (Latitude/Longitude/Altitude).
- **Topocentric (AER - Azimuth, Elevation, Range)**: The coordinate system from the perspective of an observer standing on the ground looking up at the sky.

## 4. Conjunction Analysis (Finding Collisions)
When you have 30,000 active objects in space, checking every pair for a collision at every single second over a week is computationally impossible ($O(n^2)$ complexity). 

How ASTRA-Core solves this efficiently:
1. **Temporal Octree Sweeping**: We build a recursive 3D spatial hierarchy (an Octree) containing the bounding boxes of satellite trajectories. By evaluating spatial overlap uniquely across discrete integration intervals, we instantly discard 99.9% of safely passing configurations in $O(N \log N)$ time.
2. **Dynamic Attitude Cross-Sections**: For surviving "close calls", we compute the exact TCA (Time of Closest Approach). Based on the satellite's specific hardware pointing mode (e.g. Nadir Earth-pointing), we dynamically rotate its geometric faces to calculate the exact projected surface area slicing through the probability field.

## 5. Covariance ($P_c$) & The 6x6 State Transition Matrix (STM)
A miss distance of 500 meters doesn't mean much on its own. We only have *statistical estimations* of a satellite's state. ASTRA maps this uncertainty fundamentally:

- **6x6 State Propagation**: We integrate a 6x6 State Transition Matrix alongside the Cowell force model's numerical Jacobian. This correctly ties initial velocity variance into exploding positional uncertainty over time mathematically perfectly.
- **6D Monte Carlo Encounter Sampling**: To calculate actual impact likelihood, we extract our localized 6D $C_0$ covariance. We generate tens of thousands of unique $N(\mu, \Sigma)$ 6D relative state vectors around the TCA. We project each localized error pair rectilinearly across the brief collision window to calculate exact structural minimum distances.
- **Impact %**: The final ratio of structural intersections yields the true Probability of Collision ($P_c$), enabling mission control centers to definitively act on evasive maneuvers.
