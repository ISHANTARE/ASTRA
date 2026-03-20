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
1. **Sweep and Prune (Spatial Filtering)**: We project all satellite orbits onto a 1D axis and sort them. We instantly ignore pairs that are nowhere near each other. We then use 3D bounding boxes (Axis-Aligned Bounding Boxes or AABBs) to filter out 99.9% of pairs in milliseconds.
2. **Hermite Spline Interpolation**: For the remaining "close calls" (Conjunctions), we don't just calculate positions frame-by-frame. Instead, we generate a continuous mathematical curve (a cubic spline) of the distance between the two satellites over time. We use calculus to find the mathematical root (minimum) of that curve to discover the exact **Time of Closest Approach (TCA)** down to the millisecond.

## 5. Probability of Collision ($P_c$)
A miss distance of 500 meters doesn't mean much on its own, because radars aren't perfect. We only have *estimations* of a satellite's position, represented mathematically by a 3x3 **Covariance Matrix** (which forms a 3D error ellipsoid around the satellite).

ASTRA-Core uses **Mahalanobis Distance** and **B-Plane Mapping** to find the truth:
- We look at the exact moment of closest approach (TCA).
- We project the 3D error ellipsoids of both the primary and secondary satellites onto a 2D plane perpendicular to their relative velocity (the B-Plane).
- We combine their errors into a single 2D Gaussian probability distribution.
- We integrate over the combined hard-body cross-sectional area of the satellites to answer a single question: *What is the statistical probability (%) that these two objects will physically intersect?*

That final $P_c$ value is the true metric used by mission control centers globally to decide if an evasive orbital maneuver is necessary!
