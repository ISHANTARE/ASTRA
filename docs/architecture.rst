Architecture & Physics
======================

ASTRA-Core uses **SGP4** for large catalogs and **Cowell** for high-fidelity
segments. For **model assumptions** (ephemeris span, P\ :sub:`c` inputs, strict
mode), see :doc:`limitations`.

The stack is organized around numerical rigor and clear separation of parsing,
propagation, and screening.

Two Propagation Tiers
---------------------

1. **Analytical SGP4 (Vectorized)**
   The standard for processing massive space catalogs (e.g., all 9,000 active satellites). SGP4 is incredibly fast but limited by the accuracy of the underlying Two-Line Element (TLE) models. ASTRA-Core vectorizes SGP4 natively via NumPy, allowing thousands of trajectories to be projected concurrently.

2. **Numerical Cowell Integrator (7-DOF)**
   For high-fidelity operations, ASTRA implements a Dormand-Prince (RK8(7)) direct integration scheme (``scipy.integrate.solve_ivp``). 
   It simulates the true physical forces acting on the spacecraft:
   
   - **Gravity**: WGS-84 Zonal Harmonics ($J_2$, $J_3$, $J_4$).
   - **Atmospheric Drag**: Jacchia-class empirical atmospheric density models driven by live Space Weather indicators (F10.7, Ap).
   - **Third-Body Dynamics**: Lunar & Solar gravitational perturbations computed directly from JPL DE421 planetary ephemeris.
   - **SRP**: Optional cannonball solar radiation pressure with high-fidelity **conical Earth umbra** and continuous penumbra modeling.

Numba JIT Acceleration
----------------------
The innermost loop of the Cowell integrator—where forces are summed continuously—is JIT-compiled into LLVM machine-code using ``numba``.
To prevent severe I/O bottlenecks during integration, the JPL DE421 planetary positions are pre-fitted into 25-node Chebyshev polynomial splines and queried entirely in memory.
