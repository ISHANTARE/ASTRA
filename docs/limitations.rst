Operational limitations
=========================

ASTRA-Core implements standard models used in research, education, and software
integration. It is **not**, by itself, a certified conjunction or mission-closure
product. Understand these boundaries before using outputs for operations.

Ephemeris and time
------------------

* Default Sun/Moon positions use **JPL DE421** (bundled via Skyfield), nominally
  **~1900–2050**. Simulations or studies beyond that range should use **DE440**
  or another appropriate kernel and validation.
* **UT1−UTC** and Earth orientation come from Skyfield’s IERS loaders when
  available. **Strict mode** raises if required data are missing; relaxed mode
  may warn and fall back.

Atmosphere and drag
-------------------

* Empirical **Jacchia-class** density (space weather when available), **not**
  NRLMSISE. Very low LEO and re-entry analysis need dedicated atmosphere models.

Solar radiation pressure
------------------------

* **Cannonball SRP** with flux scaled from 1 AU. Features a highly realistic **conical Earth umbra** model that scales fractional solar illumination continuously through the penumbra and transitions smoothly into the dark umbra.

Collision probability
---------------------

* **P_c** quality follows **covariance quality**. Heuristic
  ``estimate_covariance()`` is not orbit-determination grade; prefer **CDM**
  covariances for decision support. **Strict mode** can reject heuristic
  covariance paths.
* **Monte Carlo P_c** uses a **linear** relative-motion model per sample; very
  slow co-orbital encounters may need smaller time steps and careful interpretation.

Network and data providers
--------------------------

* **CelesTrak** and **Space-Track** impose rate limits and terms of use. Cache
  catalogs locally for repeated runs.

Validation
----------

* The repository ships **regression and unit tests**, not a bundled GMAT/STK/CARA
  benchmark suite. Perform independent validation if your process requires it.

Strict vs relaxed mode
----------------------

Set ``astra.config.ASTRA_STRICT_MODE`` (or ``astra.set_strict_mode(True)``) so
missing ephemeris, stale space weather, or policy violations **fail fast**
instead of silent fallbacks. See :doc:`installation` and the main README.
