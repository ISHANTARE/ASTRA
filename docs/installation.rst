Installation
==============

ASTRA-Core is published on PyPI as **astra-core-engine** and supports Python
**3.10, 3.11, 3.12, and 3.13** on Windows, macOS, and Linux.

Core install
------------

.. code-block:: bash

    pip install astra-core-engine

This installs runtime dependencies:

* **numpy** (>=1.23.5, <2.3.0) — arrays and vectorized operations
* **scipy** (>=1.11.0) — ODE integration, spatial indexing, quadrature
* **skyfield** (>=1.46) — DE421 ephemeris loader, IERS timescales, frame transforms
* **sgp4** (>=2.23) — SGP4/SDP4 propagation
* **requests** (>=2.31.0) — HTTP for CelesTrak, Space-Track, and Spacebook
* **numba** (>=0.59.0) — JIT for Cowell acceleration kernels
* **defusedxml** (>=0.7.1) — hardened CDM XML parsing

Optional: visualization (Plotly)
---------------------------------

3D trajectory plotting lives in ``astra.plot``. Plotly is **not** required for
core physics. Install it with the **viz** extra:

.. code-block:: bash

    pip install "astra-core-engine[viz]"

The top-level names ``plot_trajectories`` and ``plot_ground_track`` are lightweight
wrappers, so ``from astra import *`` works in a core install. Calling either
function without Plotly installed raises a clear ``ImportError`` pointing to the
``viz`` extra.

Development and tests
---------------------

.. code-block:: bash

    git clone https://github.com/ISHANTARE/ASTRA.git
    cd ASTRA
    pip install -e ".[test]"

The **test** extra includes **pytest >=8.0**, **plotly >=5.18**, and type-stub
packages (``scipy-stubs``, ``types-requests``, ``types-defusedxml``), plus
``mypy`` so the full test suite and type checks run in CI.

Pytest is configured in ``pyproject.toml`` with strict marker/config validation.
Test-time data, Numba, pytest cache files, and temporary files default to
``ASTRA_TEST_CACHE_DIR`` or, if unset, a deterministic directory under the
test tree (``tests/.astra-test-cache``). Set ``ASTRA_TEST_CACHE_DIR`` in CI or
local shells when you need an explicitly controlled cache root outside the
workspace.

Verifying the install
---------------------

.. code-block:: python

    import astra
    print(astra.__version__)   # Should print e.g. "3.6.2"

Optional: pre-fetch IERS / cache data before long runs:

.. code-block:: python

    from astra import time as astra_time
    astra_time.prefetch_iers_data_async()

Override the data directory with the environment variable **ASTRA_DATA_DIR** if
you need a fixed cache location for air-gapped systems. The default cache is
``~/.astra/data/``.

JIT warm-up
-----------

In production worker pools, call ``astra.warmup()`` **once at startup** to
pre-compile all Numba JIT kernels and eliminate first-call latency:

.. code-block:: python

    import astra

    astra.warmup()   # pre-compiles Numba kernels — call once per process

Banner suppression
------------------

ASTRA prints a one-line startup banner to ``stderr`` on import. To suppress it
(e.g. in multi-process workers), set the environment variable:

.. code-block:: bash

    # Windows
    set ASTRA_NO_BANNER=1

    # Linux / macOS
    export ASTRA_NO_BANNER=1

Strict mode (default ON)
-------------------------

**As of v3.6.1, strict mode is enabled by default.** This is the recommended
setting for production systems. When strict mode is enabled, the library
raises typed exceptions instead of silently continuing with degraded data:

.. list-table:: Strict mode exceptions
   :widths: 25 75
   :header-rows: 1

   * - Condition
     - Exception
   * - EOP fetch failure
     - ``EphemerisError``
   * - Covariance dimension mismatch
     - ``ValueError``
   * - Monte Carlo Pc failure
     - ``ValueError``
   * - Invalid space weather
     - ``SpaceWeatherError``
   * - Missing ephemeris data
     - ``EphemerisError``

To disable strict mode for development or backwards compatibility:

.. code-block:: python

    import astra
    astra.config.ASTRA_STRICT_MODE = False

Type hints (PEP 561)
--------------------

The package ships **py.typed** for static analysis tools (mypy, pyright) that
consume inline annotations. ``mypy`` is pre-configured in ``pyproject.toml``
as a CI baseline; the current configuration is intentionally non-strict while
the numerical array-heavy APIs are incrementally typed.

Further reading
---------------

* :doc:`limitations` — operational trust boundaries
* :doc:`quickstart` — minimal workflows
