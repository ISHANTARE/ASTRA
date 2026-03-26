Installation
============

ASTRA-Core is distributed via PyPI and can be installed via ``pip`` on any OS (Windows, macOS, Linux).

Prerequisites
-------------
- **Python 3.10+**
- A working C-compiler is occasionally required depending on your OS to compile the LLVM/Numba wrappers natively on first run, however, pre-compiled wheels exist for most platforms.

Using pip
---------

The easiest way to install the engine is exactly like any regular Python package:

.. code-block:: bash

    pip install astra-core-engine

This will automatically pull in all required high-performance dependencies:

- ``numpy`` and ``scipy`` (for vectorized ODE integration)
- ``skyfield`` (for DE421 ephemerides and ISO time conversions)
- ``sgp4`` (for analytical propagation)
- ``numba`` (for JIT compiling the Cowell physics loop)
- ``plotly`` (for 3D orbital trajectory rendering)

Verifying Installation
----------------------

To verify ASTRA is successfully hooked into your Python environment, run:

.. code-block:: python

    import astra
    print(astra.__version__)
    
    # Pre-fetch the heavy Skyfield IERS Earth Orientation Parameters (EOP)
    astra.time.prefetch_iers_data_async()
