# astra/_numba_compat.py
"""ASTRA Core — Shared Numba Compatibility Shim.

AUDIT-B-01 Fix: Previously three modules (propagator.py, covariance.py,
frames.py) each independently defined their own no-op ``@njit`` decorator
for environments where Numba is not installed.  That 30-line boilerplate was
duplicated verbatim, creating a divergence risk: if the shim behaviour ever
needed to change (e.g. to report a warning once, or to add a ``cache``
argument), all three copies would have to be updated in sync.

This single module is the authoritative source of truth.  All other ASTRA
modules should import from here instead of defining their own shim.

Usage::

    from astra._numba_compat import njit, NUMBA_AVAILABLE

    @njit(fastmath=True, cache=True)
    def my_kernel(x, y):
        return x + y
"""

from __future__ import annotations

import logging as _logging
from typing import Any

__all__ = ["njit", "NUMBA_AVAILABLE"]

_logger = _logging.getLogger(__name__)

try:
    from numba import njit

    NUMBA_AVAILABLE: bool = True
except ImportError:
    NUMBA_AVAILABLE = False

    def njit(*args: Any, **kwargs: Any) -> Any:
        """No-op Numba decorator shim for environments without Numba.

        Handles both bare ``@njit`` and parameterised ``@njit(fastmath=True)``
        call forms so that existing decorated functions drop in without change.
        Functions wrapped by this shim run as plain CPython — slower than the
        JIT-compiled path, but fully correct.
        """

        def _decorator(fn: Any) -> Any:
            return fn

        # @njit used as a bare decorator (no parentheses): args[0] is the fn
        if args and callable(args[0]):
            return _decorator(args[0])
        # @njit(...) used with arguments: return the decorator factory
        return _decorator

    _logger.warning(
        "Numba not installed — ASTRA Numba-accelerated kernels will run in "
        "pure-Python fallback mode.  Performance will be significantly reduced.  "
        "Install Numba with: pip install numba>=0.59.0"
    )
