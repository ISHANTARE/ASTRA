# astra/config.py
"""ASTRA Core Global Configuration.

This module houses **all** process-global feature flags for the engine.
Centralising flags here prevents scattered ``os.environ.get(...)`` calls
from spreading across modules and becoming impossible to test or override.

Flags
-----
``ASTRA_STRICT_MODE``
    Controls the dual-profile mode (Relaxed vs Flight-Grade).  In strict
    mode every missing or low-fidelity data source raises a typed error
    instead of silently substituting a heuristic fallback.

``SPACEBOOK_ENABLED``
    Controls whether any Spacebook (COMSPOC) network calls are allowed.
    Set the ``ASTRA_SPACEBOOK_ENABLED=false`` environment variable before
    importing ``astra`` to permanently disable all Spacebook I/O, or use
    :func:`set_spacebook_enabled` at runtime (useful in tests / CI).

Thread safety
-------------
Both flags are process-global.  Direct mutation is safe for single-threaded
scripts.  For multi-threaded applications, use the provided ``set_*``
functions which acquire the module-level RLock before updating the flag,
preventing races during concurrent reads/writes.
"""

from __future__ import annotations

import os
import threading

# ---------------------------------------------------------------------------
# Locks — one per flag to avoid cross-flag contention.
# ---------------------------------------------------------------------------
_STRICT_MODE_LOCK: threading.RLock = threading.RLock()
_SPACEBOOK_LOCK: threading.RLock = threading.RLock()

# ---------------------------------------------------------------------------
# ASTRA_STRICT_MODE
# Default: Relaxed mode — synthesises fallback data and emits warnings.
# Set True to enforce strict orbital equations and raise on missing data.
# ---------------------------------------------------------------------------
ASTRA_STRICT_MODE: bool = False


def set_strict_mode(enabled: bool) -> None:
    """Thread-safe setter for :data:`ASTRA_STRICT_MODE`.

    Acquires the module lock before updating the flag, which is required
    when toggling strict mode from a different thread than physics workers.

    Args:
        enabled: ``True`` for flight-grade strict mode, ``False`` for
                 beginner-friendly relaxed mode.
    """
    global ASTRA_STRICT_MODE
    with _STRICT_MODE_LOCK:
        ASTRA_STRICT_MODE = enabled


# ---------------------------------------------------------------------------
# SPACEBOOK_ENABLED  [CF-6 Fix]
# Single authoritative read of ASTRA_SPACEBOOK_ENABLED happens exactly once
# at import time. All modules must import from here, not call os.environ.get
# directly, so that tests can override the flag via set_spacebook_enabled().
# ---------------------------------------------------------------------------
SPACEBOOK_ENABLED: bool = (
    os.environ.get("ASTRA_SPACEBOOK_ENABLED", "true").strip().lower() != "false"
)


def set_spacebook_enabled(enabled: bool) -> None:
    """Thread-safe setter for :data:`SPACEBOOK_ENABLED`.

    Allows tests and CLI tools to enable or disable Spacebook calls without
    restarting the process.  The change takes effect immediately for all
    subsequent calls to any Spacebook-guarded function.

    Args:
        enabled: ``True`` to allow Spacebook network I/O (default),
                 ``False`` to disable all Spacebook calls and force fallback
                 to CelesTrak or raise ``SpacebookError`` where applicable.

    Example::

        import astra.config as cfg
        cfg.set_spacebook_enabled(False)   # disable for offline tests
        # ... run tests ...
        cfg.set_spacebook_enabled(True)    # restore
    """
    global SPACEBOOK_ENABLED
    with _SPACEBOOK_LOCK:
        SPACEBOOK_ENABLED = enabled
