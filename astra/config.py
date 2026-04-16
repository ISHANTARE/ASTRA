# astra/config.py
"""ASTRA Core Global Configuration.

This module houses stateful application flags governing the dual-profile
architecture of the engine (Beginner Relaxed vs Enterprise Flight-Grade).

Thread safety:
    ``ASTRA_STRICT_MODE`` is a process-global flag. For single-threaded
    scripts, mutate it directly::

        import astra.config as cfg
        cfg.ASTRA_STRICT_MODE = True

    For multi-threaded applications, use ``set_strict_mode()`` which
    acquires the module lock before changing the flag, preventing races
    during concurrent mode reads/writes.
"""

import threading

# Module-level lock for ASTRA_STRICT_MODE updates.
_STRICT_MODE_LOCK: threading.RLock = threading.RLock()

# Default: Relaxed mode — synthesizes fallback data and emits warnings.
# Set True to enforce strict orbital equations and raise on missing data.
ASTRA_STRICT_MODE: bool = False


def set_strict_mode(enabled: bool) -> None:
    """Thread-safe setter for ``ASTRA_STRICT_MODE``.

    Acquires the module lock before updating the flag, which is required
    when toggling strict mode from a different thread than physics workers.

    Args:
        enabled: ``True`` for flight-grade strict mode, ``False`` for
                 beginner-friendly relaxed mode.
    """
    global ASTRA_STRICT_MODE
    with _STRICT_MODE_LOCK:
        ASTRA_STRICT_MODE = enabled
