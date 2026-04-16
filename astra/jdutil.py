"""Julian date ↔ UTC datetime conversions (stdlib only; avoids import cycles).

Public API
----------
jd_utc_to_datetime / datetime_utc_to_jd  — canonical UTC-aware implementations.
jd_to_datetime / datetime_to_jd          — short-form aliases (used by ocm.py etc.).

Timezone note
-------------
``np.datetime64`` has no concept of time-zones (it stores a UTC integer count).
Constructing it from a *timezone-aware* ``datetime`` object emits:

    UserWarning: no explicit representation of timezones available for np.datetime64

This is silenced by anchoring the J2000 reference on an **ISO string** (timezone-
naive, implicitly UTC) so numpy never needs to round-trip through a tzinfo object.
Timezone-awareness is re-attached explicitly on the Python side before returning.
"""

from __future__ import annotations

import numpy as np
from datetime import datetime, timezone

# J2000 reference epoch: 2000-01-01T12:00:00 UTC.
_J2000_JD: float = 2451545.0
_J2000_EPOCH: datetime = datetime(2000, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

# numpy anchor without tzinfo to suppress UserWarning about timezone representation.
# The string "2000-01-01T12:00:00" is unambiguously UTC in Julian-Date arithmetic.
_J2000_NP: np.datetime64 = np.datetime64("2000-01-01T12:00:00", "us")


def jd_utc_to_datetime(jd: float | np.ndarray) -> datetime | np.ndarray:
    """Convert UTC Julian Date(s) to timezone-aware UTC datetime(s).

    Args:
        jd: Julian Date scalar or numpy array.

    Returns:
        Scalar: timezone-aware ``datetime`` (UTC).
        Array:  object-dtype numpy array of timezone-aware ``datetime`` values.

    Example::

        from astra.jdutil import jd_utc_to_datetime
        dt = jd_utc_to_datetime(2451545.0)
        # -> datetime(2000, 1, 1, 12, 0, tzinfo=timezone.utc)
    """
    jd_arr = np.asarray(jd, dtype=float)
    days = jd_arr - _J2000_JD
    # Represent offset as integer microseconds to avoid float precision loss.
    us_offsets = np.round(days * 86400.0 * 1_000_000.0).astype("int64")
    offsets_td64 = us_offsets.astype("timedelta64[us]")
    dt64 = _J2000_NP + offsets_td64

    from typing import cast

    if jd_arr.ndim == 0:
        # Scalar: extract Python datetime and attach UTC
        raw = dt64.item()  # returns datetime without tzinfo (numpy UTC epoch)
        return cast(datetime, raw).replace(tzinfo=timezone.utc)

    # Vectorised array case
    out = np.array(
        [cast(datetime, d).replace(tzinfo=timezone.utc) for d in dt64.astype(object)],
        dtype=object,
    )
    return out.reshape(jd_arr.shape)


def datetime_utc_to_jd(dt: datetime | np.ndarray) -> float | np.ndarray:
    """Convert UTC-aware datetime(s) to Julian Date.

    Args:
        dt: Single timezone-aware ``datetime``, or numpy array of datetimes.
            Naive datetimes are assumed to be UTC.

    Returns:
        Julian Date as ``float`` (scalar) or ``np.ndarray`` (array input).

    Example::

        from astra.jdutil import datetime_utc_to_jd
        from datetime import datetime, timezone
        jd = datetime_utc_to_jd(datetime(2000, 1, 1, 12, tzinfo=timezone.utc))
        # -> 2451545.0
    """
    if isinstance(dt, np.ndarray):
        dt64 = np.asarray(dt, dtype="datetime64[us]")
        delta = dt64 - _J2000_NP
        return _J2000_JD + delta.astype(float) / 86_400_000_000.0

    # Scalar datetime branch — delta is a stdlib timedelta, total_seconds() is valid.
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    dt_utc = dt.astimezone(timezone.utc)
    delta_s: float = (dt_utc - _J2000_EPOCH).total_seconds()
    return _J2000_JD + delta_s / 86400.0


# ---------------------------------------------------------------------------
# Short-form public aliases
# ---------------------------------------------------------------------------
# ``jd_to_datetime`` and ``datetime_to_jd`` are the names that ocm.py and other
# callers import.  They delegate to the canonical implementations above so
# there is exactly one code path.


def jd_to_datetime(jd: float | np.ndarray) -> datetime | np.ndarray:
    """Alias for ``jd_utc_to_datetime`` — convert UTC Julian Date(s) to datetime(s).

    Args:
        jd: Julian Date scalar or array.

    Returns:
        Timezone-aware UTC datetime, or object array of datetimes.

    Example::

        from astra.jdutil import jd_to_datetime
        dt = jd_to_datetime(2451545.0)   # 2000-01-01 12:00:00+00:00
    """
    return jd_utc_to_datetime(jd)


def datetime_to_jd(dt: datetime | np.ndarray) -> float | np.ndarray:
    """Alias for ``datetime_utc_to_jd`` — convert UTC datetime(s) to Julian Date.

    Args:
        dt: Timezone-aware datetime, or numpy array of datetimes.
            Naive datetimes are assumed UTC.

    Returns:
        Julian Date float, or numpy array of floats.

    Example::

        from astra.jdutil import datetime_to_jd
        from datetime import datetime, timezone
        jd = datetime_to_jd(datetime(2000, 1, 1, 12, tzinfo=timezone.utc))
        # -> 2451545.0
    """
    return datetime_utc_to_jd(dt)
