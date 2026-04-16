# astra/time.py
"""ASTRA Core time conversion utilities.

Provides a unified interface between Python ``datetime``, Julian Dates,
``skyfield`` Time objects, and ISO 8601 strings.

The Skyfield ``Timescale`` is the **managed** IERS-backed instance from
``data_pipeline.get_skyfield_timescale()``, not ``builtin=True``.
"""

from __future__ import annotations

from datetime import datetime, timezone
import threading
from typing import Any, Optional, Union, cast

from skyfield import timelib

from astra.jdutil import datetime_utc_to_jd, jd_utc_to_datetime

# J2000 reference epoch: 2000-01-01T12:00:00 UTC.
_J2000_JD: float = 2451545.0
_J2000_EPOCH: datetime = datetime(2000, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

_cached_ts: Optional[Any] = None
_TS_LOCK = threading.RLock()


def _get_timescale() -> Any:
    """Return the managed Skyfield timescale (IERS finals2000A)."""
    global _cached_ts
    with _TS_LOCK:
        if _cached_ts is None:
            from astra.data_pipeline import get_skyfield_timescale

            _cached_ts = get_skyfield_timescale()
        return _cached_ts


def prefetch_iers_data_async() -> None:
    """Pre-fetch IERS / ephemeris data asynchronously (non-blocking)."""

    def _fetch() -> None:
        _get_timescale()

    import threading as _threading

    _threading.Thread(target=_fetch, daemon=True).start()


def _iso_to_datetime(iso_str: str) -> datetime:
    """Parse an ISO 8601 string to a UTC-aware datetime."""
    cleaned = iso_str.strip()
    if cleaned.endswith("Z"):
        cleaned = cleaned[:-1] + "+00:00"
    dt = datetime.fromisoformat(cleaned)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _datetime_to_jd(dt: datetime) -> float:
    """Convert a UTC datetime to Julian Date."""
    return float(datetime_utc_to_jd(dt))


def _jd_to_datetime(jd: float) -> datetime:
    """Convert a Julian Date to a UTC-aware datetime."""
    return cast(datetime, jd_utc_to_datetime(jd))


def _datetime_to_iso(dt: datetime) -> str:
    """Convert a datetime to ISO 8601 UTC string."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    utc_dt = dt.astimezone(timezone.utc)
    return utc_dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def convert_time(
    value: Union[str, datetime, float],
    to_format: str,
) -> Union[float, datetime, timelib.Time, str]:
    """Universal time format converter.

    Converts between ISO 8601 strings, Python ``datetime`` objects,
    Julian Dates (float), and ``skyfield.timelib.Time`` objects.

    Args:
        value: The time value to convert.  Accepted types:
            - ``str`` — ISO 8601 format ``"YYYY-MM-DDTHH:MM:SSZ"``
            - ``datetime`` — timezone-aware or naive (assumed UTC)
            - ``float`` — Julian Date (UTC)
        to_format: Target format.  Must be one of:
            - ``"jd"``       → returns ``float`` Julian Date
            - ``"datetime"`` → returns ``datetime`` (UTC-aware)
            - ``"skyfield"`` → returns ``skyfield.timelib.Time``
            - ``"iso"``      → returns ISO 8601 string

    Returns:
        Converted time in the requested format.

    Raises:
        ValueError: If ``to_format`` is invalid or ``value`` type unsupported.
    """
    valid_formats = ("jd", "datetime", "skyfield", "iso")
    if to_format not in valid_formats:
        raise ValueError(
            f"Unsupported to_format={to_format!r}. " f"Must be one of {valid_formats}."
        )

    if isinstance(value, str):
        dt = _iso_to_datetime(value)
        jd = _datetime_to_jd(dt)
    elif isinstance(value, datetime):
        jd = _datetime_to_jd(value)
    elif isinstance(value, (int, float)):
        jd = float(value)
    else:
        raise ValueError(
            f"Unsupported input type {type(value).__name__!r}. "
            "Must be str, datetime, or float."
        )

    if to_format == "jd":
        return jd

    if to_format == "datetime":
        return _jd_to_datetime(jd)

    if to_format == "iso":
        return _datetime_to_iso(_jd_to_datetime(jd))

    ts = _get_timescale()
    dt = _jd_to_datetime(jd)
    return ts.utc(
        dt.year,
        dt.month,
        dt.day,
        dt.hour,
        dt.minute,
        dt.second + dt.microsecond * 1e-6,
    )
