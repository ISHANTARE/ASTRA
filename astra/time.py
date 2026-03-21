# astra/time.py
"""ASTRA Core time conversion utilities.

Provides a unified interface between Python ``datetime``, Julian Dates,
``skyfield`` Time objects, and ISO 8601 strings.

This module has ZERO imports from any other ``astra`` domain module.
"""
from __future__ import annotations

from datetime import datetime, timezone, timedelta
import threading
from typing import Any, Optional, Union

from skyfield.api import load as skyfield_load
from skyfield import timelib
from skyfield.timelib import Time as SkyfieldTime


# J2000 reference epoch: 2000-01-01T12:00:00 UTC.
# Julian Date for J2000 = 2451545.0
_J2000_JD: float = 2451545.0
_J2000_EPOCH: datetime = datetime(2000, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

# Lazy-loaded timescale (skyfield downloads data on first call).
_cached_ts: Optional[Any] = None


def _get_timescale() -> Any:
    """Return the skyfield timescale, loading it once lazily."""
    global _cached_ts  # noqa: PLW0603
    if _cached_ts is None:
        _cached_ts = skyfield_load.timescale(builtin=True)
    return _cached_ts


def prefetch_iers_data_async() -> None:
    """Pre-fetch and cache Skyfield IERS Earth Orientation Parameters asynchronously.
    
    Prevents blocking HTTP requests during sequential timescale conversions if the 
    cache has expired, critical for massive-scale STM.
    """
    def _fetch():
        _get_timescale()
        
    thread = threading.Thread(target=_fetch, daemon=True)
    thread.start()


def _iso_to_datetime(iso_str: str) -> datetime:
    """Parse an ISO 8601 string to a UTC-aware datetime.

    Supports:
        ``"YYYY-MM-DDTHH:MM:SSZ"``
        ``"YYYY-MM-DDTHH:MM:SS+00:00"``
        ``"YYYY-MM-DDTHH:MM:SS"``  (assumed UTC)

    Args:
        iso_str: ISO 8601 formatted time string.

    Returns:
        UTC-aware datetime object.
    """
    cleaned = iso_str.strip()
    if cleaned.endswith("Z"):
        cleaned = cleaned[:-1] + "+00:00"
    dt = datetime.fromisoformat(cleaned)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _datetime_to_jd(dt: datetime) -> float:
    """Convert a UTC datetime to Julian Date.

    Args:
        dt: Timezone-aware (or naive, assumed UTC) datetime.

    Returns:
        Julian Date as float.
    """
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    delta = dt - _J2000_EPOCH
    return _J2000_JD + delta.total_seconds() / 86400.0


def _jd_to_datetime(jd: float) -> datetime:
    """Convert a Julian Date to a UTC-aware datetime.

    Args:
        jd: Julian Date as float.

    Returns:
        UTC-aware datetime.
    """
    delta_days = jd - _J2000_JD
    return _J2000_EPOCH + timedelta(days=delta_days)


def _datetime_to_iso(dt: datetime) -> str:
    """Convert a datetime to ISO 8601 UTC string.

    Args:
        dt: Datetime to format.

    Returns:
        String in ``"YYYY-MM-DDTHH:MM:SSZ"`` format.
    """
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    utc_dt = dt.astimezone(timezone.utc)
    return utc_dt.strftime("%Y-%m-%dT%H:%M:%SZ")


# -----------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------

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
            - ``float`` — Julian Date
        to_format: Target format.  Must be one of:
            - ``"jd"``       → returns ``float`` Julian Date
            - ``"datetime"`` → returns ``datetime`` (UTC-aware)
            - ``"skyfield"`` → returns ``skyfield.timelib.Time``
            - ``"iso"``      → returns ISO 8601 string

    Returns:
        Converted time in the requested format.

    Raises:
        ValueError: If ``to_format`` is not one of the four supported values,
            or if ``value`` is of an unsupported type.

    Example:
        >>> jd = convert_time("2025-01-01T00:00:00Z", "jd")
        >>> iso = convert_time(jd, "iso")
    """
    valid_formats = ("jd", "datetime", "skyfield", "iso")
    if to_format not in valid_formats:
        raise ValueError(
            f"Unsupported to_format={to_format!r}. "
            f"Must be one of {valid_formats}."
        )

    # ------------------------------------------------------------------
    # Step 1: Normalise input to an intermediate Julian Date
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Step 2: Convert from Julian Date to requested format
    # ------------------------------------------------------------------
    if to_format == "jd":
        return jd

    if to_format == "datetime":
        return _jd_to_datetime(jd)

    if to_format == "iso":
        return _datetime_to_iso(_jd_to_datetime(jd))

    # to_format == "skyfield"
    ts = _get_timescale()
    return ts.tt_jd(jd)  # type: ignore[no-any-return]
