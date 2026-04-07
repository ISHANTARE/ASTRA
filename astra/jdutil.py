"""Julian date ↔ UTC datetime conversions (stdlib only; avoids import cycles)."""

import numpy as np
from datetime import datetime, timedelta, timezone

# J2000 reference epoch: 2000-01-01T12:00:00 UTC.
_J2000_JD: float = 2451545.0
_J2000_EPOCH: datetime = datetime(2000, 1, 1, 12, 0, 0, tzinfo=timezone.utc)


def jd_utc_to_datetime(jd: float | np.ndarray) -> datetime | np.ndarray:
    """Convert UTC Julian Date(s) to timezone-aware UTC datetime(s)."""
    jd_arr = np.asarray(jd, dtype=float)
    days = jd_arr - _J2000_JD
    seconds = days * 86400.0
    
    # Vectorized datetime construction using timedelta offsets from J2000 epoch
    # We use timedelta64 for array efficiency
    offsets = (seconds * 1e6).astype('timedelta64[us]')
    dt64 = np.datetime64(_J2000_EPOCH) + offsets
    
    # Convert back to standard datetime objects (native Skyfield/ASTRA requirement)
    if jd_arr.ndim == 0:
        # Scalar case: return single timezone-aware datetime
        ts_val = dt64.item().replace(tzinfo=timezone.utc)
        return ts_val
    else:
        # Array case: return object array of datetime
        out = np.array([dt.replace(tzinfo=timezone.utc) for dt in dt64.astype(datetime)], dtype=object)
        return out.reshape(jd_arr.shape)


def datetime_utc_to_jd(dt: datetime | np.ndarray) -> float | np.ndarray:
    """Convert UTC-aware datetime(s) to Julian Date."""
    if isinstance(dt, np.ndarray):
        dt64 = np.asarray(dt, dtype='datetime64[us]')
        delta = dt64 - np.datetime64(_J2000_EPOCH)
        return _J2000_JD + delta.astype(float) / 86400000000.0
    
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    delta = dt.astimezone(timezone.utc) - _J2000_EPOCH
    return _J2000_JD + delta.total_seconds() / 86400.0
