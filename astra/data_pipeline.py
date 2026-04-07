# astra/data_pipeline.py
"""ASTRA Core High-Fidelity Data Pipeline.

Provides automated, cached access to the real-world physical data
required by the operations-grade force model:

    1. **JPL DE421 Ephemeris** — precise Sun and Moon positions via Skyfield.
       (DE421 covers 1900–2050 at ~17 MB. For 1549–2650, replace with DE440.)
    2. **IERS Earth-Orientation Parameters** — polar motion, UT1-UTC via
       Skyfield's ``finals2000A.all`` loader.
    3. **CelesTrak Space-Weather** — daily F10.7 solar flux and Kp/Ap
       geomagnetic indices for empirical atmospheric density models.

All heavy files are downloaded once and cached to a user-configurable
directory (default ``~/.astra/data/``).  Subsequent calls use the local
cache.  The cache can be explicitly refreshed.

References:
    Park et al. (2021). JPL Planetary and Lunar Ephemerides DE440/DE441.
    IERS Conventions (2010), Technical Note No. 36.
    Vallado & Finkleman (2008). A Critical Assessment of Satellite Drag …
"""
from __future__ import annotations

import csv
import io
import math
import os
import pathlib
import threading
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import requests
from skyfield.api import Loader

from astra.log import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Default cache directory — respects ASTRA_DATA_DIR env var.
_DEFAULT_DATA_DIR = os.environ.get(
    "ASTRA_DATA_DIR",
    str(pathlib.Path.home() / ".astra" / "data"),
)

# CelesTrak space-weather CSV (rolling 5-year historical + predicted).
_CELESTRAK_SW_URL = "https://celestrak.org/SpaceData/SW-All.csv"

# Standard gravitational acceleration (m/s²)
_G0_M_S2 = 9.80665


# ---------------------------------------------------------------------------
# Skyfield Loader (DE440 + IERS EOP)
# ---------------------------------------------------------------------------

_skyfield_loader: Optional[Loader] = None
_skyfield_ts = None
_skyfield_eph = None

# RLock because _ensure_skyfield calls _get_skyfield_loader — must be re-entrant.
_SKYFIELD_INIT_LOCK = threading.RLock()


def _get_skyfield_loader(data_dir: Optional[str] = None) -> Loader:
    """Return a singleton Skyfield Loader pointed at the cache directory."""
    global _skyfield_loader
    with _SKYFIELD_INIT_LOCK:
        if _skyfield_loader is None:
            path = data_dir or _DEFAULT_DATA_DIR
            _skyfield_loader = Loader(path)
            logger.info(f"Skyfield data directory: {path}")
    return _skyfield_loader


def _ensure_skyfield(data_dir: Optional[str] = None):
    """Ensure the Skyfield timescale + ephemeris objects are initialised (thread-safe)."""
    global _skyfield_ts, _skyfield_eph
    with _SKYFIELD_INIT_LOCK:
        if _skyfield_ts is None or _skyfield_eph is None:
            load = _get_skyfield_loader(data_dir)
            # Timescale includes IERS finals2000A for UT1-UTC & polar motion.
            _skyfield_ts = load.timescale()
            # DE421 is a compact (~17 MB) JPL ephemeris covering 1900–2050 (AST-7).
            # Missions or simulations beyond ~2050 should plan DE440 or another BSP;
            # Skyfield may fail or rely on extrapolation outside the file span.
            # DE440 (> 100 MB) covers 1549–2650 but is usually overkill for LEO ops.
            _skyfield_eph = load("de421.bsp")
            logger.info("Skyfield ephemeris (de421.bsp) and IERS timescale loaded.")


def get_skyfield_timescale(data_dir: Optional[str] = None):
    """Return the managed Skyfield ``Timescale`` (IERS finals2000A), after ensuring cache load.

    Use this instead of ad-hoc ``load.timescale(builtin=True)`` so UT1-UTC and
    conversions share one current-EOP source.
    """
    _ensure_skyfield(data_dir)
    return _skyfield_ts


def sun_position_de(t_jd: float, data_dir: Optional[str] = None) -> np.ndarray:
    """Geocentric Sun position from JPL DE421 ephemeris.

    Returns the GCRS (≈ J2000 ECI) position of the Sun at Julian Date
    ``t_jd``, in kilometres.

    This replaces the Meeus analytical approximation with sub-arcsecond
    accuracy.

    Args:
        t_jd: Julian Date (TDB scale is ideal; TT or UTC are acceptable
               for the force-model level of accuracy).
        data_dir: Optional override for the data cache directory.

    Returns:
        Shape (3,) numpy array  [x, y, z]  in km, GCRS frame.
    """
    _ensure_skyfield(data_dir)
    from astra.jdutil import jd_utc_to_datetime
    dt = jd_utc_to_datetime(t_jd)
    t = _skyfield_ts.utc(dt)
    earth = _skyfield_eph["earth"]
    sun = _skyfield_eph["sun"]
    # Geometric / astrometric Sun vector from DE421 (light-time ~8.3 min; sub-km
    # difference vs apparent Sun at this force-model fidelity).
    astrometric = earth.at(t).observe(sun)
    pos_au = astrometric.position.au
    # 1 AU = 149 597 870.7 km
    return np.array(pos_au) * 149597870.7


def sun_position_teme(t_jd: float, data_dir: Optional[str] = None) -> np.ndarray:
    """Geocentric Sun position in TEME frame (km). Used by the propagator.

    Converts the DE421 GCRS output to TEME so the propagation kernel
    can subtract satellite and body positions without a frame mismatch.
    The rotation is TEME.rotation_at(t).T (transpose = inverse for orthogonal matrix).

    Args:
        t_jd: Julian Date.
        data_dir: Optional override for the data cache directory.

    Returns:
        Shape (3,) numpy array  [x, y, z]  in km, TEME frame.
    """
    from skyfield.sgp4lib import TEME
    _ensure_skyfield(data_dir)
    from astra.jdutil import jd_utc_to_datetime
    dt = jd_utc_to_datetime(t_jd)
    t = _skyfield_ts.utc(dt)
    pos_gcrs_km = sun_position_de(t_jd, data_dir)
    # Rotate GCRS → TEME: R_teme_from_gcrs = TEME.rotation_at(t).T
    R_teme_from_gcrs = TEME.rotation_at(t).T
    return R_teme_from_gcrs @ pos_gcrs_km


def moon_position_de(t_jd: float, data_dir: Optional[str] = None) -> np.ndarray:
    """Geocentric Moon position from JPL DE421 ephemeris.

    Returns the GCRS (≈ J2000 ECI) position of the Moon at Julian Date
    ``t_jd``, in kilometres.

    This replaces the Brown lunar theory approximation with sub-arcsecond
    accuracy.

    Args:
        t_jd: Julian Date.
        data_dir: Optional override for the data cache directory.

    Returns:
        Shape (3,) numpy array  [x, y, z]  in km, GCRS frame.
    """
    _ensure_skyfield(data_dir)
    from astra.jdutil import jd_utc_to_datetime
    dt = jd_utc_to_datetime(t_jd)
    t = _skyfield_ts.utc(dt)
    earth = _skyfield_eph["earth"]
    moon = _skyfield_eph["moon"]
    astrometric = earth.at(t).observe(moon)
    pos_au = astrometric.position.au
    return np.array(pos_au) * 149597870.7


def moon_position_teme(t_jd: float, data_dir: Optional[str] = None) -> np.ndarray:
    """Geocentric Moon position in TEME frame (km). Used by the propagator.

    Converts the DE421 GCRS output to TEME so the propagation kernel
    can subtract satellite and body positions without a frame mismatch.

    Args:
        t_jd: Julian Date.
        data_dir: Optional override for the data cache directory.

    Returns:
        Shape (3,) numpy array  [x, y, z]  in km, TEME frame.
    """
    from skyfield.sgp4lib import TEME
    _ensure_skyfield(data_dir)
    from astra.jdutil import jd_utc_to_datetime
    dt = jd_utc_to_datetime(t_jd)
    t = _skyfield_ts.utc(dt)
    pos_gcrs_km = moon_position_de(t_jd, data_dir)
    R_teme_from_gcrs = TEME.rotation_at(t).T
    return R_teme_from_gcrs @ pos_gcrs_km


def get_ut1_utc_correction(t_jd_utc: float | np.ndarray) -> float | np.ndarray:
    """Return UT1-UTC offset in seconds for given UTC Julian Date(s).

    Interprets the input Julian Date as **UTC** (not TT). Builds Skyfield
    ``Time`` via calendar UTC, then uses ``Time.dut1`` (UT1−UTC in seconds).

    Args:
        t_jd_utc: Julian Date in UTC (scalar or array).

    Returns:
        UT1-UTC time offset in seconds.
    """
    from astra.jdutil import jd_utc_to_datetime
    _ensure_skyfield()
    
    t_jd_arr = np.asarray(t_jd_utc)
    dt = jd_utc_to_datetime(t_jd_arr)
    t = _skyfield_ts.utc(dt)
    return t.dut1


# ---------------------------------------------------------------------------
# CelesTrak Space-Weather Cache
# ---------------------------------------------------------------------------

# In-memory cache:  {date_str "YYYY-MM-DD": (F10.7_obs, F10.7_adj, Ap_daily)}
_sw_cache: dict[str, tuple[float, float, float]] = {}
_sw_loaded: bool = False
_sw_last_success: Optional[datetime] = None   # tracks last successful refresh time
_SW_LOCK = threading.RLock()
# Serialize download+parse so two threads that both see _sw_loaded=False
# cannot trigger duplicate downloads.
_SW_DOWNLOAD_LOCK = threading.Lock()
_sw_fetch_thread: Optional[threading.Thread] = None


def _download_space_weather(data_dir: Optional[str] = None) -> str:
    """Download SW-All.csv from CelesTrak and cache locally."""
    path = pathlib.Path(data_dir or _DEFAULT_DATA_DIR)
    path.mkdir(parents=True, exist_ok=True)
    local_file = path / "SW-All.csv"

    logger.info(f"Downloading CelesTrak SW-All.csv → {local_file}")
    resp = requests.get(_CELESTRAK_SW_URL, timeout=60)
    resp.raise_for_status()
    local_file.write_text(resp.text, encoding="utf-8")
    return resp.text


def _parse_sw_csv(text: str) -> None:
    """Parse the CelesTrak CSV into the in-memory cache."""
    global _sw_loaded
    
    new_cache = {}
    reader = csv.reader(io.StringIO(text))
    header = None
    for row in reader:
        if not row or row[0].startswith("#"):
            continue
        if header is None:
            header = [h.strip() for h in row]
            continue

        try:
            date_str = row[0].strip()
            # Column layout (CelesTrak SW-All.csv, 2024-format):
            #  0: DATE          (YYYY-MM-DD)
            #  1: BSRN          (sunspot number)
            #  2: ND            (num sunspot groups)
            #  3: Kp1 … Kp8    (3-hourly Kp indices, cols 3–10)
            # 11: Kp_SUM
            # 12: Ap1 … Ap8    (3-hourly Ap, cols 12–19)
            # 20: Ap_AVG
            # 21: Cp
            # 22: C9
            # 23: ISN
            # 24: F10.7_OBS
            # 25: F10.7_ADJ

            # Safely parse; some future-predicted rows may have blanks.
            f107_obs = float(row[24]) if len(row) > 24 and row[24].strip() else 150.0
            f107_adj = float(row[25]) if len(row) > 25 and row[25].strip() else f107_obs
            ap_daily = float(row[20]) if len(row) > 20 and row[20].strip() else 15.0

            new_cache[date_str] = (f107_obs, f107_adj, ap_daily)
        except (ValueError, IndexError):
            continue   # skip malformed rows

    with _SW_LOCK:
        _sw_cache.clear()
        _sw_cache.update(new_cache)
        _sw_loaded = True

    logger.info(f"Space-weather cache loaded: {len(_sw_cache)} daily records.")


def _background_sw_fetch(data_dir: Optional[str]) -> None:
    """Daemon thread worker to download and parse space-weather cache."""
    global _sw_fetch_thread, _sw_last_success
    try:
        text = _download_space_weather(data_dir)
        _parse_sw_csv(text)
        with _SW_LOCK:
            _sw_last_success = datetime.now(timezone.utc)
    except Exception as exc:
        logger.warning(f"Background Space-Weather fetch failed: {exc}")
    finally:
        with _SW_LOCK:
            _sw_fetch_thread = None

def load_space_weather(data_dir: Optional[str] = None, force_download: bool = False) -> None:
    """Ensure space-weather data is available in memory.

    Downloads the CSV from CelesTrak on first call (or if the local
    cache is older than 24 hours), then parses into memory. If local file exists
    but is stale, spins a background thread to update it without blocking physics loops.

    Thread-safe via double-checked locking. A fast check under
    ``_SW_LOCK`` provides the common O(1) exit.  The slow path acquires
    ``_SW_DOWNLOAD_LOCK`` so only one thread downloads + parses; the rest
    re-check under that lock and exit cleanly once the winner finishes.

    Args:
        data_dir: Override data directory.
        force_download: If True, re-download even if a cached file exists.
    """
    global _sw_fetch_thread

    # Fast path — common case; already loaded.
    with _SW_LOCK:
        if _sw_loaded and not force_download:
            return

    # Slow path — serialise so only one thread downloads + parses.
    with _SW_DOWNLOAD_LOCK:
        # Re-check inside the download lock (DCL "double-check").
        with _SW_LOCK:
            if _sw_loaded and not force_download:
                return  # Another thread finished while we waited.

        path = pathlib.Path(data_dir or _DEFAULT_DATA_DIR)
        local_file = path / "SW-All.csv"

        need_download = force_download or not local_file.exists()
        stale_cache = False

        if not need_download:
            mtime = datetime.fromtimestamp(local_file.stat().st_mtime, tz=timezone.utc)
            age_hours = (datetime.now(timezone.utc) - mtime).total_seconds() / 3600.0
            if age_hours > 24.0:
                stale_cache = True

        if need_download:
            text = _download_space_weather(data_dir)
            _parse_sw_csv(text)
        else:
            text = local_file.read_text(encoding="utf-8")
            _parse_sw_csv(text)

            with _SW_LOCK:
                if stale_cache and _sw_fetch_thread is None:
                    logger.info("Space-weather cache stale. Spawning background refresh daemon...")
                    _sw_fetch_thread = threading.Thread(
                        target=_background_sw_fetch,
                        args=(data_dir,),
                        daemon=True,
                    )
                    _sw_fetch_thread.start()



def get_space_weather(t_jd: float, data_dir: Optional[str] = None) -> tuple[float, float, float]:
    """Retrieve F10.7 solar flux and Ap index for a given Julian Date.

    Returns the observed F10.7, adjusted F10.7, and daily Ap value
    for the calendar date corresponding to ``t_jd``.

    If the exact date is not in the cache (e.g. future predictions),
    returns moderate solar activity defaults (F10.7 = 150 SFU, Ap = 15).

    Args:
        t_jd: Julian Date.
        data_dir: Override data directory.

    Returns:
        Tuple (F10.7_obs, F10.7_adj, Ap_daily).
    """
    load_space_weather(data_dir)

    # Stale cache: background refresh if > 48 h since last successful load
    # global must be declared at function top, before any use of the name
    global _sw_fetch_thread
    with _SW_LOCK:
        if _sw_last_success is not None:
            age_h = (datetime.now(timezone.utc) - _sw_last_success).total_seconds() / 3600
            if age_h > 48.0 and _sw_fetch_thread is None:
                logger.warning("Space-weather cache >48 hours old. Triggering background refresh.")
                _sw_fetch_thread = threading.Thread(
                    target=_background_sw_fetch, args=(data_dir,), daemon=True
                )
                _sw_fetch_thread.start()

    # JD → calendar date via integer-split day/fraction (stable vs float drift)
    # Avoids floating-point rounding errors at midnight boundaries.
    # JD 2451545.0 = 2000-01-01T12:00:00 TT
    from datetime import timedelta
    days_offset = t_jd - 2451545.0
    whole_days = int(days_offset)
    frac_day = days_offset - whole_days
    whole_sec = int(frac_day * 86400.0)
    micro_sec = round((frac_day * 86400.0 - whole_sec) * 1e6)
    dt = datetime(2000, 1, 1, 12, 0, 0, tzinfo=timezone.utc) + timedelta(
        days=whole_days, seconds=whole_sec, microseconds=micro_sec
    )
    date_str = dt.strftime("%Y-%m-%d")

    with _SW_LOCK:
        if date_str in _sw_cache:
            return _sw_cache[date_str]

    # Fallback: check STRICT_MODE before returning synthetic defaults
    from astra import config
    from astra.errors import SpaceWeatherError
    if config.ASTRA_STRICT_MODE:
        raise SpaceWeatherError(
            f"[ASTRA STRICT] No space-weather data available for {date_str}. "
            "Future epoch propagation requires an explicit solar flux forecast. "
            "Run load_space_weather() to refresh data or set ASTRA_STRICT_MODE=False."
        )
    logger.warning(
        f"No space-weather data for {date_str} — using moderate solar activity defaults "
        "(F10.7=150 SFU, Ap=15). Atmospheric drag accuracy is compromised for this epoch."
    )
    return (150.0, 150.0, 15.0)


# ---------------------------------------------------------------------------
# Empirical Atmospheric Density (Jacchia-style with F10.7)
# ---------------------------------------------------------------------------

def atmospheric_density_empirical(
    altitude_km: float,
    f107_obs: float,
    f107_adj: float,
    ap_daily: float,
) -> float:
    """Compute atmospheric density using an empirical model parameterised
    by solar flux and geomagnetic activity.

    This is a Harris-Priester / Jacchia-71 hybrid model that accounts
    for the solar-cycle dependence of the upper atmosphere's temperature
    and, consequently, its scale height and base density.

    The model computes an effective exospheric temperature from F10.7 and
    Ap, then derives density at altitude using a diffusion-equilibrium
    exponential profile.  This is significantly more accurate than the
    static exponential model previously used.

    Args:
        altitude_km: Altitude above Earth's surface [km].
        f107_obs: Observed F10.7 solar flux [SFU].
        f107_adj: 81-day centred average F10.7 [SFU].
        ap_daily: Daily Ap geomagnetic index.

    Returns:
        Atmospheric density in kg/m³.  Returns 0.0 above 1500 km.
    """
    from astra.constants import DRAG_MIN_ALTITUDE_KM, DRAG_MAX_ALTITUDE_KM
    if altitude_km > DRAG_MAX_ALTITUDE_KM or altitude_km < DRAG_MIN_ALTITUDE_KM:
        return 0.0

    # Exospheric temperature estimate (Jacchia-71 simplified)
    # T_inf = T_c + Delta_T(F10.7) + Delta_T(Ap)
    T_c = 379.0  # nighttime minimum temperature (K)

    # Solar flux contribution (dominant driver of upper-atmosphere density)
    delta_T_solar = 3.24 * f107_adj + 1.3 * (f107_obs - f107_adj)

    # Geomagnetic heating contribution
    delta_T_geo = 28.0 * ap_daily**0.4

    T_inf = T_c + delta_T_solar + delta_T_geo

    # Clamp to physical range
    T_inf = max(500.0, min(T_inf, 2500.0))

    # Scale height depends on exospheric temperature and mean molecular mass.
    # At ~400 km, mean molecular mass ≈ 16 amu (atomic oxygen dominated).
    # H = k_B * T / (m * g)  → simplified empirical fit:
    k_boltzmann = 1.380649e-23       # J/K
    amu = 1.66054e-27                 # kg
    g_400 = 8.67                      # m/s², gravity at 400 km

    # Effective molecular mass varies with altitude and solar activity.
    # Below ~200 km, N₂ (28 amu) dominates; above ~500 km, He/H dominate.
    if altitude_km < 200.0:
        m_eff = 25.0 * amu
    elif altitude_km < 500.0:
        # Linear interpolation from 25 amu at 200 km to 16 amu at 500 km
        frac = (altitude_km - 200.0) / 300.0
        m_eff = (25.0 - 9.0 * frac) * amu
    elif altitude_km < 800.0:
        frac = (altitude_km - 500.0) / 300.0
        m_eff = (16.0 - 12.0 * frac) * amu
    else:
        m_eff = 4.0 * amu  # Helium-dominated

    g_local = 9.80665 * (6378.137 / (6378.137 + altitude_km))**2
    H_km = (k_boltzmann * T_inf) / (m_eff * g_local) / 1000.0  # convert m to km

    # Reference density at 400 km (varies with T_inf)
    # Empirical fit: log10(rho_400) ≈ -11.5 + 0.0013 * T_inf  (kg/m³)
    log10_rho_ref = -11.71 + 0.0012 * T_inf
    rho_ref = 10.0**log10_rho_ref

    ref_alt = 400.0
    rho = rho_ref * math.exp(-(altitude_km - ref_alt) / H_km)

    # Clamp to physical bounds
    rho = max(rho, 1e-18)

    return rho
