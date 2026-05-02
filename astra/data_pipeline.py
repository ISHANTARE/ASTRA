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
from typing import Optional, cast, Any
import numpy as np
import requests
from skyfield.api import Loader
from astra.log import get_logger
from astra.constants import EARTH_EQUATORIAL_RADIUS_KM as _Re_km, G0_STD as _G0_M_S2
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
# Imported from constants.py (as _G0_M_S2 alias for
# backward compatibility with NRLMSISE-00 internal functions below).
# Canonical definition: astra.constants.G0_STD = 9.80665 m/s².
# ---------------------------------------------------------------------------
# NRLMSISE-00 Reference Constants (Atomic masses, gas constant)
# ---------------------------------------------------------------------------
_R_GAS = 8.314462618  # Universal gas constant (J/(K·mol))
_AMU = 1.660539e-27  # Atomic mass unit (kg)
# Molar masses (kg/mol)
_M_HE = 4.0026e-3
_M_O = 15.9994e-3
_M_N2 = 28.0134e-3
_M_O2 = 31.9988e-3
_M_AR = 39.948e-3
_M_H = 1.00784e-3
_M_N = 14.0067e-3
# ---------------------------------------------------------------------------
# Skyfield Loader (DE440 + IERS EOP)
# ---------------------------------------------------------------------------
_skyfield_loader: Optional[Loader] = None
_skyfield_ts = None
_skyfield_eph = None
_skyfield_eph_unavailable = False
# RLock because _ensure_skyfield calls _get_skyfield_loader — must be re-entrant.
_SKYFIELD_INIT_LOCK = threading.RLock()
def _require_skyfield_state(*, require_eph: bool) -> tuple[Any, Any]:
    """Return initialized Skyfield state or raise a typed runtime error."""
    from astra.errors import EphemerisError
    if _skyfield_ts is None:
        raise EphemerisError(
            "Skyfield timescale not initialized after _ensure_skyfield()."
        )
    if require_eph and _skyfield_eph is None:
        raise EphemerisError(
            "Skyfield ephemeris not initialized after _ensure_skyfield()."
        )
    return _skyfield_ts, _skyfield_eph  # type: ignore[no-any-return]
def _get_skyfield_loader(data_dir: Optional[str] = None) -> Loader:
    """Return a singleton Skyfield Loader pointed at the cache directory."""
    global _skyfield_loader
    with _SKYFIELD_INIT_LOCK:
        if _skyfield_loader is None:
            path = data_dir or _DEFAULT_DATA_DIR
            _skyfield_loader = Loader(path)
            logger.info(f"Skyfield data directory: {path}")
    return _skyfield_loader  # type: ignore[no-any-return]
def _ensure_skyfield(data_dir: Optional[str] = None) -> None:
    """Ensure the Skyfield timescale + ephemeris objects are initialised (thread-safe)."""
    global _skyfield_ts, _skyfield_eph, _skyfield_eph_unavailable
    with _SKYFIELD_INIT_LOCK:
        if _skyfield_ts is None:
            load = _get_skyfield_loader(data_dir)
            # Timescale includes IERS finals2000A for UT1-UTC & polar motion.
            _skyfield_ts = load.timescale()
            # DE421 is a compact (~17 MB) JPL ephemeris covering 1900–2050 (AST-7).
            # Missions or simulations beyond ~2050 should plan DE440 or another BSP;
            # Skyfield may fail or rely on extrapolation outside the file span.
            # DE440 (> 100 MB) covers 1549–2650 but is usually overkill for LEO ops.
        if _skyfield_eph is None and not _skyfield_eph_unavailable:
            load = _get_skyfield_loader(data_dir)
            try:
                _skyfield_eph = load("de421.bsp")
                logger.info("Skyfield ephemeris (de421.bsp) and IERS timescale loaded.")
            except OSError as exc:
                from astra import config
                if config.ASTRA_STRICT_MODE:
                    from astra.errors import EphemerisError
                    raise EphemerisError(
                        "[ASTRA STRICT] JPL DE421 ephemeris unavailable. "
                        "Populate the Skyfield cache or set ASTRA_STRICT_MODE=False "
                        "to use analytical Sun/Moon fallbacks."
                    ) from exc
                _skyfield_eph_unavailable = True
                logger.warning(
                    "Skyfield DE421 ephemeris unavailable (%s); continuing with "
                    "Skyfield timescale and analytical Sun/Moon fallbacks.",
                    exc,
                )


def _sun_position_approx(t_jd: float) -> np.ndarray:
    """Approximate geocentric Sun position in GCRS/ECI kilometres."""
    T = (t_jd - 2451545.0) / 36525.0
    mean_anomaly_rad = math.radians((357.5291092 + 35999.0502909 * T) % 360.0)
    center_deg = 1.9146 * math.sin(mean_anomaly_rad) + 0.02 * math.sin(2.0 * mean_anomaly_rad)
    lon_rad = math.radians((280.46646 + 36000.76983 * T + center_deg) % 360.0)
    radius_km = (
        1.00014
        - 0.01671 * math.cos(mean_anomaly_rad)
        - 0.00014 * math.cos(2.0 * mean_anomaly_rad)
    ) * 149597870.7
    obliquity_rad = math.radians(23.439291 - 0.0130042 * T)
    return np.array(
        [
            radius_km * math.cos(lon_rad),
            radius_km * math.cos(obliquity_rad) * math.sin(lon_rad),
            radius_km * math.sin(obliquity_rad) * math.sin(lon_rad),
        ],
        dtype=float,
    )


def _moon_position_approx(t_jd: float) -> np.ndarray:
    """Approximate geocentric Moon position in GCRS/ECI kilometres."""
    T = (t_jd - 2451545.0) / 36525.0
    L0 = (218.3165 + 481267.8813 * T) % 360.0
    M_moon = math.radians((134.9634 + 477198.8676 * T) % 360.0)
    M_sun = math.radians((357.5291 + 35999.0503 * T) % 360.0)
    D = math.radians((297.8502 + 445267.1115 * T) % 360.0)
    F = math.radians((93.2720 + 483202.0175 * T) % 360.0)
    lon_corr = (
        6.289 * math.sin(M_moon)
        - 1.274 * math.sin(2.0 * D - M_moon)
        + 0.658 * math.sin(2.0 * D)
        - 0.214 * math.sin(2.0 * M_moon)
        - 0.186 * math.sin(M_sun)
    )
    lat_rad = math.radians(
        5.128 * math.sin(F)
        + 0.281 * math.sin(M_moon + F)
        - 0.278 * math.sin(F - M_moon)
    )
    radius_km = (
        385000.56
        - 20905.36 * math.cos(M_moon)
        - 3699.11 * math.cos(2.0 * D - M_moon)
        - 2955.97 * math.cos(2.0 * D)
    )
    lon_rad = math.radians(L0 + lon_corr)
    obliquity_rad = math.radians(23.439291 - 0.0130042 * T)
    x_ecl = radius_km * math.cos(lat_rad) * math.cos(lon_rad)
    y_ecl = radius_km * math.cos(lat_rad) * math.sin(lon_rad)
    z_ecl = radius_km * math.sin(lat_rad)
    return np.array(
        [
            x_ecl,
            y_ecl * math.cos(obliquity_rad) - z_ecl * math.sin(obliquity_rad),
            y_ecl * math.sin(obliquity_rad) + z_ecl * math.cos(obliquity_rad),
        ],
        dtype=float,
    )
def get_skyfield_timescale(data_dir: Optional[str] = None) -> Any:
    """Return the managed Skyfield ``Timescale`` (IERS finals2000A), after ensuring cache load.
    Use this instead of ad-hoc ``load.timescale(builtin=True)`` so UT1-UTC and
    conversions share one current-EOP source.
    """
    _ensure_skyfield(data_dir)
    return _skyfield_ts  # type: ignore[no-any-return]
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
    ts, eph = _require_skyfield_state(require_eph=False)
    from astra.jdutil import jd_utc_to_datetime
    dt = cast(datetime, jd_utc_to_datetime(t_jd))
    t = ts.utc(dt)
    if eph is None:
        from astra import config
        if config.ASTRA_STRICT_MODE:
            from astra.errors import EphemerisError
            raise EphemerisError(
                "Skyfield ephemeris unavailable during Sun position query."
            )
        return _sun_position_approx(t_jd)
    earth = eph["earth"]
    sun = eph["sun"]
    # Geometric / astrometric Sun vector from DE421 (light-time ~8.3 min; sub-km
    # difference vs apparent Sun at this force-model fidelity).
    astrometric = earth.at(t).observe(sun)
    pos_au = astrometric.position.au
    # 1 AU = 149 597 870.7 km
    return np.array(pos_au) * 149597870.7  # type: ignore[no-any-return]
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
    ts, _ = _require_skyfield_state(require_eph=False)
    from astra.jdutil import jd_utc_to_datetime
    dt = cast(datetime, jd_utc_to_datetime(t_jd))
    t = ts.utc(dt)
    pos_gcrs_km = sun_position_de(t_jd, data_dir)
    # Rotate GCRS → TEME: R_teme_from_gcrs = TEME.rotation_at(t)
    R_teme_from_gcrs = TEME.rotation_at(t)
    return R_teme_from_gcrs @ pos_gcrs_km  # type: ignore[no-any-return]
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
    ts, eph = _require_skyfield_state(require_eph=False)
    from astra.jdutil import jd_utc_to_datetime
    dt = cast(datetime, jd_utc_to_datetime(t_jd))
    t = ts.utc(dt)
    if eph is None:
        from astra import config
        if config.ASTRA_STRICT_MODE:
            from astra.errors import EphemerisError
            raise EphemerisError(
                "Skyfield ephemeris unavailable during Moon position query."
            )
        return _moon_position_approx(t_jd)
    earth = eph["earth"]
    moon = eph["moon"]
    astrometric = earth.at(t).observe(moon)
    pos_au = astrometric.position.au
    return np.array(pos_au) * 149597870.7  # type: ignore[no-any-return]
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
    ts, _ = _require_skyfield_state(require_eph=False)
    from astra.jdutil import jd_utc_to_datetime
    dt = cast(datetime, jd_utc_to_datetime(t_jd))
    t = ts.utc(dt)
    pos_gcrs_km = moon_position_de(t_jd, data_dir)
    R_teme_from_gcrs = TEME.rotation_at(t)
    return R_teme_from_gcrs @ pos_gcrs_km  # type: ignore[no-any-return]
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
    ts, _ = _require_skyfield_state(require_eph=False)
    t_jd_arr = np.asarray(t_jd_utc)
    dt = cast(datetime, jd_utc_to_datetime(t_jd_arr))
    t = ts.utc(dt)
    return t.dut1  # type: ignore[no-any-return]
# ---------------------------------------------------------------------------
# CelesTrak Space-Weather Cache
# ---------------------------------------------------------------------------
# In-memory cache:  {date_str "YYYY-MM-DD": (F10.7_obs, F10.7_adj, Ap_daily)}
_sw_cache: dict[str, tuple[float, float, float]] = {}
_sw_loaded: bool = False
_sw_last_success: Optional[datetime] = None  # tracks last successful refresh time
_SW_LOCK = threading.RLock()
# Serialize download+parse so two threads that both see _sw_loaded=False
# cannot trigger duplicate downloads.
_SW_DOWNLOAD_LOCK = threading.Lock()
_sw_fetch_thread: Optional[threading.Thread] = None
def _download_space_weather(data_dir: Optional[str] = None) -> str:
    """Download SW-All.csv from CelesTrak and cache locally.
    Includes robust retry headers and post-download integrity checks.
    """
    path = pathlib.Path(data_dir or _DEFAULT_DATA_DIR)
    path.mkdir(parents=True, exist_ok=True)
    local_file = path / "SW-All.csv"
    logger.info(f"Downloading CelesTrak SW-All.csv → {local_file}")
    # DEF-013: Add robust retries for network unreliability
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
    session = requests.Session()
    retries = Retry(
        total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504]
    )
    session.mount("https://", HTTPAdapter(max_retries=retries))
    try:
        resp = session.get(_CELESTRAK_SW_URL, timeout=30, verify=True)
        resp.raise_for_status()
        # Increase payload limit to 10 MB. 
        # The historical space weather record grows by ~80 KB/year; a 2 MB cap
        # would have caused a silent failure in approximately the mid-2030s.
        if len(resp.content) > 10 * 1024 * 1024:
            raise ValueError(
                "Payload length exceeds 10MB limit. Rejecting as potentially malicious."
            )
    except (requests.RequestException, ValueError) as e:
        from astra import config
        if getattr(config, "ASTRA_STRICT_MODE", False):
            raise ValueError(f"[ASTRA STRICT] Space weather fetch failed: {e}") from e
        logger.warning(
            f"Space weather fetch failed: {e}. Falling back to default data if cache empty."
        )
        raise
    text = resp.text
    # DEF-019: Proxy sanity check to avoid caching HTML login portals
    if not text.startswith("DATE") and not text.startswith("#"):
        from astra import config
        msg = f"Invalid space weather payload from CelesTrak (starts with: {text[:20]!r}). Likely a proxy or firewall."
        if getattr(config, "ASTRA_STRICT_MODE", False):
            raise ValueError(f"[ASTRA STRICT] {msg}")
        logger.warning(msg)
        raise ValueError("Invalid CSV payload")
    local_file.write_text(text, encoding="utf-8")
    return text  # type: ignore[no-any-return]
def _parse_sw_csv(text: str) -> None:
    """Parse the CelesTrak CSV into the in-memory cache.
    DATA-01 Fix: Uses header-name based column lookup instead of hardcoded
    positional indices, making the parser robust to CelesTrak format changes.
    """
    global _sw_loaded
    new_cache = {}
    reader = csv.reader(io.StringIO(text))
    header: list[str] | None = None
    # Header-to-index mapping (DATA-01: robust against column reordering)
    idx_date: int = 0
    idx_f107obs: int = 24  # fallback defaults matching 2024 CelesTrak layout
    idx_f107adj: int = 25
    idx_ap_avg: int = 20
    for row in reader:
        if not row or row[0].startswith("#"):
            continue
        if header is None:
            header = [h.strip() for h in row]
            # DATA-01: resolve column positions by name, not hardcoded index.
            # Accepted column name variants to handle minor format changes.
            _f107obs_names = {"F10.7_OBS", "F107OBS", "F10.7OBS", "OBSERVED_F10.7"}
            _f107adj_names = {"F10.7_ADJ", "F107ADJ", "F10.7ADJ", "ADJUSTED_F10.7"}
            _ap_names = {"AP_AVG", "AP-AVG", "APAVG", "AP_DAILY"}
            _date_names = {"DATE", "DATE_UTC"}
            for i, col in enumerate(header):
                cu = col.upper().replace(" ", "_")
                if cu in _date_names:
                    idx_date = i
                elif cu in _f107obs_names:
                    idx_f107obs = i
                elif cu in _f107adj_names:
                    idx_f107adj = i
                elif cu in _ap_names:
                    idx_ap_avg = i
            logger.debug(
                "CelesTrak SW CSV columns resolved: DATE=%d, F10.7_OBS=%d, "
                "F10.7_ADJ=%d, Ap_AVG=%d",
                idx_date,
                idx_f107obs,
                idx_f107adj,
                idx_ap_avg,
            )
            continue
        try:
            date_str = row[idx_date].strip()
            if not date_str:
                continue
            # Safely parse; some future-predicted rows may have blanks.
            f107_obs = (
                float(row[idx_f107obs])
                if len(row) > idx_f107obs and row[idx_f107obs].strip()
                else 150.0
            )
            # AUDIT-SEC-02: Prevent malicious manipulation mapping F10.7 into structural failures
            if f107_obs < 50.0 and f107_obs > 0.0:
                logger.debug(
                    f"Row {date_str} bounded with historically suspect F10.7: {f107_obs}."
                )
            elif f107_obs <= 0.0:
                logger.warning(f"Row {date_str} skipped due to non-physical F10.7 <= 0: {f107_obs}")
                continue
            f107_adj = (
                float(row[idx_f107adj])
                if len(row) > idx_f107adj and row[idx_f107adj].strip()
                else f107_obs
            )
            ap_daily = (
                float(row[idx_ap_avg])
                if len(row) > idx_ap_avg and row[idx_ap_avg].strip()
                else 15.0
            )
            new_cache[date_str] = (f107_obs, f107_adj, ap_daily)
        except (ValueError, IndexError):
            continue  # skip malformed rows
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
        logger.error(f"Background Space-Weather fetch failed with unexpected error: {exc}", exc_info=True)
    finally:
        with _SW_LOCK:
            _sw_fetch_thread = None
def load_space_weather(
    data_dir: Optional[str] = None, force_download: bool = False
) -> None:
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
    # Slow path — serialise so only one thread downloads + parses.  # type: ignore[no-any-return]
    with _SW_DOWNLOAD_LOCK:
        # Re-check inside the download lock (DCL "double-check").
        with _SW_LOCK:
            if _sw_loaded and not force_download:
                return  # Another thread finished while we waited.  # type: ignore[no-any-return]
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
                    logger.info(
                        "Space-weather cache stale. Spawning background refresh daemon..."
                    )
                    _sw_fetch_thread = threading.Thread(
                        target=_background_sw_fetch,
                        args=(data_dir,),
                        daemon=True,
                    )
                    _sw_fetch_thread.start()
def get_space_weather(
    t_jd: float, data_dir: Optional[str] = None
) -> tuple[float, float, float]:
    """Retrieve F10.7 solar flux and Ap index for a given Julian Date.
    Priority hierarchy:
    1. **Spacebook:** Defaults to Spacebook (higher precision, COMSPOC-verified)
       if ``ASTRA_SPACEBOOK_ENABLED=true``.
    2. **CelesTrak:** Falls back to legacy CelesTrak CSV if Spacebook fails or
       is disabled.
    3. **Synthetic Defaults:** Returns (150, 150, 15) if no data is found and
       ``ASTRA_STRICT_MODE=False``.
    Args:
        t_jd: Julian Date.
        data_dir: Override data directory.
    Returns:
        Tuple (F10.7_obs, F10.7_adj, Ap_daily).
    """
    # ── 1. Spacebook (Primary) ──
    from astra import spacebook
    from astra import config as _astra_cfg
    from astra.errors import SpacebookError
    if _astra_cfg.SPACEBOOK_ENABLED:
        try:
            # Spacebook handles its own background refreshing
            return spacebook.get_space_weather_sb(t_jd)  # type: ignore[no-any-return]
        except ImportError:
            raise
        except (
            SpacebookError,
            ValueError,
            TypeError,
            ArithmeticError,
            requests.RequestException,
        ) as exc:
            logger.warning(
                "Spacebook Space Weather lookup failed: %s. Falling back to CelesTrak...",
                exc,
            )
    # ── 2. CelesTrak (Fallback) ──
    try:
        load_space_weather(data_dir)
    except (ValueError, requests.RequestException) as exc:
        from astra import config
        if getattr(config, "ASTRA_STRICT_MODE", False):
            raise ValueError(
                f"[ASTRA STRICT] Required CelesTrak space-weather fetch failed: {exc}"
            )
        logger.warning(
            "Falling back to synthetic space weather parameters (150/150/15)."
        )
    # Stale cache: background refresh if > 48 h since last successful load
    global _sw_fetch_thread
    with _SW_LOCK:
        if _sw_last_success is not None:
            age_h = (
                datetime.now(timezone.utc) - _sw_last_success
            ).total_seconds() / 3600
            if age_h > 24.0 and _sw_fetch_thread is None:
                logger.warning(
                    "Space-weather cache >24 hours old. Triggering background refresh."
                )
                _sw_fetch_thread = threading.Thread(
                    target=_background_sw_fetch, args=(data_dir,), daemon=True
                )
                _sw_fetch_thread.start()
    # JD → calendar date via integer-split day/fraction (stable vs float drift)
    from astra.jdutil import jd_utc_to_datetime
    dt = cast(datetime, jd_utc_to_datetime(t_jd))
    date_str = f"{dt.year:04d}-{dt.month:02d}-{dt.day:02d}"
    with _SW_LOCK:
        if date_str in _sw_cache:
            return _sw_cache[date_str]  # type: ignore[no-any-return]
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
    return (150.0, 150.0, 15.0)  # type: ignore[no-any-return]
# ---------------------------------------------------------------------------
# High-Fidelity NRLMSISE-00 Atmosphere (Numba Optimized)
# ---------------------------------------------------------------------------
@np.vectorize
def _msis_bates_temperature(
    z_km: float, z_lb_km: float, T_lb: float, T_inf: float, s: float
) -> float:
    """Compute temperature at altitude z using the Bates exospheric profile.
    Args:
        z_km: Target altitude [km].
        z_lb_km: Lower boundary altitude [km] (typically 120 km).
        T_lb: Temperature at lower boundary [K].
        T_inf: Exospheric temperature [K].
        s: Slope parameter [1/km].
    """
    if z_km <= z_lb_km:
        return T_lb  # type: ignore[no-any-return]
    # Use _Re_km from constants.EARTH_EQUATORIAL_RADIUS_KM
    xi = (z_km - z_lb_km) * (_Re_km + z_lb_km) / (_Re_km + z_km)
    return T_inf - (T_inf - T_lb) * math.exp(-s * xi)  # type: ignore[no-any-return]
def nrlmsise00_density(
    altitude_km: float,
    f107_obs: float,
    f107_adj: float,
    ap_daily: float,
) -> float:
    """Compute high-fidelity atmospheric density using NRLMSISE-00 physics.
    This function delegates to the canonical Numba implementation in propagator.py
    to guarantee identical physics across the codebase (Fixes 1.5).
    """
    from astra.propagator import _nrlmsise00_density_njit
    return float(_nrlmsise00_density_njit(altitude_km, f107_obs, f107_adj, ap_daily))
def atmospheric_density_empirical(
    altitude_km: float,
    f107_obs: float,
    f107_adj: float,
    ap_daily: float,
) -> float:
    """Compute atmospheric density using an empirical model.
    In v3.6.0, this function defaults to the High-Fidelity NRLMSISE-00
    implementation for improved accuracy across all solar regimes.
    """
    return nrlmsise00_density(altitude_km, f107_obs, f107_adj, ap_daily)  # type: ignore[no-any-return]
