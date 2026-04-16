# astra/spacebook.py
"""ASTRA Core — COMSPOC Spacebook Data Integration Module.

Provides free, unauthenticated access to COMSPOC Corporation's Spacebook
platform (https://spacebook.com), which exposes a suite of high-quality
Space Situational Awareness (SSA) data products via public HTTP APIs.

**No account or credentials are required.** All endpoints are freely
accessible via standard HTTP GET requests.

Available Data Products
-----------------------
1. **Standard TLE Catalog** — full public-catalog TLEs, refreshed frequently.
2. **Historical TLE Catalog** — archived TLEs for a specific past date.
3. **XP-TLE Catalog** — Extended-Precision TLEs, numerically refined by
   COMSPOC for higher positional accuracy.
4. **Space Weather** (F10.7, Kp, Ap) — CSSI/CelesTrak-compatible fixed-width
   format, used to drive ASTRA's empirical atmospheric drag model.
5. **Earth Orientation Parameters (EOP)** — polar motion (xp, yp) and
   UT1-UTC, used for precise ECI↔ECEF coordinate frame transformations.
6. **Satellite Catalog** — JSON/CSV catalog mapping object names, NORAD IDs,
   and COMSPOC GUIDs, with optional extended physical metadata.
7. **Synthetic Covariance Ephemeris** — per-satellite STK-format ephemeris
   containing COMSPOC's numerically propagated state vectors in TEME frame.

Environment Variables
---------------------
``ASTRA_SPACEBOOK_ENABLED`` (default: ``"true"``)
    Set to ``"false"`` to fully disable all Spacebook calls. When disabled,
    ``astra.data_pipeline.get_space_weather()`` will fall back to CelesTrak,
    and per-object GUID-based endpoints will raise ``SpacebookError``.

``ASTRA_DATA_DIR`` (default: ``~/.astra/data``)
    Root directory for local data caches. Spacebook files are stored in a
    ``spacebook/`` subdirectory underneath this root.

Cache TTLs
----------
- Space Weather:  6 hours (recent), 7 days (full)
- EOP:            24 hours (recent), 7 days (full)
- TLE/XP-TLE:     6 hours
- Satellite Catalog (JSON):  12 hours (background refresh)
- Historical TLE:  permanent (immutable past data)
- Synthetic Covariance:  1 hour (changes as new tracking data arrives)

References
----------
- Spacebook User Guide: https://spacebook.com/userguide
- COMSPOC API Section 4B: https://spacebook.com/userguide#section4B
- Schema notes: docs/spacebook_schema_notes.md (verified 2026-04-09)
"""

from __future__ import annotations
from typing import Any

import os
import pathlib
import threading
from datetime import datetime, timedelta, timezone
from typing import Optional

import requests

from astra.errors import SpacebookError, SpacebookLookupError
from astra.log import get_logger
from astra.models import SatelliteTLE
from astra.tle import load_tle_catalog

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Configuration & Feature Flag
# ---------------------------------------------------------------------------

#: Set ASTRA_SPACEBOOK_ENABLED=false to disable all Spacebook network calls.
SPACEBOOK_ENABLED: bool = (
    os.environ.get("ASTRA_SPACEBOOK_ENABLED", "true").strip().lower() != "false"
)

# Root cache directory — mirrors data_pipeline.py's convention.
_DEFAULT_DATA_DIR = os.environ.get(
    "ASTRA_DATA_DIR",
    str(pathlib.Path.home() / ".astra" / "data"),
)

# ---------------------------------------------------------------------------
# API Endpoint Constants (verified 2026-04-09 against live Spacebook APIs)
# ---------------------------------------------------------------------------

_SB_BASE = "https://spacebook.com/api"

# ── Catalog ───────────────────────────────────────────────────────────────
_SB_TLE_URL = f"{_SB_BASE}/entity/tle"
_SB_HISTORICAL_TLE = f"{_SB_BASE}/entity/tle/{{date}}"  # {date} = YYYY-MM-DD
_SB_XPTLE_URL = f"{_SB_BASE}/entity/xp-tle"
_SB_SATCAT_JSON_URL = f"{_SB_BASE}/entity/satcat"
_SB_SATCAT_DET_URL = f"{_SB_BASE}/entity/satcat/details"
_SB_SATCAT_CSV_URL = f"{_SB_BASE}/entity/satcat/csv"

# ── Environmental ─────────────────────────────────────────────────────────
_SB_SW_RECENT_URL = f"{_SB_BASE}/spaceweather/recent"
_SB_SW_FULL_URL = f"{_SB_BASE}/spaceweather/full"
_SB_EOP_RECENT_URL = f"{_SB_BASE}/eop/recent"
_SB_EOP_FULL_URL = f"{_SB_BASE}/eop/full"

# ── Per-Object (GUID-based) ───────────────────────────────────────────────
# Resolved GUID required. Use get_norad_guid() to obtain.
_SB_SYNTH_COV_URL = f"{_SB_BASE}/entity/synthetic-covariance/{{guid}}"

_HEADERS = {
    "User-Agent": (
        "ASTRA-Core/spacebook-client "
        "(https://github.com/ISHANTARE/ASTRA; astrodynamics data integration)"
    ),
    "Accept": "application/json, text/plain, */*",
}

_REQUEST_TIMEOUT = 30  # seconds

# ---------------------------------------------------------------------------
# In-Memory Caches (thread-safe)
# ---------------------------------------------------------------------------

# Space Weather cache: date_str "YYYY-MM-DD" → (f107_obs, f107_adj, ap_daily)
_sw_cache: dict[str, tuple[float, float, float]] = {}
_sw_loaded: bool = False
_sw_last_success: Optional[datetime] = None
_SW_LOCK = threading.RLock()
# CONC-01 Fix: Track the background SW refresh thread so rapid calls to
# get_space_weather_sb() when cache is stale don't spawn unbounded threads.
# Pattern mirrors data_pipeline._sw_fetch_thread.
_sw_refresh_thread: Optional[threading.Thread] = None
# CONC-02 Fix: Separate non-reentrant lock serialises the initial download so
# two threads reading _sw_loaded=False concurrently don't both call _load_sw().
_SW_DOWNLOAD_LOCK = threading.Lock()

# EOP cache: MJD (int) → (xp_arcsec, yp_arcsec, dut1_s)
_eop_cache: dict[int, tuple[float, float, float]] = {}
_eop_loaded: bool = False
_EOP_LOCK = threading.RLock()
# AUDIT-C-S02 Fix: Add a dedicated download-serialisation lock (non-reentrant)
# so two threads that both observe _eop_loaded=False do not trigger two
# simultaneous network downloads.  Mirrors the SW and GUID DCL patterns.
_EOP_DOWNLOAD_LOCK = threading.Lock()

# Satellite catalog: norad_id (int) → comspoc_guid (str UUID)
_guid_map: dict[int, str] = {}
_guid_loaded: bool = False
_guid_last_success: Optional[datetime] = None
_GUID_LOCK = threading.RLock()
_GUID_DOWNLOAD_LOCK = threading.Lock()
_guid_refresh_thread: Optional[threading.Thread] = None


# ---------------------------------------------------------------------------
# Core HTTP Helper
# ---------------------------------------------------------------------------


def _sb_get(url: str, timeout: int = _REQUEST_TIMEOUT) -> requests.Response:
    """Perform an authenticated-free HTTP GET against a Spacebook endpoint.

    Args:
        url: Full Spacebook API URL.
        timeout: Request timeout in seconds.

    Returns:
        A ``requests.Response`` with status 200.

    Raises:
        SpacebookError: On network failure, timeout, or non-200 HTTP status.
    """
    try:
        resp = requests.get(url, headers=_HEADERS, timeout=timeout)
    except requests.Timeout as exc:
        raise SpacebookError(
            f"Spacebook request timed out after {timeout}s.",
            url=url,
        ) from exc
    except requests.RequestException as exc:
        raise SpacebookError(
            f"Spacebook network error: {exc}",
            url=url,
        ) from exc

    if resp.status_code != 200:
        raise SpacebookError(
            f"Spacebook returned HTTP {resp.status_code} for {url}",
            url=url,
            status_code=resp.status_code,
        )
    return resp


def _sb_cache_path(filename: str) -> pathlib.Path:
    """Return the full path to a file in the Spacebook cache directory.

    The directory is created on first use.
    """
    path = pathlib.Path(_DEFAULT_DATA_DIR) / "spacebook"
    path.mkdir(parents=True, exist_ok=True)
    return path / filename


def _cache_age_hours(path: pathlib.Path) -> float:
    """Return the age of a cached file in hours. Returns inf if absent."""
    if not path.exists():
        return float("inf")
    mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    return (datetime.now(timezone.utc) - mtime).total_seconds() / 3600.0


# ---------------------------------------------------------------------------
# Availability Probe
# ---------------------------------------------------------------------------


def is_available(timeout: int = 4) -> bool:
    """Return True if the Spacebook API is reachable.

    Performs a lightweight HEAD-style GET against the Space Weather recent
    endpoint with a short timeout. Safe to call at startup to decide whether
    to prefer Spacebook or fall back to CelesTrak.

    Args:
        timeout: Network timeout in seconds (default: 4).

    Returns:
        True if Spacebook responds with HTTP 200, False otherwise.
    """
    if not SPACEBOOK_ENABLED:
        return False
    try:
        resp = requests.get(
            _SB_SW_RECENT_URL,
            headers=_HEADERS,
            timeout=timeout,
            stream=True,  # avoid downloading the full body just for a probe
        )
        resp.close()
        return resp.status_code == 200
    except Exception:
        return False


# ═══════════════════════════════════════════════════════════════════════════
# SPACE WEATHER
# ═══════════════════════════════════════════════════════════════════════════


def _parse_sw_text(text: str) -> dict[str, tuple[float, float, float]]:
    """Parse Spacebook / CSSI fixed-width space-weather text.

    The format is the standard CelesTrak CSSISpaceWeather fixed-width text
    (FORMAT: I4,I3,I3,I5,I3,...).

    DATA-02 Fix: Field positions are now named constants rather than bare
    integer literals.  The CSSI fixed-width format is defined by specification
    (not by a CSV header), so positional parsing is correct; named constants
    make future format changes easy to locate and update in one place.

    CSSI field positions (0-indexed, space-split tokens):
      _SW_FW_YEAR    = 0   year
      _SW_FW_MONTH   = 1   month
      _SW_FW_DAY     = 2   day
      _SW_FW_AP_AVG  = 20  daily-average Ap index
      _SW_FW_F107ADJ = 23  81-day centred F10.7 (adjusted)
      _SW_FW_F107OBS = 28  observed F10.7

    Observed, Daily-Predicted, and Monthly-Predicted blocks are all parsed
    and merged into one dict keyed by "YYYY-MM-DD".  Predicted values
    overwrite observed values for the same date only if the date is in the
    future (so observed data always wins).

    Returns:
        Dict mapping ``"YYYY-MM-DD"`` → ``(f107_obs, f107_adj, ap_daily)``.
    """
    # DATA-02: Named field positions per CSSI CSSISpaceWeather fixed-width spec.
    _SW_FW_YEAR = 0
    _SW_FW_MONTH = 1
    _SW_FW_DAY = 2
    _SW_FW_AP_AVG = 20  # Daily-average (Avg) Ap geomagnetic index
    _SW_FW_F107ADJ = 23  # 81-day centred F10.7 flux (adjusted to 1 AU)
    _SW_FW_F107OBS = 28  # Observed (daily) F10.7 flux
    _SW_FW_MIN_FIELDS = _SW_FW_F107OBS + 1  # 29 — minimum tokens required

    result: dict[str, tuple[float, float, float]] = {}

    inside_block = False
    today = datetime.now(timezone.utc).date()

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("BEGIN"):
            inside_block = True
            continue
        if line.startswith("END") or line.startswith("NUM_"):
            inside_block = False
            continue
        if not inside_block:
            continue

        fields = line.split()
        if len(fields) < _SW_FW_MIN_FIELDS:
            continue
        try:
            year = int(fields[_SW_FW_YEAR])
            month = int(fields[_SW_FW_MONTH])
            day = int(fields[_SW_FW_DAY])
            ap = float(fields[_SW_FW_AP_AVG])
            f107_adj = float(fields[_SW_FW_F107ADJ])
            f107_obs = float(fields[_SW_FW_F107OBS])
            date_str = f"{year:04d}-{month:02d}-{day:02d}"

            # Don't overwrite an already-stored observed value with a
            # predicted value for the same past date.
            record_date = datetime(year, month, day).date()
            if date_str in result and record_date < today:
                continue

            result[date_str] = (f107_obs, f107_adj, ap)
        except (ValueError, IndexError):
            continue

    return result


def _load_sw(force_full: bool = False) -> None:
    """Download and parse Spacebook Space Weather into the in-memory cache.

    Uses the 'recent' endpoint (fast, ~350 KB) unless `force_full` is True
    or the recent cache file is older than 30 days.
    """
    global _sw_loaded, _sw_last_success

    # Decide which file to use
    recent_path = _sb_cache_path("sw_recent.txt")
    full_path = _sb_cache_path("sw_full.txt")

    age_recent = _cache_age_hours(recent_path)
    age_full = _cache_age_hours(full_path)

    use_full = force_full or (age_recent > 24 * 30)  # recent is >30 days old

    if use_full:
        if age_full <= 24 * 7 and full_path.exists():
            text = full_path.read_text(encoding="utf-8", errors="replace")
            logger.debug("Spacebook SW: using cached full file.")
        else:
            logger.info(
                "Spacebook SW: downloading full history from %s", _SB_SW_FULL_URL
            )
            resp = _sb_get(_SB_SW_FULL_URL)
            text = resp.text
            full_path.write_text(text, encoding="utf-8")
    else:
        if age_recent <= 6 and recent_path.exists():
            text = recent_path.read_text(encoding="utf-8", errors="replace")
            logger.debug(
                "Spacebook SW: using cached recent file (%.1f h old).", age_recent
            )
        else:
            logger.info(
                "Spacebook SW: downloading recent data from %s", _SB_SW_RECENT_URL
            )
            resp = _sb_get(_SB_SW_RECENT_URL)
            text = resp.text
            recent_path.write_text(text, encoding="utf-8")
            # Also refresh full if very stale
            if age_full > 24 * 7:
                try:
                    logger.info("Spacebook SW: also refreshing full history.")
                    resp_full = _sb_get(_SB_SW_FULL_URL)
                    full_path.write_text(resp_full.text, encoding="utf-8")
                    text = resp_full.text  # prefer full for in-memory
                except SpacebookError as exc:
                    logger.warning("Could not download SW full: %s", exc)

    new_cache = _parse_sw_text(text)
    with _SW_LOCK:
        _sw_cache.clear()
        _sw_cache.update(new_cache)
        _sw_loaded = True
        _sw_last_success = datetime.now(timezone.utc)

    logger.info("Spacebook SW: cache loaded — %d daily records.", len(_sw_cache))


def get_space_weather_sb(t_jd: float) -> tuple[float, float, float]:
    """Retrieve F10.7 and Ap indices from Spacebook for a given Julian Date.

    Downloads and caches the Space Weather data on first call. Subsequent
    calls resolve directly from the in-memory cache (O(1)). The cache is
    automatically refreshed when data is more than 6 hours old.

    Args:
        t_jd: Julian Date (UTC scale).

    Returns:
        Tuple ``(f107_obs, f107_adj, ap_daily)`` for the calendar date
        corresponding to ``t_jd``.

    Raises:
        SpacebookError: If `SPACEBOOK_ENABLED=false` or the download fails
                        *and* no local cache file exists as a fallback.
    """
    if not SPACEBOOK_ENABLED:
        raise SpacebookError("Spacebook is disabled (ASTRA_SPACEBOOK_ENABLED=false).")

    global _sw_refresh_thread

    with _SW_LOCK:
        loaded = _sw_loaded

    if not loaded:
        # CONC-02 Fix: Double-checked locking — acquire download lock, then
        # re-read _sw_loaded under _SW_LOCK to prevent duplicate downloads
        # when multiple threads both see _sw_loaded=False concurrently.
        with _SW_DOWNLOAD_LOCK:
            with _SW_LOCK:
                loaded = _sw_loaded
            if not loaded:
                _load_sw()

    # Background refresh if cache is getting stale (but don't block)
    with _SW_LOCK:
        if _sw_last_success is not None:
            age_h = (
                datetime.now(timezone.utc) - _sw_last_success
            ).total_seconds() / 3600.0
            # CONC-01 Fix: Only spawn a new refresh thread if one isn't already running.
            # Without this guard, millions of rapid calls while cache is stale
            # (age_h > 6) could each spawn a new daemon thread, exhausting memory.
            if age_h > 6 and _sw_refresh_thread is None:

                def _refresh() -> None:
                    global _sw_refresh_thread
                    try:
                        _load_sw()
                    except SpacebookError as exc:
                        logger.warning("Background SW refresh failed: %s", exc)
                    finally:
                        with _SW_LOCK:
                            _sw_refresh_thread = None

                t = threading.Thread(target=_refresh, daemon=True)
                _sw_refresh_thread = t
                t.start()

    # JD → calendar date (stable at midnight boundaries)
    days_offset = t_jd - 2451545.0
    epoch = datetime(2000, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    dt = epoch + timedelta(days=days_offset)
    date_str = dt.strftime("%Y-%m-%d")

    with _SW_LOCK:
        if date_str in _sw_cache:
            return _sw_cache[date_str]

    raise SpacebookError(f"Spacebook SW: no data for {date_str}.")


# ═══════════════════════════════════════════════════════════════════════════
# EARTH ORIENTATION PARAMETERS (EOP)
# ═══════════════════════════════════════════════════════════════════════════


def _parse_eop_text(text: str) -> dict[int, tuple[float, float, float]]:
    """Parse Spacebook IERS-format EOP file.

    Format per header:
    ``FORMAT(I4,I3,I3,I6,2F10.6,2F11.7,4F10.6,I4)``
    ``y4  mm  dd  MJD  xp  yp  UT1-UTC  LOD  dPsi  dEps  dX  dY  DAT``

    Returns:
        Dict mapping ``MJD (int)`` → ``(xp_arcsec, yp_arcsec, dut1_s)``.
    """
    result: dict[int, tuple[float, float, float]] = {}
    inside_block = False

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("BEGIN"):
            inside_block = True
            continue
        if line.startswith("END") or line.startswith("NUM_"):
            inside_block = False
            continue
        if not inside_block:
            continue

        fields = line.split()
        if len(fields) < 7:
            continue
        try:
            mjd = int(fields[3])
            xp = float(fields[4])  # arcseconds
            yp = float(fields[5])  # arcseconds
            dut1 = float(fields[6])  # seconds (UT1-UTC)
            result[mjd] = (xp, yp, dut1)
        except (ValueError, IndexError):
            continue

    return result


def _load_eop() -> None:
    """Download and parse Spacebook EOP data into the in-memory cache."""
    global _eop_loaded

    recent_path = _sb_cache_path("eop_recent.txt")
    full_path = _sb_cache_path("eop_full.txt")

    age_recent = _cache_age_hours(recent_path)
    age_full = _cache_age_hours(full_path)

    # Use recent file (24h TTL) if fresh, otherwise refresh it
    if age_recent <= 24 and recent_path.exists():
        text = recent_path.read_text(encoding="utf-8", errors="replace")
        logger.debug("Spacebook EOP: using cached recent file.")
    else:
        logger.info(
            "Spacebook EOP: downloading recent data from %s", _SB_EOP_RECENT_URL
        )
        resp = _sb_get(_SB_EOP_RECENT_URL)
        text = resp.text
        recent_path.write_text(text, encoding="utf-8")

    new_cache = _parse_eop_text(text)

    # Supplement with full history if the full file is reasonably fresh
    if age_full <= 24 * 7 and full_path.exists():
        full_text = full_path.read_text(encoding="utf-8", errors="replace")
        full_cache = _parse_eop_text(full_text)
        # Merge: recent values (higher authority) take precedence
        merged = {**full_cache, **new_cache}
        new_cache = merged
    elif age_full > 24 * 7:
        try:
            logger.info(
                "Spacebook EOP: downloading full history from %s", _SB_EOP_FULL_URL
            )
            resp_full = _sb_get(_SB_EOP_FULL_URL)
            full_path.write_text(resp_full.text, encoding="utf-8")
            full_cache = _parse_eop_text(resp_full.text)
            new_cache = {**full_cache, **new_cache}
        except SpacebookError as exc:
            logger.warning("Could not download EOP full history: %s", exc)

    with _EOP_LOCK:
        _eop_cache.clear()
        _eop_cache.update(new_cache)
        _eop_loaded = True

    logger.info("Spacebook EOP: cache loaded — %d daily records.", len(_eop_cache))


def get_eop_sb(t_jd: float) -> tuple[float, float, float]:
    """Return Earth Orientation Parameters from Spacebook for a given Julian Date.

    Args:
        t_jd: Julian Date (UTC scale).

    Returns:
        Tuple ``(xp_arcsec, yp_arcsec, dut1_s)`` — polar motion X/Y in
        arcseconds, and UT1-UTC offset in seconds.  If no data exists for the
        exact MJD, linear interpolation between the two nearest records is used.
        Returns ``(0.0, 0.0, 0.0)`` if the cache is completely empty.

    Raises:
        SpacebookError: If Spacebook is disabled.
    """
    if not SPACEBOOK_ENABLED:
        raise SpacebookError("Spacebook is disabled (ASTRA_SPACEBOOK_ENABLED=false).")

    with _EOP_LOCK:
        loaded = _eop_loaded

    if not loaded:
        # AUDIT-C-S02 Fix: Double-checked locking — serialise the initial
        # download so two threads that both see _eop_loaded=False concurrently
        # don't issue two network requests.  Same pattern as SW and GUID caches.
        with _EOP_DOWNLOAD_LOCK:
            with _EOP_LOCK:
                loaded = _eop_loaded
            if not loaded:
                _load_eop()

    # JD → MJD
    mjd = int(t_jd - 2400000.5)

    with _EOP_LOCK:
        if mjd in _eop_cache:
            return _eop_cache[mjd]

        # Linear interpolation between two nearest neighbours
        keys = sorted(_eop_cache.keys())
        if not keys:
            logger.warning(
                "Spacebook EOP: cache empty — returning zero EOP correction."
            )
            return (0.0, 0.0, 0.0)

        # Binary-search for bracketing indices
        lo, hi = 0, len(keys) - 1
        while lo < hi - 1:
            mid = (lo + hi) // 2
            if keys[mid] <= mjd:
                lo = mid
            else:
                hi = mid

        if lo == hi or keys[hi] == keys[lo]:
            return _eop_cache[keys[lo]]

        # Interpolate
        frac = (mjd - keys[lo]) / (keys[hi] - keys[lo])
        xp0, yp0, d0 = _eop_cache[keys[lo]]
        xp1, yp1, d1 = _eop_cache[keys[hi]]
        return (
            xp0 + frac * (xp1 - xp0),
            yp0 + frac * (yp1 - yp0),
            d0 + frac * (d1 - d0),
        )


# ═══════════════════════════════════════════════════════════════════════════
# SATELLITE CATALOG & GUID RESOLUTION
# ═══════════════════════════════════════════════════════════════════════════


def _load_satcat_guid_map() -> None:
    """Download the Spacebook satellite catalog and build the NORAD→GUID map.

    Uses a 12-hour TTL local cache.  A background thread handles stale
    refreshes without blocking the caller.

    The catalog JSON schema (verified 2026-04-09):
    ``[{"id": "<UUID>", "noradId": <int_no_leading_zeros>, ...}, ...]``
    """
    global _guid_loaded, _guid_last_success, _guid_refresh_thread

    cache_path = _sb_cache_path("satcat.json")
    age_h = _cache_age_hours(cache_path)

    need_download = not cache_path.exists() or age_h > 12
    stale = cache_path.exists() and age_h > 12

    if not need_download and cache_path.exists():
        # Load from local cache
        import json

        try:
            raw = cache_path.read_text(encoding="utf-8", errors="replace")
            data = json.loads(raw)
        except Exception as exc:
            logger.warning(
                "Spacebook satcat: cache corrupted (%s); re-downloading.", exc
            )
            need_download = True

    if need_download:
        logger.info("Spacebook satcat: downloading from %s", _SB_SATCAT_JSON_URL)
        try:
            resp = _sb_get(_SB_SATCAT_JSON_URL, timeout=60)
        except SpacebookError as exc:
            # If we have any local file, use it rather than failing cold
            if cache_path.exists():
                logger.warning(
                    "Spacebook satcat: download failed (%s); using stale cache.", exc
                )
                import json

                raw = cache_path.read_text(encoding="utf-8", errors="replace")
                data = json.loads(raw)
            else:
                raise
        else:
            import json

            cache_path.write_text(resp.text, encoding="utf-8")
            data = json.loads(resp.text)

    # Build the map
    new_map: dict[int, str] = {}
    if isinstance(data, list):
        for rec in data:
            if not isinstance(rec, dict):
                continue
            nid = rec.get("noradId")
            guid = rec.get("id")
            if nid is not None and guid:
                try:
                    new_map[int(nid)] = str(guid)
                except (ValueError, TypeError):
                    continue

    with _GUID_LOCK:
        _guid_map.clear()
        _guid_map.update(new_map)
        _guid_loaded = True
        _guid_last_success = datetime.now(timezone.utc)

    logger.info("Spacebook satcat: GUID map loaded — %d objects.", len(_guid_map))

    # AUDIT-C-05 Fix: The old code recursively called _load_satcat_guid_map()
    # from the background thread, but because the parent call already wrote the
    # fresh cache file, the recursive call found age_h ≈ 0 and short-circuited
    # with a local read — i.e. it did NOT perform a network refresh.  This
    # background thread is supposed to asynchronously download a FRESH copy.
    # Fix: extract a dedicated _download_satcat_fresh() helper that forces a
    # network download and merges the result into the existing in-memory map,
    # then clear the sentinel in a guaranteed finally block.
    if stale:

        def _bg_satcat_refresh() -> None:
            global _guid_refresh_thread
            try:
                logger.info("Spacebook satcat: background refresh starting download.")
                resp = _sb_get(_SB_SATCAT_JSON_URL, timeout=60)
                import json as _json

                cache_path.write_text(resp.text, encoding="utf-8")
                data_fresh = _json.loads(resp.text)
                new_map: dict[int, str] = {}
                if isinstance(data_fresh, list):
                    for rec in data_fresh:
                        if not isinstance(rec, dict):
                            continue
                        nid = rec.get("noradId")
                        guid = rec.get("id")
                        if nid is not None and guid:
                            try:
                                new_map[int(nid)] = str(guid)
                            except (ValueError, TypeError):
                                continue
                with _GUID_LOCK:
                    _guid_map.clear()
                    _guid_map.update(new_map)
                    _guid_last_success = datetime.now(timezone.utc)
                logger.info(
                    "Spacebook satcat: background refresh complete — %d objects.",
                    len(new_map),
                )
            except Exception as exc:
                logger.warning("Background satcat refresh failed: %s", exc)
            finally:
                # AUDIT-C-05: Always clear the sentinel so future stale
                # checks can schedule a new refresh thread.
                with _GUID_LOCK:
                    _guid_refresh_thread = None

        with _GUID_LOCK:
            if _guid_refresh_thread is None:
                _guid_refresh_thread = threading.Thread(
                    target=_bg_satcat_refresh, daemon=True
                )
                _guid_refresh_thread.start()


def get_norad_guid(norad_id: int) -> str:
    """Resolve a NORAD Catalog ID to a COMSPOC GUID.

    The GUID is required by all per-object Spacebook endpoints (synthetic
    covariance, reference ephemeris, etc.).  The satellite catalog is
    downloaded and cached locally on first call, then served from memory.

    Args:
        norad_id: Integer NORAD Catalog number.  Leading zeros are ignored.

    Returns:
        COMSPOC UUID string (e.g. ``"bf72c797-cee3-45b2-8de1-5bc16ac62ea8"``).

    Raises:
        SpacebookLookupError: If the NORAD ID is not found in the catalog.
        SpacebookError: If Spacebook is disabled or unreachable.
    """
    if not SPACEBOOK_ENABLED:
        raise SpacebookError("Spacebook is disabled (ASTRA_SPACEBOOK_ENABLED=false).")

    with _GUID_LOCK:
        loaded = _guid_loaded

    if not loaded:
        with _GUID_DOWNLOAD_LOCK:
            with _GUID_LOCK:
                loaded = _guid_loaded
            if not loaded:
                _load_satcat_guid_map()

    with _GUID_LOCK:
        guid = _guid_map.get(int(norad_id))

    if guid is None:
        raise SpacebookLookupError(
            f"NORAD ID {norad_id} not found in Spacebook satellite catalog. "
            "The object may not be tracked by COMSPOC, or the catalog may be "
            "stale. Try calling refresh_satcat_cache() to force a re-download.",
            norad_id=int(norad_id),
        )

    return guid


def refresh_satcat_cache() -> int:
    """Force a full re-download of the Spacebook satellite catalog.

    Use this when ``get_norad_guid()`` raises ``SpacebookLookupError`` for a
    recently launched satellite that should be in the catalog.

    Returns:
        Number of objects in the refreshed catalog.

    Raises:
        SpacebookError: If the download fails.
    """
    global _guid_loaded
    with _GUID_LOCK:
        _guid_loaded = False
    _load_satcat_guid_map()
    with _GUID_LOCK:
        return len(_guid_map)


# ═══════════════════════════════════════════════════════════════════════════
# TLE / XP-TLE CATALOG FETCHES
# ═══════════════════════════════════════════════════════════════════════════


def fetch_tle_catalog() -> list[SatelliteTLE]:
    """Fetch the current standard TLE catalog from Spacebook.

    Returns:
        List of ``SatelliteTLE`` objects parsed from the Spacebook catalog.
        Each object carries ``source="spacebook_tle"`` in its ``name`` field
        to distinguish data provenance.

    Raises:
        SpacebookError: If Spacebook is unreachable and no local cache exists.
    """
    return _fetch_tle_endpoint(
        url=_SB_TLE_URL,
        cache_filename="tle_catalog.txt",
        ttl_hours=6,
        source_tag="spacebook_tle",
    )


def fetch_xp_tle_catalog() -> list[SatelliteTLE]:
    """Fetch the Extended-Precision XP-TLE catalog from Spacebook.

    XP-TLEs are COMSPOC's numerically refined state vectors expressed in
    standard TLE format.  They are more accurate than public Space-Track
    TLEs but still represent ephemeris best estimated at the TLE epoch.

    When used as initial conditions for ASTRA's Cowell RK87 propagator,
    XP-TLEs significantly reduce the starting-state error compared to
    standard public TLEs.

    Returns:
        List of ``SatelliteTLE`` objects tagged with ``source="spacebook_xptle"``.

    Raises:
        SpacebookError: If Spacebook is unreachable and no local cache exists.
    """
    return _fetch_tle_endpoint(
        url=_SB_XPTLE_URL,
        cache_filename="xp_tle_catalog.txt",
        ttl_hours=6,
        source_tag="spacebook_xptle",
    )


def fetch_historical_tle(date: datetime) -> list[SatelliteTLE]:
    """Fetch the full TLE catalog for a specific historical date from Spacebook.

    Historical catalog files are immutable — once fetched for a given date,
    the file is cached permanently and never re-downloaded.

    Args:
        date: The historical date (UTC). Only the date component is used;
              time-of-day is ignored.

    Returns:
        List of ``SatelliteTLE`` objects from the catalog for that date.

    Raises:
        SpacebookError: If the download fails (e.g. future date, or date
                        precedes the Spacebook archive start).
    """
    date_str = date.strftime("%Y-%m-%d")
    url = _SB_HISTORICAL_TLE.format(date=date_str)
    cache_filename = f"historical_tle_{date_str}.txt"

    # Historical data is immutable — never expire
    return _fetch_tle_endpoint(
        url=url,
        cache_filename=cache_filename,
        ttl_hours=float("inf"),  # never expire
        source_tag=f"spacebook_hist_{date_str}",
    )


def _fetch_tle_endpoint(
    url: str,
    cache_filename: str,
    ttl_hours: float,
    source_tag: str,
) -> list[SatelliteTLE]:
    """Internal: download-or-cache a TLE endpoint and return parsed objects."""
    if not SPACEBOOK_ENABLED:
        raise SpacebookError("Spacebook is disabled (ASTRA_SPACEBOOK_ENABLED=false).")

    cache_path = _sb_cache_path(cache_filename)
    age_h = _cache_age_hours(cache_path)

    if age_h <= ttl_hours and cache_path.exists():
        text = cache_path.read_text(encoding="utf-8", errors="replace")
        logger.debug(
            "Spacebook TLE: using cache '%s' (%.1f h old).", cache_filename, age_h
        )
    else:
        logger.info("Spacebook TLE: downloading from %s", url)
        try:
            resp = _sb_get(url, timeout=60)
            text = resp.text
            cache_path.write_text(text, encoding="utf-8")
        except SpacebookError:
            if cache_path.exists():
                logger.warning(
                    "Spacebook TLE: download failed; using stale cache '%s'.",
                    cache_filename,
                )
                text = cache_path.read_text(encoding="utf-8", errors="replace")
            else:
                raise

    tles = load_tle_catalog(text.splitlines())

    # Source-tag each object so downstream code knows it came from Spacebook.
    # SatelliteTLE is a dataclass — set the extra attribute if possible.
    for tle in tles:
        try:
            object.__setattr__(tle, "_spacebook_source", source_tag)
        except (AttributeError, TypeError):
            pass  # frozen dataclasses: skip gracefully

    logger.debug(
        "Spacebook TLE: parsed %d objects from '%s'.", len(tles), cache_filename
    )
    return tles


# ═══════════════════════════════════════════════════════════════════════════
# PER-OBJECT: SYNTHETIC COVARIANCE (STK EPHEMERIS FORMAT)
# ═══════════════════════════════════════════════════════════════════════════


def fetch_synthetic_covariance_stk(norad_id: int) -> str:
    """Fetch COMSPOC SynCoPate ephemeris with covariance for a satellite.

    The response is an AGI/STK ``.e`` (DotE) format ephemeris containing:
    - COMSPOC's numerically propagated state vectors (TEME frame, km, km/s)
    - ``EphemerisTimePosVel`` block: per-epoch position + velocity
    - Inline 6×6 covariance data (lower-triangular format)
    - Coordinate system: TEMEOfDate
    - Distance unit: Kilometers

    This is the highest-fidelity state-vector timeseries available from
    Spacebook.  Parse with ``astra.ocm.parse_stk_ephemeris()`` to obtain
    a list of ``NumericalState`` objects for direct use in ASTRA propagation
    or benchmark validation.

    Args:
        norad_id: NORAD Catalog number of the satellite.

    Returns:
        Raw STK ephemeris text string.  The caller is responsible for parsing.

    Raises:
        SpacebookLookupError: If ``norad_id`` is not in the Spacebook catalog.
        SpacebookError: If the download fails or Spacebook is disabled.
    """
    if not SPACEBOOK_ENABLED:
        raise SpacebookError("Spacebook is disabled (ASTRA_SPACEBOOK_ENABLED=false).")

    guid = get_norad_guid(norad_id)
    url = _SB_SYNTH_COV_URL.format(guid=guid)
    cache_filename = f"synth_cov_{norad_id}.stk"
    cache_path = _sb_cache_path(cache_filename)
    age_h = _cache_age_hours(cache_path)

    if age_h <= 1.0 and cache_path.exists():
        logger.debug(
            "Spacebook covariance: using cache for NORAD %d (%.1f h old).",
            norad_id,
            age_h,
        )
        return cache_path.read_text(encoding="utf-8", errors="replace")

    logger.info(
        "Spacebook covariance: downloading ephemeris for NORAD %d (GUID=%s).",
        norad_id,
        guid,
    )
    resp = _sb_get(url, timeout=45)
    text = resp.text
    cache_path.write_text(text, encoding="utf-8")
    logger.info(
        "Spacebook covariance: %d bytes for NORAD %d.", len(resp.content), norad_id
    )
    return text


# ═══════════════════════════════════════════════════════════════════════════
# SATCAT DETAILS — Physical Parameters
# ═══════════════════════════════════════════════════════════════════════════


def fetch_satcat_details(norad_id: int) -> dict[str, Any]:
    """Fetch detailed physical parameters for a satellite from Spacebook.

    Returns COMSPOC-derived physical metadata from the DISCOSweb and
    Jonathan McDowell catalogs, including:
    - ``adoptedMass`` — best-estimate total mass (kg)
    - ``ballisticCoefficient`` — COMSPOC-derived B* (m²/kg)
    - ``crossSectionMedian`` — median radar cross section (m²)
    - ``solarRadiationPressure`` — CR × (A/m) coefficient

    Args:
        norad_id: NORAD Catalog number.

    Returns:
        Dictionary with the satellite's detail record, or an empty dict
        if the satellite is not found in the details catalog.

    Raises:
        SpacebookError: If the download fails and no cache exists.
    """
    if not SPACEBOOK_ENABLED:
        raise SpacebookError("Spacebook is disabled (ASTRA_SPACEBOOK_ENABLED=false).")

    import json

    cache_path = _sb_cache_path("satcat_details.json")
    age_h = _cache_age_hours(cache_path)

    if age_h > 24 or not cache_path.exists():
        logger.info("Spacebook details: downloading from %s", _SB_SATCAT_DET_URL)
        try:
            resp = _sb_get(_SB_SATCAT_DET_URL, timeout=120)
            cache_path.write_bytes(resp.content)
            data = resp.json()
        except SpacebookError:
            if cache_path.exists():
                logger.warning("Spacebook details: download failed; using cache.")
                data = json.loads(
                    cache_path.read_text(encoding="utf-8", errors="replace")
                )
            else:
                raise
    else:
        data = json.loads(cache_path.read_text(encoding="utf-8", errors="replace"))

    if isinstance(data, list):
        for rec in data:
            if not isinstance(rec, dict):
                continue
            if int(rec.get("noradId", -1)) == int(norad_id):
                return rec

    return {}
