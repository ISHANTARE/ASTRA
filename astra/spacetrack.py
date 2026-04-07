# astra/spacetrack.py
"""ASTRA Core Space-Track.org Data Ingestion Module.

Provides authenticated access to the 18th Space Defense Squadron's
Space-Track.org catalog — the most authoritative and complete source
of orbital element data, covering over 27,000 tracked objects.

Authentication:
    Space-Track requires a free account. Credentials are read from
    environment variables to keep them out of source code:

    Windows::

        setx SPACETRACK_USER your@email.com
        setx SPACETRACK_PASS yourpassword

    Linux / macOS::

        export SPACETRACK_USER=your@email.com
        export SPACETRACK_PASS=yourpassword

    Then restart your Python session. If the environment variables are not
    set, ``AstraError`` is raised with exact instructions on how to fix it.

Supported Formats:
    - ``format="json"`` (default, recommended): Returns ``list[SatelliteOMM]``
      with full physical metadata (mass, RCS, ballistic coefficient).
    - ``format="tle"``: Returns ``list[SatelliteTLE]`` for legacy workflows.

Example::

    import astra

    # Fetch Starlink constellation in OMM format (recommended)
    starlinks = astra.fetch_spacetrack_group("starlink")

    # Use it directly in the physics pipeline
    leo = astra.filter_altitude(
        [astra.make_debris_object(s) for s in starlinks], 500, 600
    )
"""
from __future__ import annotations

import os
import threading
import json
from typing import Literal, Union, Optional

import requests

from astra.errors import AstraError
from astra.log import get_logger
from astra.models import SatelliteTLE, SatelliteOMM
from astra.version import __version__

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Space-Track API Constants
# ---------------------------------------------------------------------------

_ST_BASE_URL = "https://www.space-track.org"
_ST_LOGIN_URL = f"{_ST_BASE_URL}/ajaxauth/login"
_ST_LOGOUT_URL = f"{_ST_BASE_URL}/ajaxauth/logout"
_ST_QUERY_URL = f"{_ST_BASE_URL}/basicspacedata/query/class/gp"
_ST_SATCAT_URL = f"{_ST_BASE_URL}/basicspacedata/query/class/satcat"

_HEADERS = {
    "User-Agent": f"ASTRA-Core Engine/{__version__} (https://github.com/ISHANTARE/ASTRA)",
}

FormatLiteral = Literal["json", "tle"]

# ---------------------------------------------------------------------------
# Module-level Session Cache
# ---------------------------------------------------------------------------
_SESSION_CACHE: dict[str, requests.Session] = {}
_SESSION_LOCK = threading.Lock()
_ST_WARNING_THRESHOLD = 50000

# ---------------------------------------------------------------------------
# Credentials Helper
# ---------------------------------------------------------------------------

_CREDENTIAL_HELP = """
Space-Track credentials not found in environment variables.

To fix this, set the following environment variables with your
Space-Track.org account credentials (free registration at space-track.org):

  Windows (run in Command Prompt, then restart Python):
    setx SPACETRACK_USER your@email.com
    setx SPACETRACK_PASS yourpassword

  Linux / macOS (add to ~/.bashrc or ~/.zshrc):
    export SPACETRACK_USER=your@email.com
    export SPACETRACK_PASS=yourpassword

After setting the variables, restart your Python session and try again.
""".strip()


def _get_credentials() -> tuple[str, str]:
    """Read Space-Track credentials from environment variables.

    Returns:
        Tuple of (username, password).

    Raises:
        AstraError: If either environment variable is missing, with clear
                    instructions on how to set them.
    """
    user = os.environ.get("SPACETRACK_USER")
    password = os.environ.get("SPACETRACK_PASS")

    if not user or not password:
        raise AstraError(_CREDENTIAL_HELP)

    return user, password


# ---------------------------------------------------------------------------
# Authenticated Session Factory
# ---------------------------------------------------------------------------

def _create_session(username: str, password: str) -> requests.Session:
    """Authenticate with Space-Track and return a session with active cookies.

    Args:
        username: Space-Track.org account email.
        password: Space-Track.org account password.

    Returns:
        An authenticated ``requests.Session`` object.

    Raises:
        AstraError: On network failure or invalid credentials (HTTP 401).
    """
    with _SESSION_LOCK:
        if username in _SESSION_CACHE:
            return _SESSION_CACHE[username]

        session = requests.Session()
        session.headers.update(_HEADERS)

        try:
            resp = session.post(
                _ST_LOGIN_URL,
                data={"identity": username, "password": password},
                timeout=30.0,
            )
        except requests.RequestException as exc:
            raise AstraError(f"Failed to connect to Space-Track.org: {exc}") from exc

        if resp.status_code == 401 or "Failed" in resp.text:
            raise AstraError(
                "Space-Track.org authentication failed. "
                "Check your credentials in SPACETRACK_USER and SPACETRACK_PASS."
            )

        if not resp.ok:
            raise AstraError(
                f"Space-Track.org login returned HTTP {resp.status_code}. "
                "The service may be temporarily unavailable."
            )

        logger.info("Space-Track.org session authenticated successfully.")
        _SESSION_CACHE[username] = session
        return session

def spacetrack_logout() -> None:
    """Clear session cache and logout of Space-Track."""
    username, _ = _get_credentials()
    with _SESSION_LOCK:
        session = _SESSION_CACHE.pop(username, None)
    if session:
        try:
            session.get(_ST_LOGOUT_URL, timeout=10.0)
            logger.info("Successfully logged out of Space-Track.")
        except requests.RequestException as exc:
            logger.warning(f"Error during Space-Track logout: {exc}")
    else:
        logger.info("No active Space-Track session to logout.")


# ---------------------------------------------------------------------------
# Internal Query Helper
# ---------------------------------------------------------------------------

def _query_spacetrack(
    session: requests.Session,
    group: str,
    fmt: FormatLiteral,
) -> str:
    """Execute a GP data query against Space-Track.org.

    Args:
        session: Authenticated session from ``_create_session()``.
        group: Satellite group name (e.g. ``"starlink"``, ``"gps-ops"``).
        fmt: ``"json"`` for OMM, ``"tle"`` for legacy TLE.

    Returns:
        Raw response text (JSON string or TLE lines).
    """
    # Map friendly names directly to Space-Track's optimized filters
    _ST_MAP = {
        "active": "/DECAY_DATE/null-val/EPOCH/>now-30",
        "starlink": "/OBJECT_NAME/~~STARLINK",
        "gps-ops": "/OBJECT_NAME/~~NAVSTAR",
        "iridium-33-debris": "/OBJECT_NAME/~~IRIDIUM 33 DEB",
        "cosmos-2251-debris": "/OBJECT_NAME/~~COSMOS 2251 DEB",
    }
    
    query_filter = _ST_MAP.get(group.lower(), f"/OBJECT_NAME/~~{group.upper()}")
    url = f"{_ST_QUERY_URL}{query_filter}/FORMAT/{fmt}"

    try:
        resp = session.get(url, timeout=60.0)
        resp.raise_for_status()
        
        # Check rate-limit header
        rate_limit = resp.headers.get("X-RateLimit-Remaining")
        if rate_limit is not None:
            try:
                remaining = int(rate_limit)
                if remaining < 10:
                    logger.warning(
                        f"Space-Track.org rate limit low: only {remaining} queries remaining."
                    )
            except ValueError:
                pass

    except requests.RequestException as exc:
        raise AstraError(
            f"Failed to fetch Space-Track group '{group}' [{fmt}]: {exc}"
        ) from exc

    text = resp.text
    if not text.strip():
        raise AstraError(
            f"Space-Track returned an empty response for group '{group}'. "
            "Check that the group name is valid."
        )

    # Pagination guard
    line_count = len(text.splitlines())
    record_count = line_count if fmt == "json" else line_count // 2
    if record_count > _ST_WARNING_THRESHOLD:
        logger.warning(
            f"Space-Track response contains {record_count} records. "
            "Results > 87,000 may be truncated by Space-Track API caps."
        )

    return text


def _parse_spacetrack_response(
    text: str, fmt: FormatLiteral
) -> Union[list[SatelliteOMM], list[SatelliteTLE]]:
    """Route a raw Space-Track response to the correct parser."""
    if fmt == "json":
        from astra.omm import parse_omm_json
        return parse_omm_json(text)
    else:
        from astra.tle import load_tle_catalog
        return load_tle_catalog(text.splitlines())


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_spacetrack_group(
    group: str,
    format: FormatLiteral = "json",
) -> Union[list[SatelliteOMM], list[SatelliteTLE]]:
    """Fetch a satellite group from Space-Track.org using authenticated access.

    Data formats: ✓ SatelliteOMM (format="json", default)  ✓ SatelliteTLE (format="tle")

    Credentials are read automatically from environment variables
    ``SPACETRACK_USER`` and ``SPACETRACK_PASS``. If not set, an ``AstraError``
    is raised with exact instructions on how to configure them.

    Args:
        group: Group name string (e.g. ``"starlink"``, ``"gps-ops"``,
               ``"iridium-33-debris"``).
        format: ``"json"`` (default) for OMM with full physical metadata,
                ``"tle"`` for legacy TLE format.

    Returns:
        - ``list[SatelliteOMM]`` when ``format="json"``
        - ``list[SatelliteTLE]`` when ``format="tle"``

    Raises:
        AstraError: If credentials are missing, authentication fails, or the
                    network request fails.

    Example::

        import astra

        # Fetch Starlink in OMM format (recommended — includes RCS, mass)
        starlinks = astra.fetch_spacetrack_group("starlink")

        # Fetch GPS constellation in legacy TLE format
        gps_tles = astra.fetch_spacetrack_group("gps-ops", format="tle")
    """
    username, password = _get_credentials()
    logger.info(
        f"Fetching Space-Track group '{group}' [{format}] as {username}..."
    )
    session = _create_session(username, password)
    text = _query_spacetrack(session, group, format)
    return _parse_spacetrack_response(text, format)


def fetch_spacetrack_active(
    format: FormatLiteral = "json",
) -> Union[list[SatelliteOMM], list[SatelliteTLE]]:
    """Fetch all active satellites from Space-Track.org using authenticated access.

    Data formats: ✓ SatelliteOMM (format="json", default)  ✓ SatelliteTLE (format="tle")

    Credentials are read automatically from environment variables
    ``SPACETRACK_USER`` and ``SPACETRACK_PASS``.

    Args:
        format: ``"json"`` (default) for OMM with full physical metadata,
                ``"tle"`` for legacy TLE format.

    Returns:
        - ``list[SatelliteOMM]`` when ``format="json"``
        - ``list[SatelliteTLE]`` when ``format="tle"``

    Raises:
        AstraError: If credentials are missing or authentication fails.

    Example::

        import astra
        catalog = astra.fetch_spacetrack_active()
        print(f"Loaded {len(catalog)} active satellites from Space-Track.")
    """
    return fetch_spacetrack_group("active", format=format)


def fetch_spacetrack_satcat(norad_ids: Optional[list[str]] = None) -> list[dict]:
    """Fetch metadata from the General Perturbations Satellite Catalog (SATCAT).

    The SATCAT contains object type classification, launch date, decay date,
    and country of origin.

    Args:
        norad_ids: Optional list of NORAD IDs to filter by. If None, fetches the
                   entire catalog (which may be very large).

    Returns:
        List of dictionaries containing SATCAT metadata.
    """
    username, password = _get_credentials()
    logger.info("Fetching Space-Track SATCAT data...")
    session = _create_session(username, password)

    def _fetch_page(url: str) -> list[dict]:
        try:
            resp = session.get(url, timeout=60.0)
            resp.raise_for_status()
            
            rate_limit = resp.headers.get("X-RateLimit-Remaining")
            if rate_limit is not None:
                try:
                    remaining = int(rate_limit)
                    if remaining < 10:
                        logger.warning(f"Space-Track.org rate limit low: {remaining} queries remaining.")
                except ValueError:
                    pass
                    
        except requests.RequestException as exc:
            raise AstraError(f"Failed to fetch Space-Track SATCAT: {exc}") from exc

        if not resp.text.strip():
            return []

        try:
            return json.loads(resp.text)
        except json.JSONDecodeError as exc:
            raise AstraError(f"Failed to parse SATCAT JSON: {exc}") from exc

    if not norad_ids:
        return _fetch_page(f"{_ST_SATCAT_URL}/FORMAT/json")

    # Batch IDs into chunks to avoid HTTP 414 URI Too Long limits
    results = []
    batch_size = 100
    for i in range(0, len(norad_ids), batch_size):
        batch = norad_ids[i:i + batch_size]
        id_str = ",".join(str(nid) for nid in batch)
        url = f"{_ST_SATCAT_URL}/NORAD_CAT_ID/{id_str}/FORMAT/json"
        results.extend(_fetch_page(url))
        
    return results
