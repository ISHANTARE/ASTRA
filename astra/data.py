"""ASTRA Core Data Ingestion Module.

Bridges the computation engine to live orbital data providers like CelesTrak
and Space-Track. Converts real-time API responses into ASTRA trajectory pipelines.

Supports dual-format ingestion:
    - ``format="tle"``  (default): Returns a ``list[SatelliteTLE]``.
    - ``format="json"`` (OMM):     Returns a ``list[SatelliteOMM]``.

Example::

    # Legacy TLE (default, unchanged behaviour)
    tles = astra.fetch_celestrak_group("starlink")

    # Modern OMM with full physical metadata
    omms = astra.fetch_celestrak_group("starlink", format="json")
"""

from __future__ import annotations
from typing import Any

from typing import Literal, Union, cast

from astra.errors import AstraError
from astra.models import SatelliteTLE, SatelliteOMM
from astra.tle import load_tle_catalog
from astra.log import get_logger
from astra.version import __version__

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

_session = requests.Session()
_retry = Retry(
    total=3,
    backoff_factor=1.0,
    status_forcelist=(429, 500, 502, 503, 504),
    allowed_methods=["HEAD", "GET", "OPTIONS"]
)
_adapter = HTTPAdapter(max_retries=_retry)
_session.mount("http://", _adapter)
_session.mount("https://", _adapter)

logger = get_logger(__name__)

_HEADERS = {
    "User-Agent": (
        f"ASTRA-Core/{__version__} (CelesTrak catalog client; "
        f"https://pypi.org/project/astra-core-engine/)"
    ),
}
_BASE_URL = "https://celestrak.org/NORAD/elements/gp.php"
_SUP_GP_URL = "https://celestrak.org/NORAD/elements/supplemental/sup-gp.php"

FormatLiteral = Literal["tle", "json"]


# ---------------------------------------------------------------------------
# Internal Helpers
# ---------------------------------------------------------------------------


def _format_celestrak_supgp(fmt: FormatLiteral) -> str:
    """FORMAT value for sup-gp.php (CelesTrak expects uppercase TLE / JSON)."""
    return "TLE" if fmt == "tle" else "JSON"


def _supplemental_params(group: str, fmt: FormatLiteral) -> dict[str, Any] | None:
    """Build query params for sup-gp.php, or None if no supplemental route exists.

    ``GROUP=active`` has no ``FILE=active`` equivalent on the supplemental API;
    ``gps-ops`` maps to ``SOURCE=GPS-A`` (broadcast almanac), not the legacy ops list.
    """
    key = group.strip().lower()
    form = _format_celestrak_supgp(fmt)
    if key == "active":
        return None
    if key in ("gps-ops", "gps_ops"):
        return {"SOURCE": "GPS-A", "FORMAT": form}
    return {"FILE": group.strip().lower(), "FORMAT": form}


def _rate_limited(response: requests.Response) -> bool:
    return response.status_code == 403 and (
        "Data is updated once every 2 hours" in response.text
    )


def _legacy_response_triggers_supplemental(response: requests.Response) -> bool:
    """True when gp.php should be retried via sup-gp.php."""
    if response.status_code >= 500:
        return True
    if response.status_code == 200:
        text = (response.text or "").strip()
        if not text:
            return True
        if "invalid query" in text.lower():
            return True
    return False


def _fetch_supplemental_raw(
    group: str, fmt: FormatLiteral, params: dict[str, str]
) -> str:
    """Download raw text/JSON from CelesTrak supplemental sup-gp.php."""
    try:
        response = _session.get(
            _SUP_GP_URL, params=params, headers=_HEADERS, timeout=20.0, verify=True
        )
        if _rate_limited(response):
            raise AstraError(
                f"CelesTrak rate limit reached for group '{group}'. "
                f"Cached data should be used. {response.text.strip()}"
            )
        response.raise_for_status()
    except requests.RequestException as e:
        raise AstraError(
            f"Failed to fetch CelesTrak group '{group}' [{fmt}] via supplemental sup-gp.php: {e}"
        ) from e
    text = (response.text or "").strip()
    if not text or "invalid query" in text.lower():
        raise AstraError(
            f"CelesTrak supplemental sup-gp.php returned no usable data for group '{group}' [{fmt}]."
        )
    return response.text


def _fetch_group_raw(group: str, fmt: FormatLiteral) -> str:
    """Download raw text/JSON from CelesTrak (legacy gp.php, then sup-gp.php if needed)."""
    url = f"{_BASE_URL}?GROUP={group}&FORMAT={fmt}"
    sup_params = _supplemental_params(group, fmt)

    try:
        response = _session.get(url, headers=_HEADERS, timeout=20.0, verify=True)
    except requests.RequestException as e:
        if sup_params is None:
            raise AstraError(
                f"Failed to fetch CelesTrak group '{group}' [{fmt}]: {e}"
            ) from e
        logger.info(
            "CelesTrak gp.php request failed for group %r [%s]; using supplemental sup-gp.php",
            group,
            fmt,
        )
        return _fetch_supplemental_raw(group, fmt, sup_params)

    if _rate_limited(response):
        raise AstraError(
            f"CelesTrak rate limit reached for group '{group}'. "
            f"Cached data should be used. {response.text.strip()}"
        )

    if _legacy_response_triggers_supplemental(response):
        if sup_params is None:
            raise AstraError(
                f"CelesTrak gp.php failed for group '{group}' [{fmt}] "
                f"(HTTP {response.status_code}) and this group has no supplemental "
                "sup-gp.php mapping (e.g. GROUP=active)."
            )
        logger.info(
            "CelesTrak gp.php unavailable for group %r [%s]; using supplemental sup-gp.php",
            group,
            fmt,
        )
        return _fetch_supplemental_raw(group, fmt, sup_params)

    try:
        response.raise_for_status()
    except requests.RequestException as e:
        raise AstraError(
            f"Failed to fetch CelesTrak group '{group}' [{fmt}]: {e}"
        ) from e
    return response.text


def _parse_response(
    text: str, fmt: FormatLiteral
) -> Union[list[SatelliteTLE], list[SatelliteOMM]]:
    """Route a raw API response to the correct parser based on the format string."""
    if fmt == "tle":
        return load_tle_catalog(text.splitlines())
    elif fmt == "json":
        from astra.omm import parse_omm_json

        return parse_omm_json(text)
    else:
        raise AstraError(f"Unsupported format '{fmt}'. Use 'tle' or 'json'.")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def fetch_celestrak_active(
    format: FormatLiteral = "tle",
) -> Union[list[SatelliteTLE], list[SatelliteOMM]]:
    """Fetch the active satellite catalog from CelesTrak.

    Downloads the entire live active catalog and parses it into ASTRA data models.
    Uses legacy ``gp.php`` only (there is no ``sup-gp.php`` equivalent for the full
    active catalog).

    Args:
        format: ``"tle"`` (default) for legacy TLE format,
                ``"json"`` for modern OMM JSON with physical metadata.

    Returns:
        List of ``SatelliteTLE`` (format="tle") or ``SatelliteOMM`` (format="json").
    """
    logger.info(f"Fetching active satellite catalog from CelesTrak [{format}]...")
    text = _fetch_group_raw("active", format)
    return _parse_response(text, format)


def fetch_celestrak_group(
    group: str, format: FormatLiteral = "tle"
) -> Union[list[SatelliteTLE], list[SatelliteOMM]]:
    """Fetch a specific constellation/group from CelesTrak.

    Valid groups include: ``'starlink'``, ``'gps-ops'``, ``'iridium-33-debris'``, etc.

    If legacy ``gp.php`` fails (HTTP 5xx, empty body, or invalid-query body), the client
    retries against supplemental ``sup-gp.php`` where supported. For ``gps-ops``, the
    supplemental path uses broadcast almanac data (``SOURCE=GPS-A``), not the legacy ops list.

    Args:
        group: CelesTrak group name string.
        format: ``"tle"`` (default) for legacy TLE format,
                ``"json"`` for modern OMM JSON with physical metadata.

    Returns:
        List of ``SatelliteTLE`` (format="tle") or ``SatelliteOMM`` (format="json").
    """
    text = _fetch_group_raw(group, format)
    return _parse_response(text, format)


def fetch_celestrak_comprehensive(
    format: FormatLiteral = "tle",
) -> Union[list[SatelliteTLE], list[SatelliteOMM]]:
    """Fetch active payloads plus major debris clouds for a pseudo-full catalog.

    Since CelesTrak does not expose a single unauthenticated 'all' endpoint,
    this function assembles the ~25,000+ most important objects across key groups.

    Args:
        format: ``"tle"`` (default) for legacy TLE format,
                ``"json"`` for modern OMM JSON with physical metadata.

    Returns:
        Deduplicated list of ``SatelliteTLE`` or ``SatelliteOMM`` objects.
    """
    groups = [
        "active",  # All active payloads (~15k)
        "1999-025",  # Fengyun-1C debris (~3k)
        "iridium-33-debris",  # Iridium 33 debris (~300)
        "cosmos-2251-debris",  # Cosmos 2251 debris (~1k)
        "1982-092",  # Cosmos 1408 debris (~500)
        "2019-006",  # MICROSAT-R debris (~100)
        "analyst",  # Analyst objects
    ]

    logger.info(
        f"Assembling comprehensive catalog from {len(groups)} CelesTrak groups [{format}]..."
    )

    seen_ids: set[str] = set()
    unified_catalog: list[Any] = []

    for g in groups:
        try:
            logger.debug(f"Fetching group: {g}")
            objects = fetch_celestrak_group(g, format=format)
            for obj in objects:
                if obj.norad_id not in seen_ids:
                    seen_ids.add(obj.norad_id)
                    unified_catalog.append(obj)
        except AstraError:
            pass  # Skip if a specific group fails

    return unified_catalog


# ---------------------------------------------------------------------------
# Explicit OMM Sibling Functions (Discoverable API)
# ---------------------------------------------------------------------------
# These thin wrappers exist purely for discoverability.
# When a user types `astra.fetch_celestrak_` in their IDE, they immediately
# see both the TLE and OMM variants without needing to know about format=.


def fetch_celestrak_active_omm() -> list[SatelliteOMM]:
    """Fetch the active satellite catalog from CelesTrak in OMM JSON format.

    Returns high-fidelity ``SatelliteOMM`` objects that include physical
    metadata unavailable in TLEs: mass, radar cross-section (RCS), and
    ballistic coefficient.

    Data formats: ✓ SatelliteOMM only (use ``fetch_celestrak_active`` for TLEs)

    Returns:
        List of ``SatelliteOMM`` objects for all active satellites.

    Example::

        import astra
        # OMM — high-fidelity with RCS and mass metadata
        satellites = astra.fetch_celestrak_active_omm()

        # TLE — legacy format (default)
        satellites = astra.fetch_celestrak_active()
    """
    return cast(list[SatelliteOMM], fetch_celestrak_active(format="json"))


def fetch_celestrak_group_omm(group: str) -> list[SatelliteOMM]:
    """Fetch a specific satellite group from CelesTrak in OMM JSON format.

    Returns high-fidelity ``SatelliteOMM`` objects with physical metadata
    unavailable in TLEs: mass, radar cross-section (RCS), and ballistic
    coefficient.

    Data formats: ✓ SatelliteOMM only (use ``fetch_celestrak_group`` for TLEs)

    Args:
        group: CelesTrak group name (e.g. ``"starlink"``, ``"gps-ops"``).

    Returns:
        List of ``SatelliteOMM`` objects.

    Example::

        import astra
        # OMM — high-fidelity
        starlinks = astra.fetch_celestrak_group_omm("starlink")

        # TLE — legacy format (default)
        starlinks = astra.fetch_celestrak_group("starlink")
    """
    return cast(list[SatelliteOMM], fetch_celestrak_group(group, format="json"))


def fetch_celestrak_comprehensive_omm() -> list[SatelliteOMM]:
    """Fetch a comprehensive multi-group catalog from CelesTrak in OMM JSON format.

    Assembles ~25,000+ objects from active satellites, Fengyun-1C debris,
    Iridium 33 debris, Cosmos 2251 debris, and other major debris clouds.

    Data formats: ✓ SatelliteOMM only (use ``fetch_celestrak_comprehensive`` for TLEs)

    Returns:
        Deduplicated list of ``SatelliteOMM`` objects.

    Example::

        import astra
        # OMM — high-fidelity comprehensive catalog
        catalog = astra.fetch_celestrak_comprehensive_omm()

        # TLE — legacy format
        catalog = astra.fetch_celestrak_comprehensive()
    """
    return cast(list[SatelliteOMM], fetch_celestrak_comprehensive(format="json"))
