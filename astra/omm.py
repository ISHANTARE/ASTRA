# astra/omm.py
"""ASTRA Core OMM (Orbit Mean-Elements Message) Parser.

Provides isolated ingestion and validation of CCSDS OMM JSON payloads
as published by Space-Track.org and CelesTrak.

This module acts as the dedicated "parser funnel" for the modern OMM format.
It does NOT contain any physics or propagation logic. Its only job is to safely
translate a JSON dictionary from an API response into a ``SatelliteOMM``
dataclass, applying all necessary unit conversions so the physics engine
receives correctly-scaled floats.

Key Conversions (OMM JSON → SGP4 Engine):
    - Angles (inclination, RAAN, arg. perigee, mean anomaly): degrees → radians.
    - Mean Motion: rev/day → rad/min.
    - Epoch (ISO-8601 timestamp string): → Julian Date float.

References:
    CCSDS 502.0-B-3 Recommendation for Space Data System Standards.
    Space-Track.org API Documentation, OMM JSON format.
    Celestrak GP Data Documentation (https://celestrak.org/SOCRATES/help.php).
"""

from __future__ import annotations
from typing import Any

import json
import math
import pathlib
from datetime import datetime, timezone
from typing import Optional

from astra.errors import AstraError, InvalidTLEError
from astra.log import get_logger
from astra.models import SatelliteOMM, SatelliteTLE

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Epoch Conversion Helpers
# ---------------------------------------------------------------------------


def _epoch_iso_to_jd(epoch_str: str) -> float:
    """Convert an ISO-8601 timestamp string to a Julian Date float.

    Space-Track OMM epoch strings are formatted as ``YYYY-MM-DDTHH:MM:SS.ffffff``
    (UTC). This converts them to the Julian Date representation used
    uniformly throughout ASTRA-Core using only the Python standard library.

    Note:
        Python's ``datetime`` does not account for leap seconds (TAI-UTC).  For
        all epochs after 1972, the resulting JD differs from an astropy leap-second
        aware value by at most 37 s / 86400 s/day ≈ 4.3 × 10⁻⁴ days, well within
        SGP4's own accuracy limits.

    Args:
        epoch_str: ISO-8601 epoch from OMM JSON, e.g. ``"2024-07-15T12:00:00.000000"``.

    Returns:
        Julian Date as float64.

    Raises:
        AstraError: If the epoch string cannot be parsed.
    """
    clean = epoch_str.strip().rstrip("Z")
    # Handle space-separated variant (YYYY-MM-DD HH:MM:SS)
    clean = clean.replace(" ", "T")
    try:
        dt = datetime.fromisoformat(clean).replace(tzinfo=timezone.utc)
    except ValueError as exc:
        raise AstraError(
            f"Failed to parse OMM epoch '{epoch_str}': {exc}. "
            "Expected ISO-8601 format e.g. '2024-07-15T12:00:00.000000'."
        ) from exc

    # Julian Date of 2000-01-01T12:00:00 UTC = 2451545.0
    _J2000 = datetime(2000, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    delta_s = (dt - _J2000).total_seconds()
    return 2451545.0 + delta_s / 86400.0


# ---------------------------------------------------------------------------
# Single OMM Record Parser
# ---------------------------------------------------------------------------


def parse_omm_record(record: dict[str, Any]) -> SatelliteOMM:
    """Parse a single OMM JSON dictionary into a ``SatelliteOMM`` dataclass.

    Applies all mandatory unit conversions to guarantee the returned object
    is ready for direct injection into the SGP4 physics engine via
    ``Satrec.sgp4init()``.

    Args:
        record: A single dictionary from a Space-Track or CelesTrak OMM JSON
                response. Must contain at minimum the Keplerian element keys.

    Returns:
        A fully populated ``SatelliteOMM`` instance.

    Raises:
        InvalidTLEError: If mandatory orbital element fields are absent or
                         non-numeric.
    """

    def _get_float(key: str, default: Optional[float] = None) -> Optional[float]:
        val = record.get(key)
        if val is None or str(val).strip() == "":
            return default
        try:
            return float(val)
        except (ValueError, TypeError):
            return default

    def _req_float(key: str) -> float:
        """Retrieve a required float field; raise if absent."""
        val = _get_float(key)
        if val is None:
            raise InvalidTLEError(
                f"OMM record is missing required field '{key}'. "
                f"Available keys: {list(record.keys())}"
            )
        return val

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------
    norad_id_raw = record.get("NORAD_CAT_ID")
    if norad_id_raw is None or str(norad_id_raw).strip() == "":
        raise InvalidTLEError(
            "OMM record is missing required field 'NORAD_CAT_ID'. "
            "Cannot assign a unique identifier to this object. "
            f"Available keys: {list(record.keys())}"
        )
    norad_id = str(norad_id_raw).strip()
    name = str(record.get("OBJECT_NAME", record.get("OBJECT_ID", "UNKNOWN"))).strip()
    object_type_raw = str(record.get("OBJECT_TYPE", "UNKNOWN")).upper()
    # Normalize to our canonical types
    _type_map = {
        "PAYLOAD": "PAYLOAD",
        "ROCKET BODY": "ROCKET_BODY",
        "ROCKET_BODY": "ROCKET_BODY",
        "DEBRIS": "DEBRIS",
        "DEBRIS/PAYLOAD": "DEBRIS",
    }
    object_type = _type_map.get(object_type_raw, "UNKNOWN")

    # ------------------------------------------------------------------
    # Epoch
    # ------------------------------------------------------------------
    epoch_str = str(record.get("EPOCH", "")).strip()
    if not epoch_str:
        raise InvalidTLEError(f"OMM record for '{name}' is missing EPOCH field.")
    epoch_jd = _epoch_iso_to_jd(epoch_str)

    # ------------------------------------------------------------------
    # Keplerian Elements (with mandatory unit conversions)
    # ------------------------------------------------------------------
    # OMM stores angles in DEGREES; SGP4 needs RADIANS.
    inclination_rad = math.radians(_req_float("INCLINATION"))
    raan_rad = math.radians(_req_float("RA_OF_ASC_NODE"))
    argpo_rad = math.radians(_req_float("ARG_OF_PERICENTER"))
    mo_rad = math.radians(_req_float("MEAN_ANOMALY"))
    eccentricity = _req_float("ECCENTRICITY")  # dimensionless [0, 1)

    # OMM stores Mean Motion in rev/day; SGP4 needs rad/min.
    mean_motion_rev_day = _req_float("MEAN_MOTION")
    mean_motion_rad_min = mean_motion_rev_day * (2.0 * math.pi) / 1440.0

    # SGP4 ballistic coefficient (1/earth_radii)
    bstar = _req_float("BSTAR")
    # Modern OMM format provides drag terms directly; default to 0 for older sources
    mean_motion_dot = _get_float("MEAN_MOTION_DOT", 0.0)
    mean_motion_ddot = _get_float("MEAN_MOTION_DDOT", 0.0)

    # ------------------------------------------------------------------
    # Optional High-Fidelity Physical Properties (OMM-only data)
    # ------------------------------------------------------------------
    # RCS_SIZE from Space-Track: SMALL, MEDIUM, LARGE → rough m² conversions
    rcs_size_str = str(record.get("RCS_SIZE", "")).upper().strip()
    _rcs_map = {"SMALL": 0.1, "MEDIUM": 1.0, "LARGE": 10.0}
    rcs_m2: Optional[float] = _rcs_map.get(rcs_size_str, None)

    # Prefer explicit numerical RCS if available
    rcs_m2_explicit = _get_float("RCS")
    if rcs_m2_explicit is not None:
        rcs_m2 = rcs_m2_explicit

    if rcs_m2 is None:
        logger.debug(
            f"OMM record for '{name}' missing RCS data (explicit or categorical)."
        )

    mass_kg: Optional[float] = _get_float("MASS")
    if mass_kg is None:
        logger.debug(
            f"OMM record for '{name}' missing MASS field. Numerical propagation will require explicit mass."
        )

    cd_area_over_mass: Optional[float] = _get_float("CD_AREA_OVER_MASS")

    # ------------------------------------------------------------------
    # Physical range checks on elements
    # ------------------------------------------------------------------
    _validation_errors = []
    if not (0.0 <= eccentricity < 1.0):
        _validation_errors.append(f"ECCENTRICITY={eccentricity} out of range [0, 1)")
    if mean_motion_rev_day <= 0.0:
        _validation_errors.append(f"MEAN_MOTION={mean_motion_rev_day} must be positive")
    if not (0.0 <= math.degrees(inclination_rad) <= 180.0):
        _validation_errors.append(
            f"INCLINATION={math.degrees(inclination_rad):.4f} out of range [0, 180]"
        )
    if abs(bstar) > 1.0:
        _validation_errors.append(
            f"|BSTAR|={abs(bstar):.4e} > 1.0 (physically unrealistic for atmospheric drag coefficient)"
        )
    if _validation_errors:
        raise InvalidTLEError(
            f"OMM record for '{name}' (NORAD {norad_id}) failed physical validation: "
            + "; ".join(_validation_errors)
        )

    return SatelliteOMM(
        norad_id=norad_id,
        name=name,
        epoch_jd=epoch_jd,
        object_type=object_type,
        inclination_rad=inclination_rad,
        raan_rad=raan_rad,
        argpo_rad=argpo_rad,
        mo_rad=mo_rad,
        eccentricity=eccentricity,
        mean_motion_rad_min=mean_motion_rad_min,
        bstar=bstar,
        mean_motion_dot=float(mean_motion_dot or 0.0),
        mean_motion_ddot=float(mean_motion_ddot or 0.0),
        rcs_m2=rcs_m2,
        mass_kg=mass_kg,
        cd_area_over_mass=cd_area_over_mass,
    )


# ---------------------------------------------------------------------------
# Bulk JSON Catalog Parser
# ---------------------------------------------------------------------------


def parse_omm_json(json_text: str) -> list[SatelliteOMM]:
    """Parse a bulk CelesTrak / Space-Track OMM JSON response string.

    Handles both a JSON array of records (CelesTrak style) and ensures
    robust error logging for individual malformed records without aborting
    the entire catalog ingestion.

    Args:
        json_text: Raw JSON string from an HTTP response body.

    Returns:
        List of successfully parsed ``SatelliteOMM`` instances. Malformed
        individual records are skipped and logged as warnings.

    Raises:
        AstraError: If the top-level JSON structure is unparseable.

    Example::

        import astra
        satellites = astra.parse_omm_json(response_text)
        print(f"Loaded {len(satellites)} OMM objects.")
    """
    if len(json_text) > 50 * 1024 * 1024:
        raise AstraError(
            "OMM JSON payload exceeds maximum allowed size (50MB). Potential Denial of Service."
        )

    try:
        data = json.loads(json_text)
    except json.JSONDecodeError as exc:
        raise AstraError(f"Failed to parse OMM JSON payload: {exc}") from exc

    if not isinstance(data, list):
        raise AstraError(
            f"Expected an OMM JSON array at the top level, got {type(data).__name__}. "
            "Ensure the Space-Track/CelesTrak endpoint uses FORMAT=JSON."
        )

    results: list[SatelliteOMM] = []
    errors = 0
    for i, record in enumerate(data):
        try:
            omm = parse_omm_record(record)
            results.append(omm)
        except (AstraError, InvalidTLEError, KeyError) as exc:
            logger.warning(
                f"Skipping OMM record #{i} (NORAD: {record.get('NORAD_CAT_ID', '?')}): {exc}"
            )
            errors += 1

    if errors:
        logger.warning(
            f"OMM parsing complete: {len(results)} loaded, {errors} skipped."
        )
    else:
        logger.info(
            f"OMM parsing complete: {len(results)} records loaded successfully."
        )

    return results


# ---------------------------------------------------------------------------
# XP-TLE Translation
# ---------------------------------------------------------------------------


def xptle_to_satellite_omm(tle_objects: list["SatelliteTLE"]) -> list[SatelliteOMM]:
    """Convert SatelliteTLE objects (e.g. from Spacebook XP-TLE) to SatelliteOMM.

    Extracts the Keplerian elements using SGP4's internal parser and populates
    a generic SatelliteOMM structure. The resulting objects will inherit metadata
    tags (like ``_spacebook_source``) transparently.

    Since TLEs lack mass and RCS by definition, these physical fields will be ``None``.

    Args:
        tle_objects: List of ``SatelliteTLE`` instances.

    Returns:
        List of ``SatelliteOMM`` instances matching the TLE element states.
    """
    from sgp4.api import Satrec

    results = []

    for tle in tle_objects:
        try:
            satrec = Satrec.twoline2rv(tle.line1, tle.line2)
            omm = SatelliteOMM(
                norad_id=tle.norad_id,
                name=tle.name,
                epoch_jd=tle.epoch_jd,
                object_type=tle.object_type,
                inclination_rad=satrec.inclo,
                raan_rad=satrec.nodeo,
                argpo_rad=satrec.argpo,
                mo_rad=satrec.mo,
                eccentricity=satrec.ecco,
                mean_motion_rad_min=satrec.no_kozai,
                bstar=satrec.bstar,
                mean_motion_dot=getattr(satrec, "ndot", 0.0),
                mean_motion_ddot=getattr(satrec, "nddot", 0.0),
                rcs_m2=None,
                mass_kg=None,
                cd_area_over_mass=None,
            )
            # Propagate Spacebook provenance tags if present
            source = getattr(tle, "_spacebook_source", None)
            if source:
                object.__setattr__(omm, "_spacebook_source", source)

            results.append(omm)
        except Exception as exc:
            logger.warning(
                f"Failed to convert TLE for {tle.name} (NORAD {tle.norad_id}) to OMM: {exc}"
            )

    return results


# ---------------------------------------------------------------------------
# Local File Loader
# ---------------------------------------------------------------------------


def load_omm_file(filepath: str) -> list[SatelliteOMM]:
    """Load a local OMM JSON file from disk and parse it into ``SatelliteOMM`` objects.

    This is the OMM equivalent of ``load_tle_catalog()`` and is intended for
    users who download OMM data manually from Space-Track.org or CelesTrak
    and want to load it from a local path.

    Data formats: ✓ SatelliteOMM only (use ``load_tle_catalog`` for TLEs)

    Args:
        filepath: Path to a local ``.json`` file containing an OMM JSON array.

    Returns:
        List of ``SatelliteOMM`` instances.

    Raises:
        AstraError: If the file does not exist, is not readable, or the JSON
                    structure is invalid.

    Example::

        import astra
        satellites = astra.load_omm_file("starlink_omm.json")
        print(f"Loaded {len(satellites)} Starlink satellites from OMM file.")
    """
    path = pathlib.Path(filepath)
    if not path.exists():
        raise AstraError(
            f"OMM file not found: '{filepath}'. "
            "Download OMM data from https://celestrak.org or https://www.space-track.org."
        )
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise AstraError(f"Failed to read OMM file '{filepath}': {exc}") from exc

    logger.info(f"Loading OMM catalog from file: {filepath}")
    return parse_omm_json(text)


# ---------------------------------------------------------------------------
# OMM Validator
# ---------------------------------------------------------------------------


def validate_omm(record: dict[str, Any]) -> bool:
    """Validate that an OMM JSON dictionary contains physically sensible values.

    Performs lightweight sanity checks on the orbital elements without
    attempting a full parse. Use before ``parse_omm_record()`` to pre-screen
    records from untrusted sources.

    Checks performed:
        - Required keys are present and non-empty.
        - Eccentricity is in [0, 1).
        - Mean motion is positive.
        - Inclination is in [0, 180].
        - Epoch string is parseable.

    Args:
        record: A single OMM dictionary (as returned from ``json.loads()``).

    Returns:
        ``True`` if the record passes all sanity checks, ``False`` otherwise.

    Example::

        import astra, json
        records = json.loads(open("catalog.json").read())
        valid = [r for r in records if astra.validate_omm(r)]
        print(f"{len(valid)}/{len(records)} records passed validation.")
    """
    required_keys = [
        "INCLINATION",
        "RA_OF_ASC_NODE",
        "ARG_OF_PERICENTER",
        "MEAN_ANOMALY",
        "ECCENTRICITY",
        "MEAN_MOTION",
        "BSTAR",
        "EPOCH",
    ]

    for key in required_keys:
        val = record.get(key)
        if val is None or str(val).strip() == "":
            logger.debug(f"validate_omm: missing required key '{key}'")
            return False

    try:
        ecc = float(record["ECCENTRICITY"])
        if not (0.0 <= ecc < 1.0):
            logger.debug(f"validate_omm: eccentricity {ecc} out of range [0, 1)")
            return False

        mm = float(record["MEAN_MOTION"])
        if mm <= 0.0:
            logger.debug(f"validate_omm: mean_motion {mm} must be positive")
            return False

        inc = float(record["INCLINATION"])
        if not (0.0 <= inc <= 180.0):
            logger.debug(f"validate_omm: inclination {inc} out of range [0, 180]")
            return False

        _epoch_iso_to_jd(str(record["EPOCH"]))  # raises AstraError if unparseable

    except (ValueError, TypeError, AstraError) as exc:
        logger.debug(f"validate_omm: field conversion failed: {exc}")
        return False

    return True
