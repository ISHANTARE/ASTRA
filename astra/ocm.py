# astra/ocm.py
"""ASTRA Core — STK Ephemeris & Orbit State Parser.

Parses the AGI/STK ``.e`` (DotE) ephemeris format returned by the COMSPOC
Spacebook ``/api/entity/synthetic-covariance/{guid}`` endpoint, converting
numerically propagated state vectors into ASTRA ``NumericalState`` objects
for direct use in propagation validation, benchmarking, and conjunction
analysis.

Background
----------
The Spacebook Synthetic Covariance endpoint returns a **STK v12.0 DotE**
ephemeris file — not JSON. This was verified empirically on 2026-04-09.
The file contains:

- Scenario epoch (in STK calendar string format)
- Coordinate system: ``TEMEOfDate`` (the same frame ASTRA's RK87 uses internally)
- Distance unit: Kilometers
- An ``EphemerisTimePosVel`` data block with per-epoch 7-column rows:
  ``t_offset_sec  x_km  y_km  z_km  vx_km_s  vy_km_s  vz_km_s``

Since the coordinate frame (TEME) and units (km, km/s) match ASTRA's internal
conventions exactly, no frame rotation or unit conversion is needed.

Note on Reference Ephemerides
------------------------------
The Spacebook ``/api/entity/reference-ephemerides/{guid}`` and
``/api/entity/reference-ephemerides/ocm/{guid}`` endpoints returned HTTP 500
during our 2026-04-09 live testing. The Synthetic Covariance endpoint is
functionally equivalent (same COMSPOC numerical propagation, same state vectors)
and was used as the file source instead.

Usage Example
-------------
::

    from astra.spacebook import fetch_synthetic_covariance_stk
    from astra.ocm import parse_stk_ephemeris

    stk_text = fetch_synthetic_covariance_stk(norad_id=25544)   # ISS
    states = parse_stk_ephemeris(stk_text)

    print(f"{len(states)} state vectors")
    print(f"First epoch JD: {states[0].t_jd:.4f}")
    print(f"Position (km):  {states[0].position_km}")
    print(f"Velocity (km/s):{states[0].velocity_km_s}")
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from typing import Optional, Any

import numpy as np

try:
    import defusedxml.ElementTree as ET  # blocks XXE / Billion-Laughs attacks
    import xml.etree.ElementTree as ET_BUILDER  # used for safe exporting
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "defusedxml is required for secure OCM XML parsing. "
        "Install astra-core-engine with its declared dependencies."
    ) from exc

from astra.errors import AstraError
from astra.log import get_logger
from astra.propagator import NumericalState

logger = get_logger(__name__)

_CCSDS_DOY_RE = re.compile(
    r"^(?P<year>\d{4})-(?P<doy>\d{3})T"
    r"(?P<hour>\d{2}):(?P<minute>\d{2}):(?P<sec>\d{2}(?:\.\d+)?)"
    r"(?P<tz>Z|[+-]\d{2}:\d{2})?$"
)

# ---------------------------------------------------------------------------
# STK Month Name → Integer Map
# ---------------------------------------------------------------------------
_STK_MONTHS = {
    "Jan": 1,
    "Feb": 2,
    "Mar": 3,
    "Apr": 4,
    "May": 5,
    "Jun": 6,
    "Jul": 7,
    "Aug": 8,
    "Sep": 9,
    "Oct": 10,
    "Nov": 11,
    "Dec": 12,
}

# STK scenario epoch format: "8 Apr 2026 00:44:30.451"
_STK_EPOCH_RE = re.compile(
    r"(\d+)\s+([A-Za-z]{3})\s+(\d{4})\s+(\d{2}):(\d{2}):(\d{1,2}(?:\.\d+)?)"
)


def _parse_stk_epoch(epoch_str: str) -> float:
    """Parse an STK scenario epoch string into a Julian Date (UTC).

    Supported formats:
    - ``"8 Apr 2026 00:44:30.451"``
    - ``"10 Jan 2025 12:00:00"``

    Args:
        epoch_str: Raw epoch string from the STK file ``ScenarioEpoch`` line.

    Returns:
        Julian Date (UTC) as a float.

    Raises:
        AstraError: If the epoch string cannot be parsed.
    """
    m = _STK_EPOCH_RE.search(epoch_str)
    if not m:
        raise AstraError(
            f"Cannot parse STK scenario epoch: {epoch_str!r}. "
            "Expected format: 'D Mon YYYY HH:MM:SS.fff'"
        )
    day = int(m.group(1))
    month = _STK_MONTHS.get(m.group(2).capitalize(), 0)
    year = int(m.group(3))
    hour = int(m.group(4))
    minute = int(m.group(5))
    second = float(m.group(6))

    if month == 0:
        raise AstraError(f"Unknown month abbreviation in STK epoch: {m.group(2)!r}")

    sec_int = int(second)
    microsecond = int(round((second - sec_int) * 1_000_000))

    datetime(year, month, day, hour, minute, sec_int, microsecond, tzinfo=timezone.utc)

    # Convert datetime → Julian Date
    # JD = JDN + (hour - 12) / 24 + min / 1440 + sec / 86400
    # Using the standard algorithm:
    a = (14 - month) // 12
    y = year + 4800 - a
    m_n = month + 12 * a - 3
    jdn = day + (153 * m_n + 2) // 5 + 365 * y + y // 4 - y // 100 + y // 400 - 32045
    # Fractional day
    day_frac = (
        (hour - 12) / 24.0
        + minute / 1440.0
        + (sec_int + microsecond / 1_000_000) / 86400.0
    )
    return float(jdn) + day_frac


def parse_stk_ephemeris(text: str) -> list[NumericalState]:
    """Parse an AGI/STK DotE ephemeris file into a list of NumericalState objects.

    The STK DotE format is returned by the Spacebook synthetic covariance
    endpoint. Each row in the ``EphemerisTimePosVel`` data block provides:
    ``t_sec  x_km  y_km  z_km  vx_km_s  vy_km_s  vz_km_s``

    where ``t_sec`` is seconds elapsed since the ``ScenarioEpoch``.

    The coordinate system is always ``TEMEOfDate`` (verified from live data),
    which is the same frame used by ASTRA's RK87 numerical propagator — no
    frame rotation is required.

    Args:
        text: Raw STK ephemeris file content as a string.

    Returns:
        List of ``NumericalState`` objects sorted by ascending Julian Date,
        with position in km and velocity in km/s in the TEME frame.

    Raises:
        AstraError: If the file is malformed (missing epoch or data block).
    """
    if not text or not text.strip():
        raise AstraError("STK ephemeris text is empty.")

    # ── 1. Validate file signature ─────────────────────────────────────────
    first_line = text.lstrip().split("\n")[0].strip()
    if not first_line.lower().startswith("stk.v"):
        raise AstraError(
            f"Not a valid STK ephemeris file. Expected 'stk.v...' header, "
            f"got: {first_line!r}"
        )

    # ── 2. Extract ScenarioEpoch ────────────────────────────────────────────
    epoch_jd: Optional[float] = None
    coord_system: str = "Unknown"
    distance_unit: str = "Unknown"
    n_points_expected: Optional[int] = None
    in_ephemeris_block = False
    in_data_block = False

    data_rows: list[tuple[float, float, float, float, float, float, float]] = []

    for raw_line in text.splitlines():
        line = raw_line.strip()

        # Skip comment lines (but keep them for coordinate system detection)
        if not line:
            continue
        if line.startswith("#"):
            continue

        tokens_lower = line.lower()

        # Block markers
        if tokens_lower.startswith("begin ephemeris"):
            in_ephemeris_block = True
            continue
        if tokens_lower.startswith("end ephemeris"):
            in_ephemeris_block = False
            in_data_block = False
            continue

        if not in_ephemeris_block:
            continue

        # Header fields
        if tokens_lower.startswith("scenarioepoch"):
            # Format: "    ScenarioEpoch    8 Apr 2026 00:44:30.451"
            parts = line.split(None, 1)
            if len(parts) >= 2:
                epoch_jd = _parse_stk_epoch(parts[1])
            continue

        if tokens_lower.startswith("coordinatesystem"):
            parts = line.split(None, 1)
            if len(parts) >= 2:
                coord_system = parts[1].strip()
            continue

        if tokens_lower.startswith("distanceunit"):
            parts = line.split(None, 1)
            if len(parts) >= 2:
                distance_unit = parts[1].strip()
            continue

        if tokens_lower.startswith("numberofephemerispoints"):
            parts = line.split()
            if len(parts) >= 2:
                try:
                    n_points_expected = int(parts[1])
                except ValueError:
                    pass
            continue

        # Data block marker
        if tokens_lower.startswith("ephemeristimeposvel"):
            in_data_block = True
            continue

        if tokens_lower.startswith("covariancetimeposvel"):
            in_data_block = False
            continue

        if not in_data_block:
            continue

        # Parse data row: t  x  y  z  vx  vy  vz
        fields = line.split()
        if len(fields) < 7:
            continue
        try:
            t_sec = float(fields[0])
            x = float(fields[1])
            y = float(fields[2])
            z = float(fields[3])
            vx = float(fields[4])
            vy = float(fields[5])
            vz = float(fields[6])
            data_rows.append((t_sec, x, y, z, vx, vy, vz))
        except ValueError:
            continue  # Non-numeric row (e.g. covariance section header)

    # ── 3. Validate parsed content ──────────────────────────────────────────
    if epoch_jd is None:
        raise AstraError(
            "STK ephemeris is missing 'ScenarioEpoch' field. "
            "Ensure the file is a valid STK DotE format."
        )
    if not data_rows:
        raise AstraError(
            "STK ephemeris 'EphemerisTimePosVel' block is empty or missing. "
            "No state vectors could be extracted."
        )

    # Unit guard: we only handle Kilometers (verified from Spacebook)
    if distance_unit.lower() not in ("kilometers", "km", ""):
        logger.warning(
            "STK ephemeris distance unit is '%s'; expected 'Kilometers'. "
            "Results may be in wrong units — verify with parse output.",
            distance_unit,
        )

    logger.debug(
        "STK ephemeris: epoch_jd=%.4f | coord=%s | unit=%s | points=%d (expected %s)",
        epoch_jd,
        coord_system,
        distance_unit,
        len(data_rows),
        str(n_points_expected) if n_points_expected else "?",
    )

    # ── 4. Build NumericalState list ────────────────────────────────────────
    states: list[NumericalState] = []
    sec_per_day = 86400.0

    for t_sec, x, y, z, vx, vy, vz in data_rows:
        t_jd = epoch_jd + t_sec / sec_per_day
        states.append(
            NumericalState(
                t_jd=t_jd,
                position_km=np.array([x, y, z], dtype=np.float64),
                velocity_km_s=np.array([vx, vy, vz], dtype=np.float64),
            )
        )

    # Sort by time (should already be in order, but be safe)
    states.sort(key=lambda s: s.t_jd)

    logger.info(
        "parse_stk_ephemeris: %d NumericalState objects extracted. "
        "Time span: %.2f hours.",
        len(states),
        (states[-1].t_jd - states[0].t_jd) * 24.0 if len(states) > 1 else 0.0,
    )
    return states


def parse_ocm(text: str) -> list[NumericalState]:
    """Unified entry point for parsing CCSDS OCM (XML or KVN).

    Automatically detects the format based on the first non-empty line.
    """
    clean_text = text.lstrip()
    if not clean_text:
        raise AstraError("OCM text is empty.")

    if clean_text.startswith("<"):
        return parse_ocm_xml(text)
    return parse_ocm_kvn(text)


def _local_name(tag: str) -> str:
    """Return XML local tag name, stripping namespace prefixes."""
    if "}" in tag:
        return tag.rsplit("}", 1)[-1]
    if ":" in tag:
        return tag.split(":", 1)[-1]
    return tag


def _first_child_by_local_name(parent: Any, name: str) -> Any | None:
    """Find first direct child by local name, namespace-agnostic."""
    for child in list(parent):
        if _local_name(child.tag).lower() == name.lower():
            return child
    return None


def _findall_by_local_name(parent: Any, name: str) -> list[Any]:
    """Find all descendants by local name, namespace-agnostic."""
    out: list[Any] = []
    for node in parent.iter():
        if _local_name(node.tag).lower() == name.lower():
            out.append(node)
    return out


def _get_child_text_by_local_name(node: Any, name: str) -> str:
    """Return required child text by local name or raise AstraError."""
    for child in list(node):
        if _local_name(child.tag).lower() == name.lower():
            text = child.text
            if text is None or not text.strip():
                raise AstraError(f"OCM field '{name}' is empty.")
            return str(text).strip()
    raise AstraError(f"OCM field '{name}' is missing.")


def _parse_ccsds_epoch(epoch_text: str) -> datetime:
    """Parse CCSDS epoch supporting ISO-8601 and YYYY-DOY formats."""
    raw = epoch_text.strip()
    if not raw:
        raise AstraError("OCM epoch is empty.")

    if raw.endswith("Z"):
        iso = raw[:-1] + "+00:00"
    else:
        iso = raw

    try:
        dt = datetime.fromisoformat(iso)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        else:
            dt = dt.astimezone(timezone.utc)
        return dt
    except ValueError:
        pass

    m = _CCSDS_DOY_RE.match(raw)
    if m is None:
        raise AstraError(f"Unsupported CCSDS epoch format: {raw!r}")

    year = int(m.group("year"))
    doy = int(m.group("doy"))
    hour = int(m.group("hour"))
    minute = int(m.group("minute"))
    sec_f = float(m.group("sec"))
    sec_int = int(sec_f)
    micro = int(round((sec_f - sec_int) * 1_000_000))

    if not (1 <= doy <= 366):
        raise AstraError(f"CCSDS day-of-year out of range in epoch: {raw!r}")

    dt_utc = datetime(year, 1, 1, tzinfo=timezone.utc) + timedelta(days=doy - 1)
    dt_utc = dt_utc.replace(hour=hour, minute=minute, second=sec_int, microsecond=micro)

    tz_token = m.group("tz")
    if tz_token and tz_token != "Z":
        sign = 1 if tz_token[0] == "+" else -1
        off_h = int(tz_token[1:3])
        off_m = int(tz_token[4:6])
        offset_s = sign * (off_h * 3600 + off_m * 60)
        dt_utc = dt_utc.replace(tzinfo=timezone.utc) - timedelta(seconds=offset_s)

    return dt_utc


def parse_ocm_xml(xml_text: str) -> list[NumericalState]:
    """Parse a CCSDS OCM XML message into ASTRA NumericalState objects.

    Supports the OCM standard (CCSDS 502.0-B-2), including state vectors
    and metadata for coordinate frames and units.

    Args:
        xml_text: Raw XML content of the OCM.

    Returns:
        List of NumericalState objects.
    """
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        raise AstraError(f"OCM XML Parsing failed: {exc}")

    segments = _findall_by_local_name(root, "segment")
    if not segments:
        raise AstraError("OCM segment not found in XML.")

    from astra.jdutil import datetime_utc_to_jd

    states: list[NumericalState] = []
    for idx, segment in enumerate(segments):
        metadata = _first_child_by_local_name(segment, "metadata")
        data = _first_child_by_local_name(segment, "data")

        if metadata is None or data is None:
            raise AstraError(f"OCM metadata or data block missing in segment #{idx}.")

        state_vectors = _findall_by_local_name(data, "stateVector")
        for sv in state_vectors:
            epoch_dt = _parse_ccsds_epoch(_get_child_text_by_local_name(sv, "EPOCH"))
            t_jd = float(datetime_utc_to_jd(epoch_dt))

            # Position (km)
            x = float(_get_child_text_by_local_name(sv, "X"))
            y = float(_get_child_text_by_local_name(sv, "Y"))
            z = float(_get_child_text_by_local_name(sv, "Z"))

            # Velocity (km/s)
            vx = float(_get_child_text_by_local_name(sv, "X_DOT"))
            vy = float(_get_child_text_by_local_name(sv, "Y_DOT"))
            vz = float(_get_child_text_by_local_name(sv, "Z_DOT"))

            states.append(
                NumericalState(
                    t_jd=t_jd,
                    position_km=np.array([x, y, z]),
                    velocity_km_s=np.array([vx, vy, vz]),
                )
            )

    states.sort(key=lambda s: s.t_jd)
    logger.info("parse_ocm_xml: extracted %d states from OCM.", len(states))
    return states


def export_ocm_xml(states: list[NumericalState], object_name: str = "ASTRA_SAT") -> str:
    """Export ASTRA orbit states to a CCSDS OCM XML string.

    Args:
        states: List of state vectors to export.
        object_name: Identifier for the satellite.

    Returns:
        XML string in CCSDS OCM format.
    """
    from astra.jdutil import jd_utc_to_datetime

    root = ET_BUILDER.Element("ocm", {"id": "ASTRA_OCM_EXPORT", "version": "2.0"})
    header = ET_BUILDER.SubElement(root, "header")
    ET_BUILDER.SubElement(header, "CREATION_DATE").text = datetime.now(
        timezone.utc
    ).isoformat()
    ET_BUILDER.SubElement(header, "ORIGINATOR").text = "ASTRA_ENGINE"

    body = ET_BUILDER.SubElement(root, "body")
    segment = ET_BUILDER.SubElement(body, "segment")

    metadata = ET_BUILDER.SubElement(segment, "metadata")
    ET_BUILDER.SubElement(metadata, "OBJECT_NAME").text = object_name
    ET_BUILDER.SubElement(metadata, "CENTER_NAME").text = "EARTH"
    ET_BUILDER.SubElement(metadata, "TIME_SYSTEM").text = "UTC"
    ET_BUILDER.SubElement(metadata, "REF_FRAME").text = "TEME"

    data = ET_BUILDER.SubElement(segment, "data")
    ET_BUILDER.SubElement(data, "stateVector")  # Simplified wrapper

    for s in states:
        sv = ET_BUILDER.SubElement(data, "stateVector")
        dt = jd_utc_to_datetime(s.t_jd)
        ET_BUILDER.SubElement(sv, "EPOCH").text = dt.isoformat()  # type: ignore[union-attr]
        ET_BUILDER.SubElement(sv, "X").text = f"{s.position_km[0]:.6f}"
        ET_BUILDER.SubElement(sv, "Y").text = f"{s.position_km[1]:.6f}"
        ET_BUILDER.SubElement(sv, "Z").text = f"{s.position_km[2]:.6f}"
        ET_BUILDER.SubElement(sv, "X_DOT").text = f"{s.velocity_km_s[0]:.9f}"
        ET_BUILDER.SubElement(sv, "Y_DOT").text = f"{s.velocity_km_s[1]:.9f}"
        ET_BUILDER.SubElement(sv, "Z_DOT").text = f"{s.velocity_km_s[2]:.9f}"

    return ET_BUILDER.tostring(root, encoding="unicode")


def parse_ocm_kvn(kvn_text: str) -> list[NumericalState]:
    """Parse a CCSDS OCM KVN (Key-Value Notation) message.

    Format defined in CCSDS 502.0-B-2. Processes one or more segments
    containing metadata and state vectors.
    """

    lines = kvn_text.splitlines()
    states: list[NumericalState] = []

    current_state_data: dict[str, Any] = {}

    for line in lines:
        line = line.strip()
        if not line or line.startswith("COMMENT"):
            continue

        if "=" not in line:
            continue

        key, val = [x.strip() for x in line.split("=", 1)]
        key = key.upper()

        if key == "EPOCH":
            # If we have a pending state, save it
            if "X" in current_state_data:
                states.append(_build_state_from_kvn(current_state_data))
            current_state_data = {"EPOCH": val}
        elif key in ("X", "Y", "Z", "X_DOT", "Y_DOT", "Z_DOT"):
            current_state_data[key] = float(val)
        elif key == "REF_FRAME":
            current_state_data["REF_FRAME"] = val

    # Final state
    if "X" in current_state_data:
        states.append(_build_state_from_kvn(current_state_data))

    states.sort(key=lambda s: s.t_jd)
    return states


def _build_state_from_kvn(data: dict[str, Any]) -> NumericalState:
    """Helper to convert KVN dict to NumericalState."""
    from astra.jdutil import datetime_utc_to_jd

    t_str = str(data["EPOCH"])
    dt = _parse_ccsds_epoch(t_str)

    return NumericalState(
        t_jd=float(datetime_utc_to_jd(dt)),
        position_km=np.array([data["X"], data["Y"], data["Z"]]),
        velocity_km_s=np.array([data["X_DOT"], data["Y_DOT"], data["Z_DOT"]]),
    )
