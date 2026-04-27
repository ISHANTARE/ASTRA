"""ASTRA Core Conjunction Data Message (CDM) Parser.

Implements structural parsing for standard CCSDS CDMs provided by Space-Track
and the US Space Force. XML is parsed with ``defusedxml`` to mitigate XXE and
billion-laughs risks on untrusted input.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Any

import defusedxml.ElementTree as ET

from astra.errors import AstraError
from astra.log import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class CDMObject:
    """Represents a single object inside a CDM (Object1 or Object2).

    Attributes:
        object_designator: Unique identifier (e.g. NORAD ID).
        object_name: Common name of the object.
        position_xyz: Cartesian position vector in **km** (J2000/GCRF).
        velocity_xyz: Cartesian velocity vector in **km/s** (J2000/GCRF).
        covariance_matrix: 21-element upper triangular RTN covariance in **m²** and **m²/s**.
    """

    object_designator: str
    object_name: str
    position_xyz: tuple[float, float, float]
    velocity_xyz: tuple[float, float, float]
    covariance_matrix: list[float]


@dataclass(frozen=True)
class ConjunctionDataMessage:
    """Represents the complete payload of a CCSDS XML CDM.

    Attributes:
        message_id: Unique message identifier.
        creation_date: UTC timestamp of message generation.
        tca_time: UTC timestamp of Time of Closest Approach.
        miss_distance_m: Scalar distance at TCA in **meters**.
        relative_velocity_m_s: Scalar relative speed at TCA in **m/s**.
        collision_probability: Probability of collision (Pc) in range [0, 1].
        object_1: The primary object in the conjunction.
        object_2: The secondary object in the conjunction.
    """

    message_id: str
    creation_date: datetime
    tca_time: datetime
    miss_distance_m: float
    relative_velocity_m_s: float
    collision_probability: Optional[float]
    object_1: CDMObject
    object_2: CDMObject


def _parse_time(time_str: str) -> datetime:
    """Parse standard ISO 8601 formatting from CCSDS XML."""
    clean_str = time_str.replace("Z", "+00:00")
    return datetime.fromisoformat(clean_str)  # type: ignore[no-any-return]


def _findtext(element: Any, tag: str, default: str = "") -> str:
    """Search for a tag at any depth within the XML tree."""
    result = element.findtext(f".//{tag}", default=default)
    return str(result)


def _sanitize_cdm_xml(xml_string: str) -> str:
    """Strip namespace declarations and prefixed tags for reliable ``findtext``."""
    s = xml_string
    s = re.sub(r'\sxmlns(?::[a-zA-Z0-9_-]+)?="[^"]*"', "", s)
    s = re.sub(r"<([a-zA-Z0-9]+):([a-zA-Z0-9_]+)", r"<\2", s)
    s = re.sub(r"</([a-zA-Z0-9]+):([a-zA-Z0-9_]+)", r"</\2", s)
    return s  # type: ignore[no-any-return]


def _parse_cdm_object(root: Any, prefix: str) -> CDMObject:
    """Parse a single CDM object (OBJECT1 or OBJECT2) from the XML tree."""
    designator = _findtext(root, f"{prefix}_OBJECT_DESIGNATOR", "UNKNOWN")
    if not designator or designator == "UNKNOWN":
        designator = _findtext(root, "OBJECT_DESIGNATOR", "UNKNOWN")

    name = _findtext(root, f"{prefix}_OBJECT_NAME", "Unknown")
    if not name or name == "Unknown":
        name = _findtext(root, "OBJECT_NAME", "Unknown")

    x = float(_findtext(root, f"{prefix}_X", "0.0"))
    y = float(_findtext(root, f"{prefix}_Y", "0.0"))
    z = float(_findtext(root, f"{prefix}_Z", "0.0"))
    vx = float(_findtext(root, f"{prefix}_X_DOT", "0.0"))
    vy = float(_findtext(root, f"{prefix}_Y_DOT", "0.0"))
    vz = float(_findtext(root, f"{prefix}_Z_DOT", "0.0"))

    cov_tags = [
        "CR_R",
        "CT_R",
        "CT_T",
        "CN_R",
        "CN_T",
        "CN_N",
        "CRDOT_R",
        "CRDOT_T",
        "CRDOT_N",
        "CRDOT_RDOT",
        "CTDOT_R",
        "CTDOT_T",
        "CTDOT_N",
        "CTDOT_RDOT",
        "CTDOT_TDOT",
        "CNDOT_R",
        "CNDOT_T",
        "CNDOT_N",
        "CNDOT_RDOT",
        "CNDOT_TDOT",
        "CNDOT_NDOT",
    ]
    cov = []
    for tag in cov_tags:
        val_str = _findtext(root, tag, "0.0")
        cov.append(float(val_str))

    return CDMObject(  # type: ignore[no-any-return]
        object_designator=designator,
        object_name=name,
        position_xyz=(x, y, z),
        velocity_xyz=(vx, vy, vz),
        covariance_matrix=cov,
    )


def parse_cdm_xml(xml_string: str) -> ConjunctionDataMessage:
    """Parses a standard CCSDS XML Conjunction Data Message.

    Args:
        xml_string: Raw XML response from Space-Track or local CDM file.

    Returns:
        ConjunctionDataMessage object containing structured geometry and covariance.

    Raises:
        AstraError: If the XML format is invalid or missing critical CDM tags.
    """
    logger.info("Parsing CCSDS Conjunction Data Message (XML)...")
    try:
        clean_xml = _sanitize_cdm_xml(xml_string)
        root = ET.fromstring(clean_xml)

        msg_id = _findtext(root, "MESSAGE_ID", "UNKNOWN")
        creation_str = _findtext(root, "CREATION_DATE", "1970-01-01T00:00:00Z")
        tca_str = _findtext(root, "TCA", "1970-01-01T00:00:00Z")
        miss_m = float(_findtext(root, "MISS_DISTANCE", "0.0"))
        rel_vel = float(_findtext(root, "RELATIVE_SPEED", "0.0"))

        pc_str = _findtext(root, "COLLISION_PROBABILITY", "")
        pc_val = float(pc_str) if pc_str else None

        obj1 = _parse_cdm_object(root, "OBJECT1")
        obj2 = _parse_cdm_object(root, "OBJECT2")

        # --------------------------------------------------------------
        # Physical Validation (SE-F)
        # --------------------------------------------------------------
        validation_errors = []
        if miss_m < 0.0:
            validation_errors.append(f"Negative miss distance: {miss_m} m")
        if rel_vel < 0.0:
            validation_errors.append(f"Negative relative velocity: {rel_vel} m/s")
        if pc_val is not None and not (0.0 <= pc_val <= 1.0):
            validation_errors.append(
                f"Probability of collision {pc_val} out of range [0, 1]"
            )

        if validation_errors:
            error_msg = f"CDM {msg_id} failed physical validation: " + "; ".join(
                validation_errors
            )
            logger.error(error_msg)
            raise AstraError(error_msg)

        tca = _parse_time(tca_str)
        creation = _parse_time(creation_str)

        logger.debug(f"Decoded CDM {msg_id} - TCA: {tca.isoformat()} - Miss: {miss_m}m")

        return ConjunctionDataMessage(  # type: ignore[no-any-return]
            message_id=msg_id,
            creation_date=creation,
            tca_time=tca,
            miss_distance_m=miss_m,
            relative_velocity_m_s=rel_vel,
            collision_probability=pc_val,
            object_1=obj1,
            object_2=obj2,
        )
    except (ET.ParseError, ValueError, TypeError, KeyError) as e:
        logger.error(f"CDM Parsing failed: {e}")
        raise AstraError(f"Invalid CCSDS CDM format: {e}") from e


# ---------------------------------------------------------------------------
# CDM Exporter
# ---------------------------------------------------------------------------


def export_cdm_xml(
    cdm: "ConjunctionDataMessage",
    originator: str = "ASTRA-CORE",
) -> str:
    """Serialise a :class:`ConjunctionDataMessage` to a CCSDS XML CDM string.

    Produces a standards-compliant XML document that can be shared with
    mission-control systems or Space-Track partners.  This completes the
    read-write symmetry: ``parse_cdm_xml(export_cdm_xml(cdm)) == cdm``.

    Args:
        cdm: The :class:`ConjunctionDataMessage` to serialise.
        originator: Originator identifier written into the XML header.

    Returns:
        A UTF-8 XML string conforming to CCSDS 508.0-B-1 CDM schema.

    Example::

        xml_str = export_cdm_xml(event_cdm, originator="ISRO-ISTRAC")
        with open("conjunction.xml", "w") as f:
            f.write(xml_str)
    """
    from xml.etree.ElementTree import Element, SubElement, tostring
    import xml.dom.minidom as minidom

    def _sub(parent, tag, text=""):
        el = SubElement(parent, tag)
        el.text = str(text)
        return el

    def _add_object(root, obj, prefix):
        _sub(root, f"{prefix}_OBJECT_DESIGNATOR", obj.object_designator)
        _sub(root, f"{prefix}_OBJECT_NAME", obj.object_name)
        x, y, z = obj.position_xyz
        vx, vy, vz = obj.velocity_xyz
        _sub(root, f"{prefix}_X", f"{x:.6f}")
        _sub(root, f"{prefix}_Y", f"{y:.6f}")
        _sub(root, f"{prefix}_Z", f"{z:.6f}")
        _sub(root, f"{prefix}_X_DOT", f"{vx:.9f}")
        _sub(root, f"{prefix}_Y_DOT", f"{vy:.9f}")
        _sub(root, f"{prefix}_Z_DOT", f"{vz:.9f}")
        cov_tags = [
            "CR_R", "CT_R", "CT_T", "CN_R", "CN_T", "CN_N",
            "CRDOT_R", "CRDOT_T", "CRDOT_N", "CRDOT_RDOT",
            "CTDOT_R", "CTDOT_T", "CTDOT_N", "CTDOT_RDOT", "CTDOT_TDOT",
            "CNDOT_R", "CNDOT_T", "CNDOT_N", "CNDOT_RDOT", "CNDOT_TDOT", "CNDOT_NDOT",
        ]
        for tag, val in zip(cov_tags, obj.covariance_matrix):
            _sub(root, tag, f"{val:.6e}")

    root = Element("CDM")
    # Header
    _sub(root, "ORIGINATOR", originator)
    _sub(root, "MESSAGE_ID", cdm.message_id)
    _sub(root, "CREATION_DATE", cdm.creation_date.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z")
    # Relative metadata
    _sub(root, "TCA", cdm.tca_time.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z")
    _sub(root, "MISS_DISTANCE", f"{cdm.miss_distance_m:.4f}")
    _sub(root, "RELATIVE_SPEED", f"{cdm.relative_velocity_m_s:.4f}")
    if cdm.collision_probability is not None:
        _sub(root, "COLLISION_PROBABILITY", f"{cdm.collision_probability:.6e}")
    # Objects
    _add_object(root, cdm.object_1, "OBJECT1")
    _add_object(root, cdm.object_2, "OBJECT2")

    raw = tostring(root, encoding="unicode")
    pretty = minidom.parseString(raw).toprettyxml(indent="  ", encoding=None)
    # toprettyxml prepends <?xml ...?> — normalise to UTF-8 declaration
    lines = pretty.split("\n")
    if lines[0].startswith("<?xml"):
        lines[0] = '<?xml version="1.0" encoding="UTF-8"?>'
    return "\n".join(lines)


def parse_cdm_kvn(kvn_string: str) -> "ConjunctionDataMessage":
    """Parse a CCSDS KVN (Key-Value Notation) Conjunction Data Message.

    Supports the KVN subset of CDMs as specified in CCSDS 508.0-B-1.
    Each line has the form ``KEY = VALUE``. Object blocks are delimited
    by ``OBJECT = OBJECT1`` / ``OBJECT = OBJECT2`` keywords.

    Args:
        kvn_string: Raw KVN CDM text.

    Returns:
        ConjunctionDataMessage with parsed geometry and covariance.

    Raises:
        AstraError: If mandatory fields are missing or values are invalid.

    Example::

        with open("conjunction.kvn") as f:
            cdm = parse_cdm_kvn(f.read())
        print(cdm.tca_time, cdm.miss_distance_m)
    """
    logger.info("Parsing CCSDS Conjunction Data Message (KVN)...")

    # --- Keyword → storage map -------------------------------------------
    header: dict[str, str] = {}
    obj1_kv: dict[str, str] = {}
    obj2_kv: dict[str, str] = {}

    _COV_KEYS = [
        "CR_R", "CT_R", "CT_T", "CN_R", "CN_T", "CN_N",
        "CRDOT_R", "CRDOT_T", "CRDOT_N", "CRDOT_RDOT",
        "CTDOT_R", "CTDOT_T", "CTDOT_N", "CTDOT_RDOT", "CTDOT_TDOT",
        "CNDOT_R", "CNDOT_T", "CNDOT_N", "CNDOT_RDOT", "CNDOT_TDOT", "CNDOT_NDOT",
    ]

    current_target: dict[str, str] = header

    try:
        for raw_line in kvn_string.splitlines():
            line = raw_line.strip()
            if not line or line.startswith("COMMENT") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key = key.strip().upper()
            val = val.strip()

            # Route object-block fields to the correct dict
            if key == "OBJECT":
                if val.upper() == "OBJECT1":
                    current_target = obj1_kv
                elif val.upper() == "OBJECT2":
                    current_target = obj2_kv
                else:
                    current_target = header
                continue

            current_target[key] = val

        # --- Required header fields ---
        _required = ("TCA", "MISS_DISTANCE", "RELATIVE_SPEED", "MESSAGE_ID", "CREATION_DATE")
        missing = [k for k in _required if k not in header]
        if missing:
            raise AstraError(
                f"KVN CDM is missing required header fields: {missing}"
            )

        tca = _parse_time(header["TCA"])
        creation = _parse_time(header["CREATION_DATE"])
        miss_m = float(header["MISS_DISTANCE"])
        rel_vel = float(header["RELATIVE_SPEED"])
        pc_str = header.get("COLLISION_PROBABILITY", "")
        pc_val = float(pc_str) if pc_str else None

        # Physical validation
        validation_errors = []
        if miss_m < 0.0:
            validation_errors.append(f"Negative miss distance: {miss_m} m")
        if rel_vel < 0.0:
            validation_errors.append(f"Negative relative velocity: {rel_vel} m/s")
        if pc_val is not None and not (0.0 <= pc_val <= 1.0):
            validation_errors.append(f"Collision probability {pc_val} out of range [0, 1]")
        if validation_errors:
            raise AstraError(
                f"KVN CDM failed physical validation: " + "; ".join(validation_errors)
            )

        def _build_object(kv: dict[str, str], label: str) -> CDMObject:
            try:
                designator = kv.get("OBJECT_DESIGNATOR", "UNKNOWN")
                name = kv.get("OBJECT_NAME", "Unknown")
                x = float(kv.get("X", "0.0"))
                y = float(kv.get("Y", "0.0"))
                z = float(kv.get("Z", "0.0"))
                vx = float(kv.get("X_DOT", "0.0"))
                vy = float(kv.get("Y_DOT", "0.0"))
                vz = float(kv.get("Z_DOT", "0.0"))
                cov = [float(kv.get(tag, "0.0")) for tag in _COV_KEYS]
                return CDMObject(
                    object_designator=designator,
                    object_name=name,
                    position_xyz=(x, y, z),
                    velocity_xyz=(vx, vy, vz),
                    covariance_matrix=cov,
                )
            except (ValueError, KeyError) as exc:
                raise AstraError(
                    f"KVN CDM {label} block has invalid numeric field: {exc}"
                ) from exc

        obj1 = _build_object(obj1_kv, "OBJECT1")
        obj2 = _build_object(obj2_kv, "OBJECT2")

        return ConjunctionDataMessage(
            message_id=header["MESSAGE_ID"],
            creation_date=creation,
            tca_time=tca,
            miss_distance_m=miss_m,
            relative_velocity_m_s=rel_vel,
            collision_probability=pc_val,
            object_1=obj1,
            object_2=obj2,
        )

    except AstraError:
        raise
    except Exception as exc:
        raise AstraError(f"KVN CDM parsing failed: {exc}") from exc


def export_cdm_kvn(
    cdm: "ConjunctionDataMessage",
    originator: str = "ASTRA-CORE",
) -> str:
    """Serialise a :class:`ConjunctionDataMessage` to a CCSDS KVN CDM string.

    Produces a standards-compliant KVN document (CCSDS 508.0-B-1) that is
    interoperable with legacy mission-control systems (ODAS, older STK
    pipelines) that do not support XML CDMs.  Completes the read-write API
    symmetry: ``parse_cdm_kvn(export_cdm_kvn(cdm))`` round-trips losslessly.
    Args:
        cdm: The :class:`ConjunctionDataMessage` to serialise.
        originator: Originator identifier written into the KVN header.

    Returns:
        A KVN string conforming to CCSDS 508.0-B-1 CDM schema.

    Example::

        kvn_str = export_cdm_kvn(event_cdm, originator="ISRO-ISTRAC")
        with open("conjunction.kvn", "w") as f:
            f.write(kvn_str)
    """
    _COV_KEYS = [
        "CR_R", "CT_R", "CT_T", "CN_R", "CN_T", "CN_N",
        "CRDOT_R", "CRDOT_T", "CRDOT_N", "CRDOT_RDOT",
        "CTDOT_R", "CTDOT_T", "CTDOT_N", "CTDOT_RDOT", "CTDOT_TDOT",
        "CNDOT_R", "CNDOT_T", "CNDOT_N", "CNDOT_RDOT", "CNDOT_TDOT", "CNDOT_NDOT",
    ]

    def _ts(dt: "datetime") -> str:
        return dt.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"

    def _obj_block(obj: CDMObject, label: str) -> list[str]:
        lines: list[str] = []
        lines.append(f"OBJECT                           = {label}")
        lines.append(f"OBJECT_DESIGNATOR                = {obj.object_designator}")
        lines.append(f"OBJECT_NAME                      = {obj.object_name}")
        x, y, z = obj.position_xyz
        vx, vy, vz = obj.velocity_xyz
        lines.append(f"X                                = {x:.6f}")
        lines.append(f"Y                                = {y:.6f}")
        lines.append(f"Z                                = {z:.6f}")
        lines.append(f"X_DOT                            = {vx:.9f}")
        lines.append(f"Y_DOT                            = {vy:.9f}")
        lines.append(f"Z_DOT                            = {vz:.9f}")
        for tag, val in zip(_COV_KEYS, obj.covariance_matrix):
            lines.append(f"{tag:<32} = {val:.6e}")
        return lines

    rows: list[str] = []
    rows.append("CCSDS_CDM_VERS                   = 1.0")
    rows.append(f"CREATION_DATE                    = {_ts(cdm.creation_date)}")
    rows.append(f"ORIGINATOR                       = {originator}")
    rows.append(f"MESSAGE_ID                       = {cdm.message_id}")
    rows.append(f"TCA                              = {_ts(cdm.tca_time)}")
    rows.append(f"MISS_DISTANCE                    = {cdm.miss_distance_m:.4f}")
    rows.append(f"RELATIVE_SPEED                   = {cdm.relative_velocity_m_s:.4f}")
    if cdm.collision_probability is not None:
        rows.append(f"COLLISION_PROBABILITY            = {cdm.collision_probability:.6e}")
    rows.extend(_obj_block(cdm.object_1, "OBJECT1"))
    rows.extend(_obj_block(cdm.object_2, "OBJECT2"))

    return "\n".join(rows) + "\n"

