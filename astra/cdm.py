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
