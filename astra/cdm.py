"""ASTRA Core Conjunction Data Message (CDM) Parser.

Implements structural parsing for standard CCSDS CDMs provided by Space-Track 
and the US Space Force. This elevates the library to handle official government 
collision warning formats.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from astra.errors import AstraError
from astra.log import get_logger

logger = get_logger(__name__)

@dataclass
class CDMObject:
    """Represents a single object inside a CDM (Object1 or Object2)."""
    object_designator: str
    object_name: str
    position_xyz: tuple[float, float, float]  # km in J2000/GCRF
    velocity_xyz: tuple[float, float, float]  # km/s in J2000/GCRF
    covariance_matrix: list[float]            # 21-element upper triangular RTN covariance

@dataclass
class ConjunctionDataMessage:
    """Represents the complete payload of a CCSDS XML CDM."""
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
    return datetime.fromisoformat(clean_str)

def _findtext(element: ET.Element, tag: str, default: str = "") -> str:
    """Search for a tag at any depth within the XML tree."""
    result = element.findtext(f".//{tag}", default=default)
    return result

def _parse_cdm_object(root: ET.Element, prefix: str) -> CDMObject:
    """Parse a single CDM object (OBJECT1 or OBJECT2) from the XML tree.
    
    Extracts the object designator, name, state vector, and covariance matrix.
    """
    designator = _findtext(root, f"{prefix}_OBJECT_DESIGNATOR", "UNKNOWN")
    if not designator or designator == "UNKNOWN":
        designator = _findtext(root, "OBJECT_DESIGNATOR", "UNKNOWN")
    
    name = _findtext(root, f"{prefix}_OBJECT_NAME", "Unknown")
    if not name or name == "Unknown":
        name = _findtext(root, "OBJECT_NAME", "Unknown")
    
    # State vector (km, km/s)
    x = float(_findtext(root, f"{prefix}_X", "0.0"))
    y = float(_findtext(root, f"{prefix}_Y", "0.0"))
    z = float(_findtext(root, f"{prefix}_Z", "0.0"))
    vx = float(_findtext(root, f"{prefix}_X_DOT", "0.0"))
    vy = float(_findtext(root, f"{prefix}_Y_DOT", "0.0"))
    vz = float(_findtext(root, f"{prefix}_Z_DOT", "0.0"))
    
    # RTN Covariance (21 elements, upper-triangular row-major)
    cov_tags = [
        "CR_R", "CT_R", "CT_T", "CN_R", "CN_T", "CN_N",
        "CRDOT_R", "CRDOT_T", "CRDOT_N", "CRDOT_RDOT",
        "CTDOT_R", "CTDOT_T", "CTDOT_N", "CTDOT_RDOT", "CTDOT_TDOT",
        "CNDOT_R", "CNDOT_T", "CNDOT_N", "CNDOT_RDOT", "CNDOT_TDOT", "CNDOT_NDOT",
    ]
    cov = []
    for tag in cov_tags:
        val_str = _findtext(root, tag, "0.0")
        cov.append(float(val_str))
    
    return CDMObject(
        object_designator=designator,
        object_name=name,
        position_xyz=(x, y, z),
        velocity_xyz=(vx, vy, vz),
        covariance_matrix=cov,
    )

def parse_cdm_xml(xml_string: str) -> ConjunctionDataMessage:
    """Parses a standard CCSDS XML Conjunction Data Message.
    
    Extracts all fields including object state vectors and RTN covariance 
    matrices from the standard CCSDS CDM format.
    
    Args:
        xml_string: Raw XML response from Space-Track or local CDM file.
        
    Returns:
        ConjunctionDataMessage object containing structured geometry and covariance.
        
    Raises:
        AstraError: If the XML format is invalid or missing critical CDM tags.
    """
    logger.info("Parsing CCSDS Conjunction Data Message (XML)...")
    try:
        # Strip XML namespaces. They vary between agencies and complicate parsing.
        clean_xml = xml_string
        if "xmlns=" in clean_xml:
            import re
            clean_xml = re.sub(r'\sxmlns="[^"]+"', '', clean_xml, count=1)
            
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
        
        tca = _parse_time(tca_str)
        creation = _parse_time(creation_str)
        
        logger.debug(f"Decoded CDM {msg_id} - TCA: {tca.isoformat()} - Miss: {miss_m}m")
        
        return ConjunctionDataMessage(
            message_id=msg_id,
            creation_date=creation,
            tca_time=tca,
            miss_distance_m=miss_m,
            relative_velocity_m_s=rel_vel,
            collision_probability=pc_val,
            object_1=obj1,
            object_2=obj2,
        )
    except Exception as e:
        logger.error(f"CDM Parsing failed: {e}")
        raise AstraError(f"Invalid CCSDS CDM format: {e}") from e
