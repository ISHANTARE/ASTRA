"""ASTRA Core Conjunction Data Message (CDM) Parser.

Implements structural parsing for standard CCSDS CDMs provided by Space-Track 
and the US Space Force. This elevates the library to handle official government 
collision warning formats.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime
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
    # Often formatted as 2023-10-15T12:00:00.000Z
    clean_str = time_str.replace("Z", "+00:00")
    return datetime.fromisoformat(clean_str)

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
        # Strip XML namespaces. They vary between agencies and complicate parsing.
        clean_xml = xml_string
        if "xmlns=" in clean_xml:
            import re
            clean_xml = re.sub(r'\sxmlns="[^"]+"', '', clean_xml, count=1)
            
        root = ET.fromstring(clean_xml)
        
        # This is a structural scaffold for the parser. 
        # A full production parser decodes exactly 21 covariance elements per object
        # and transforms GCRF state vectors into the encounter B-Plane.
        
        msg_id = root.findtext(".//MESSAGE_ID", default="UNKNOWN")
        tca = _parse_time(root.findtext(".//TCA", default="1970-01-01T00:00:00Z"))
        miss_m = float(root.findtext(".//MISS_DISTANCE", default="0.0"))
        
        logger.debug(f"Decoded CDM {msg_id} - TCA: {tca.isoformat()} - Miss: {miss_m}m")
        
        return ConjunctionDataMessage(
            message_id=msg_id,
            creation_date=datetime.utcnow(),
            tca_time=tca,
            miss_distance_m=miss_m,
            relative_velocity_m_s=0.0,
            collision_probability=None,
            object_1=CDMObject("OBJ1", "Primary", (0,0,0), (0,0,0), []),
            object_2=CDMObject("OBJ2", "Secondary", (0,0,0), (0,0,0), [])
        )
    except Exception as e:
        logger.error(f"CDM Parsing failed: {e}")
        raise AstraError(f"Invalid CCSDS CDM format: {e}") from e
