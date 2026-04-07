"""Tests for CCSDS CDM XML parsing (defusedxml hardening)."""

from __future__ import annotations

import pytest

from astra.cdm import parse_cdm_xml
from astra.errors import AstraError


def _minimal_cdm_xml() -> str:
    """Minimal valid CDM-like XML with required tags for parse_cdm_xml."""
    z = "0.0"
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<cdm>
  <MESSAGE_ID>TEST-CDM-001</MESSAGE_ID>
  <CREATION_DATE>2021-06-01T00:00:00Z</CREATION_DATE>
  <TCA>2021-06-02T12:00:00.000Z</TCA>
  <MISS_DISTANCE>250.5</MISS_DISTANCE>
  <RELATIVE_SPEED>14750.0</RELATIVE_SPEED>
  <COLLISION_PROBABILITY>4.7E-4</COLLISION_PROBABILITY>
  <OBJECT1_OBJECT_DESIGNATOR>25544</OBJECT1_OBJECT_DESIGNATOR>
  <OBJECT1_OBJECT_NAME>OBJ-A</OBJECT1_OBJECT_NAME>
  <OBJECT1_X>7000.0</OBJECT1_X>
  <OBJECT1_Y>0.0</OBJECT1_Y>
  <OBJECT1_Z>0.0</OBJECT1_Z>
  <OBJECT1_X_DOT>0.0</OBJECT1_X_DOT>
  <OBJECT1_Y_DOT>7.5</OBJECT1_Y_DOT>
  <OBJECT1_Z_DOT>0.0</OBJECT1_Z_DOT>
  <OBJECT2_OBJECT_DESIGNATOR>99999</OBJECT2_OBJECT_DESIGNATOR>
  <OBJECT2_OBJECT_NAME>OBJ-B</OBJECT2_OBJECT_NAME>
  <OBJECT2_X>7000.1</OBJECT2_X>
  <OBJECT2_Y>0.0</OBJECT2_Y>
  <OBJECT2_Z>0.0</OBJECT2_Z>
  <OBJECT2_X_DOT>0.0</OBJECT2_X_DOT>
  <OBJECT2_Y_DOT>7.4</OBJECT2_Y_DOT>
  <OBJECT2_Z_DOT>0.0</OBJECT2_Z_DOT>
  <CR_R>{z}</CR_R><CT_R>{z}</CT_R><CT_T>{z}</CT_T><CN_R>{z}</CN_R><CN_T>{z}</CN_T><CN_N>{z}</CN_N>
  <CRDOT_R>{z}</CRDOT_R><CRDOT_T>{z}</CRDOT_T><CRDOT_N>{z}</CRDOT_N><CRDOT_RDOT>{z}</CRDOT_RDOT>
  <CTDOT_R>{z}</CTDOT_R><CTDOT_T>{z}</CTDOT_T><CTDOT_N>{z}</CTDOT_N><CTDOT_RDOT>{z}</CTDOT_RDOT>
  <CTDOT_TDOT>{z}</CTDOT_TDOT>
  <CNDOT_R>{z}</CNDOT_R><CNDOT_T>{z}</CNDOT_T><CNDOT_N>{z}</CNDOT_N><CNDOT_RDOT>{z}</CNDOT_RDOT>
  <CNDOT_TDOT>{z}</CNDOT_TDOT><CNDOT_NDOT>{z}</CNDOT_NDOT>
</cdm>
"""


def test_parse_cdm_xml_minimal_roundtrip():
    cdm = parse_cdm_xml(_minimal_cdm_xml())
    assert cdm.message_id == "TEST-CDM-001"
    assert cdm.miss_distance_m == pytest.approx(250.5)
    assert cdm.relative_velocity_m_s == pytest.approx(14750.0)
    assert cdm.collision_probability == pytest.approx(4.7e-4)
    assert cdm.object_1.object_designator == "25544"
    assert cdm.object_2.object_designator == "99999"


def test_parse_cdm_xml_multiscript_namespace_stripped():
    """Prefixed tags and xmlns stripping still yield parsed fields."""
    inner = _minimal_cdm_xml().replace(
        "<cdm>", '<a:cdm xmlns:a="urn:test">'
    ).replace("</cdm>", "</a:cdm>")
    cdm = parse_cdm_xml(inner)
    assert cdm.message_id == "TEST-CDM-001"


def test_parse_cdm_xml_malformed_raises_astra_error():
    with pytest.raises(AstraError, match="Invalid CCSDS CDM"):
        parse_cdm_xml("<not-xml")


def test_parse_cdm_xml_entity_expansion_rejected():
    """External entities are not expanded (defusedxml)."""
    from defusedxml.common import EntitiesForbidden

    evil = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE foo [<!ENTITY xxe SYSTEM "file:///etc/passwd">]>
<cdm><MESSAGE_ID>&xxe;</MESSAGE_ID></cdm>
"""
    with pytest.raises((EntitiesForbidden, AstraError)):
        parse_cdm_xml(evil)
