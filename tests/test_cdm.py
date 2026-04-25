"""tests/test_cdm.py — Round-trip and correctness tests for CDM parser + exporter.

[FM-4 Fix — Finding #13]
Tests the parse/export symmetry: parse_cdm_xml(export_cdm_xml(cdm)) round-trips
cleanly. Also validates physical field constraints and XML structure.
"""
import math
from datetime import datetime, timezone

import pytest

from astra.cdm import (
    CDMObject,
    ConjunctionDataMessage,
    export_cdm_xml,
    parse_cdm_xml,
)


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------

def _make_cov21():
    """Return a 21-element upper-triangular covariance (small but non-zero)."""
    return [1e-4 * (i + 1) for i in range(21)]


def _make_cdm() -> ConjunctionDataMessage:
    obj1 = CDMObject(
        object_designator="25544",
        object_name="ISS (ZARYA)",
        position_xyz=(6778.137, 0.0, 0.0),
        velocity_xyz=(0.0, 7.66, 0.0),
        covariance_matrix=_make_cov21(),
    )
    obj2 = CDMObject(
        object_designator="48274",
        object_name="DEBRIS-A",
        position_xyz=(6778.500, 0.005, 0.002),
        velocity_xyz=(0.0, 7.65, 0.0),
        covariance_matrix=_make_cov21(),
    )
    return ConjunctionDataMessage(
        message_id="AOS-2026-001-001",
        creation_date=datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
        tca_time=datetime(2026, 1, 2, 6, 30, 0, tzinfo=timezone.utc),
        miss_distance_m=520.0,
        relative_velocity_m_s=14500.0,
        collision_probability=1.23e-5,
        object_1=obj1,
        object_2=obj2,
    )


# ---------------------------------------------------------------------------
# export_cdm_xml
# ---------------------------------------------------------------------------

class TestExportCdmXml:
    def test_returns_string(self):
        cdm = _make_cdm()
        xml = export_cdm_xml(cdm)
        assert isinstance(xml, str), "export_cdm_xml must return a string"

    def test_xml_has_cdm_root(self):
        cdm = _make_cdm()
        xml = export_cdm_xml(cdm)
        assert "<CDM>" in xml, "XML must contain a <CDM> root element"

    def test_xml_contains_message_id(self):
        cdm = _make_cdm()
        xml = export_cdm_xml(cdm)
        assert "AOS-2026-001-001" in xml, "XML must contain the MESSAGE_ID"

    def test_xml_contains_tca(self):
        cdm = _make_cdm()
        xml = export_cdm_xml(cdm)
        assert "2026-01-02" in xml, "XML must contain TCA date"

    def test_xml_contains_miss_distance(self):
        cdm = _make_cdm()
        xml = export_cdm_xml(cdm)
        assert "520" in xml, "XML must contain MISS_DISTANCE value"

    def test_xml_contains_collision_probability(self):
        cdm = _make_cdm()
        xml = export_cdm_xml(cdm)
        assert "COLLISION_PROBABILITY" in xml
        assert "1.23" in xml, "Collision probability mantissa must appear in XML"

    def test_xml_omits_pc_when_none(self):
        cdm = _make_cdm()
        # Rebuild with no Pc
        cdm2 = ConjunctionDataMessage(
            message_id=cdm.message_id,
            creation_date=cdm.creation_date,
            tca_time=cdm.tca_time,
            miss_distance_m=cdm.miss_distance_m,
            relative_velocity_m_s=cdm.relative_velocity_m_s,
            collision_probability=None,
            object_1=cdm.object_1,
            object_2=cdm.object_2,
        )
        xml = export_cdm_xml(cdm2)
        assert "COLLISION_PROBABILITY" not in xml, (
            "COLLISION_PROBABILITY tag must be absent when Pc is None"
        )

    def test_originator_written(self):
        cdm = _make_cdm()
        xml = export_cdm_xml(cdm, originator="ISRO-ISTRAC")
        assert "ISRO-ISTRAC" in xml

    def test_object_designators_present(self):
        cdm = _make_cdm()
        xml = export_cdm_xml(cdm)
        assert "25544" in xml
        assert "48274" in xml

    def test_covariance_elements_present(self):
        cdm = _make_cdm()
        xml = export_cdm_xml(cdm)
        assert "CR_R" in xml, "Covariance tag CR_R must be in XML"
        assert "CNDOT_NDOT" in xml, "Covariance tag CNDOT_NDOT must be in XML"


# ---------------------------------------------------------------------------
# Round-trip symmetry: parse(export(cdm)) ≈ cdm
# ---------------------------------------------------------------------------

class TestRoundTrip:
    def test_message_id_preserved(self):
        original = _make_cdm()
        xml = export_cdm_xml(original)
        parsed = parse_cdm_xml(xml)
        assert parsed.message_id == original.message_id

    def test_miss_distance_preserved(self):
        original = _make_cdm()
        xml = export_cdm_xml(original)
        parsed = parse_cdm_xml(xml)
        assert abs(parsed.miss_distance_m - original.miss_distance_m) < 0.01, (
            f"miss_distance_m: expected {original.miss_distance_m}, "
            f"got {parsed.miss_distance_m}"
        )

    def test_collision_probability_preserved(self):
        original = _make_cdm()
        xml = export_cdm_xml(original)
        parsed = parse_cdm_xml(xml)
        assert parsed.collision_probability is not None
        rel_err = abs(parsed.collision_probability - original.collision_probability) \
                  / original.collision_probability
        assert rel_err < 1e-4, (
            f"Pc relative error {rel_err:.2e} exceeds 1e-4"
        )

    def test_object1_position_preserved(self):
        original = _make_cdm()
        xml = export_cdm_xml(original)
        parsed = parse_cdm_xml(xml)
        for i, (o, p) in enumerate(
            zip(original.object_1.position_xyz, parsed.object_1.position_xyz)
        ):
            assert abs(o - p) < 1e-3, (
                f"position_xyz[{i}]: expected {o}, got {p}"
            )

    def test_relative_velocity_preserved(self):
        original = _make_cdm()
        xml = export_cdm_xml(original)
        parsed = parse_cdm_xml(xml)
        assert abs(parsed.relative_velocity_m_s - original.relative_velocity_m_s) < 0.1

    def test_tca_time_preserved(self):
        original = _make_cdm()
        xml = export_cdm_xml(original)
        parsed = parse_cdm_xml(xml)
        delta_s = abs((parsed.tca_time - original.tca_time).total_seconds())
        assert delta_s < 1.0, f"TCA time shifted by {delta_s:.1f} s after round-trip"

    def test_covariance_matrix_preserved(self):
        original = _make_cdm()
        xml = export_cdm_xml(original)
        parsed = parse_cdm_xml(xml)
        for i, (o, p) in enumerate(
            zip(original.object_1.covariance_matrix, parsed.object_1.covariance_matrix)
        ):
            rel_err = abs(o - p) / max(abs(o), 1e-30)
            assert rel_err < 1e-4, (
                f"covariance_matrix[{i}]: expected {o:.6e}, got {p:.6e} (rel_err={rel_err:.2e})"
            )
