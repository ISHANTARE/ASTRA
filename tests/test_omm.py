# tests/test_omm.py
"""Comprehensive test suite for the ASTRA-Core OMM integration.

Tests cover:
    1. JSON parsing correctness (unit conversions, field extraction)
    2. End-to-end SGP4 propagation via SatelliteOMM
    3. Physical sanity checks on propagated positions
    4. make_debris_object() with an OMM source
    5. Resilient handling of malformed/missing fields
    6. Backwards compatibility -- TLEs still work unchanged
    7. load_omm_file() — local file ingestion
    8. validate_omm() — pre-parse sanity checking
"""

import json
import math
import pytest
import numpy as np

import astra
from astra.models import SatelliteTLE, SatelliteOMM, DebrisObject
from astra.omm import parse_omm_record, parse_omm_json
from astra.orbit import propagate_trajectory, propagate_orbit
from astra.debris import make_debris_object
from astra.errors import AstraError, InvalidTLEError

# ---------------------------------------------------------------------------
# Fixtures -- realistic OMM data matching a known ISS-like LEO orbit
# ---------------------------------------------------------------------------

# This represents the ISS in OMM JSON format.
# Values deliberately chosen to be physically consistent with a ~410 km LEO orbit.
ISS_OMM_RECORD = {
    "OBJECT_NAME": "ISS (ZARYA)",
    "OBJECT_ID": "1998-067A",
    "NORAD_CAT_ID": "25544",
    "OBJECT_TYPE": "PAYLOAD",
    "EPOCH": "2021-01-01T00:00:00.000000",
    "MEAN_MOTION": "15.48922536",  # rev/day — real ISS value
    "ECCENTRICITY": ".0001364",
    "INCLINATION": "51.6442",  # degrees
    "RA_OF_ASC_NODE": "284.1199",  # degrees
    "ARG_OF_PERICENTER": "338.5498",  # degrees
    "MEAN_ANOMALY": "21.5664",  # degrees
    "BSTAR": ".34282E-4",
    "RCS_SIZE": "LARGE",
    "MASS": "419725",
}

ISS_OMM_JSON = json.dumps([ISS_OMM_RECORD])


@pytest.fixture
def iss_omm() -> SatelliteOMM:
    return parse_omm_record(ISS_OMM_RECORD)


@pytest.fixture
def iss_tle() -> SatelliteTLE:
    """Equivalent ISS TLE for cross-format comparison."""
    line1 = "1 25544U 98067A   21001.00000000  .00001480  00000-0  34282-4 0  9990"
    line2 = "2 25544  51.6442 284.1199 0001364 338.5498  21.5664 15.48922536 12341"
    return SatelliteTLE.from_strings(line1, line2, name="ISS (ZARYA)")


# ===========================================================================
# 1. PARSING TESTS — Unit conversion correctness
# ===========================================================================


class TestOMMParsing:

    def test_norad_id_parsed(self, iss_omm):
        assert iss_omm.norad_id == "25544"

    def test_name_parsed(self, iss_omm):
        assert iss_omm.name == "ISS (ZARYA)"

    def test_object_type_normalized(self, iss_omm):
        assert iss_omm.object_type == "PAYLOAD"

    def test_epoch_jd_is_float(self, iss_omm):
        # JD for 2021-01-01T00:00:00 UTC ≈ 2459215.5
        assert isinstance(iss_omm.epoch_jd, float)
        assert abs(iss_omm.epoch_jd - 2459215.5) < 1.0  # within 1 day tolerance

    def test_inclination_converted_to_radians(self, iss_omm):
        expected_rad = math.radians(51.6442)
        assert abs(iss_omm.inclination_rad - expected_rad) < 1e-9

    def test_raan_converted_to_radians(self, iss_omm):
        expected_rad = math.radians(284.1199)
        assert abs(iss_omm.raan_rad - expected_rad) < 1e-9

    def test_argpo_converted_to_radians(self, iss_omm):
        expected_rad = math.radians(338.5498)
        assert abs(iss_omm.argpo_rad - expected_rad) < 1e-9

    def test_mean_anomaly_converted_to_radians(self, iss_omm):
        expected_rad = math.radians(21.5664)
        assert abs(iss_omm.mo_rad - expected_rad) < 1e-9

    def test_mean_motion_converted_to_rad_per_min(self, iss_omm):
        # 15.48922536 rev/day → × 2π / 1440 min/day
        expected = 15.48922536 * (2.0 * math.pi) / 1440.0
        assert abs(iss_omm.mean_motion_rad_min - expected) < 1e-12

    def test_eccentricity_parsed(self, iss_omm):
        assert abs(iss_omm.eccentricity - 0.0001364) < 1e-9

    def test_bstar_parsed(self, iss_omm):
        assert abs(iss_omm.bstar - 3.4282e-5) < 1e-12

    def test_rcs_large_mapped_correctly(self, iss_omm):
        # "LARGE" RCS → 10.0 m²
        assert iss_omm.rcs_m2 == 10.0

    def test_mass_parsed(self, iss_omm):
        assert iss_omm.mass_kg == 419725.0


# ===========================================================================
# 2. JSON BULK PARSER TESTS
# ===========================================================================


class TestBulkOMMParser:

    def test_parse_list_of_one(self):
        results = parse_omm_json(ISS_OMM_JSON)
        assert len(results) == 1
        assert isinstance(results[0], SatelliteOMM)

    def test_parse_returns_correct_norad_id(self):
        results = parse_omm_json(ISS_OMM_JSON)
        assert results[0].norad_id == "25544"

    def test_empty_array_returns_empty_list(self):
        results = parse_omm_json("[]")
        assert results == []

    def test_malformed_record_is_skipped_not_fatal(self):
        bad_record = {
            "OBJECT_NAME": "BROKEN",
            "NORAD_CAT_ID": "99999",
        }  # missing all orbital elements
        payload = json.dumps([ISS_OMM_RECORD, bad_record])
        results = parse_omm_json(payload)
        # Only the valid record should survive
        assert len(results) == 1
        assert results[0].norad_id == "25544"

    def test_invalid_json_raises_astra_error(self):
        with pytest.raises(AstraError):
            parse_omm_json("this is not json")

    def test_non_array_json_raises_astra_error(self):
        with pytest.raises(AstraError):
            parse_omm_json(json.dumps({"single": "object"}))

    def test_missing_required_field_raises_on_single_parse(self):
        bad = dict(ISS_OMM_RECORD)
        del bad["INCLINATION"]
        with pytest.raises(InvalidTLEError):
            parse_omm_record(bad)

    def test_missing_epoch_raises_on_single_parse(self):
        bad = dict(ISS_OMM_RECORD)
        del bad["EPOCH"]
        with pytest.raises((AstraError, InvalidTLEError)):
            parse_omm_record(bad)


# ===========================================================================
# 3. SGP4 PROPAGATION — Physical Sanity Checks
# ===========================================================================

EARTH_RADIUS_KM = 6378.137


class TestOMMPropagation:

    def test_propagate_orbit_returns_orbital_state(self, iss_omm):
        state = propagate_orbit(iss_omm, iss_omm.epoch_jd, 0.0)
        assert state.error_code == 0

    def test_propagated_position_is_near_earth(self, iss_omm):
        """Position magnitude must be within plausible Earth-orbit range."""
        state = propagate_orbit(iss_omm, iss_omm.epoch_jd, 0.0)
        r_km = float(np.linalg.norm(state.position_km))
        altitude_km = r_km - EARTH_RADIUS_KM
        assert (
            200.0 <= altitude_km <= 600.0
        ), f"Unexpected altitude: {altitude_km:.1f} km"

    def test_propagated_velocity_is_plausible_for_leo(self, iss_omm):
        """LEO orbital speed should be between 7 and 8 km/s."""
        state = propagate_orbit(iss_omm, iss_omm.epoch_jd, 0.0)
        v_km_s = float(np.linalg.norm(state.velocity_km_s))
        assert 7.0 <= v_km_s <= 8.0, f"Unexpected velocity: {v_km_s:.3f} km/s"

    def test_propagate_trajectory_returns_correct_shape(self, iss_omm):
        times, positions, velocities = propagate_trajectory(
            iss_omm, iss_omm.epoch_jd, iss_omm.epoch_jd + 1.0 / 24.0, step_minutes=5.0
        )
        assert positions.shape[1] == 3
        assert len(times) == positions.shape[0]
        assert len(times) > 0
        assert velocities.shape == positions.shape

    def test_omm_and_tle_propagate_to_similar_position(self, iss_omm, iss_tle):
        """Cross-validate: OMM and equivalent TLE should agree within 100 km at epoch."""
        state_omm = propagate_orbit(iss_omm, iss_omm.epoch_jd, 0.0)
        state_tle = propagate_orbit(iss_tle, iss_tle.epoch_jd, 0.0)

        r_omm = np.array(state_omm.position_km)
        r_tle = np.array(state_tle.position_km)
        delta_km = float(np.linalg.norm(r_omm - r_tle))
        # Small BSTAR string parsing differences may introduce minor deltas
        assert (
            delta_km < 100.0
        ), f"TLE vs OMM position delta too large: {delta_km:.1f} km"

    def test_no_nan_in_trajectory(self, iss_omm):
        _, positions, _vel = propagate_trajectory(
            iss_omm, iss_omm.epoch_jd, iss_omm.epoch_jd + 1.0, step_minutes=5.0
        )
        assert not np.any(np.isnan(positions))


# ===========================================================================
# 4. DEBRIS OBJECT INTEGRATION
# ===========================================================================


class TestOMMDebrisObject:

    def test_make_debris_object_from_omm(self, iss_omm):
        obj = make_debris_object(iss_omm)
        assert isinstance(obj, DebrisObject)

    def test_debris_source_is_satellite_omm(self, iss_omm):
        obj = make_debris_object(iss_omm)
        assert isinstance(obj.source, SatelliteOMM)

    def test_debris_altitude_is_plausible(self, iss_omm):
        obj = make_debris_object(iss_omm)
        assert 200.0 <= obj.altitude_km <= 600.0

    def test_debris_inclination_is_correct(self, iss_omm):
        obj = make_debris_object(iss_omm)
        assert abs(obj.inclination_deg - 51.6442) < 0.01

    def test_debris_rcs_harvested_from_omm(self, iss_omm):
        obj = make_debris_object(iss_omm)
        assert obj.rcs_m2 == 10.0  # "LARGE" → 10.0 m²

    def test_debris_tle_shim_raises_for_omm_source(self, iss_omm):
        """Backwards-compat shim must raise AttributeError for OMM-backed objects."""
        obj = make_debris_object(iss_omm)
        with pytest.raises(AttributeError):
            _ = obj.tle


# ===========================================================================
# 5. BACKWARDS COMPATIBILITY — TLE path still works
# ===========================================================================


class TestTLEBackwardsCompatibility:

    def test_tle_debris_object_still_uses_source(self, iss_tle):
        obj = make_debris_object(iss_tle)
        assert isinstance(obj.source, SatelliteTLE)

    def test_tle_shim_still_works(self, iss_tle):
        obj = make_debris_object(iss_tle)
        # .tle shim must return the TLE without error for legacy consumers
        assert obj.tle is iss_tle

    def test_tle_propagation_unchanged(self, iss_tle):
        state = propagate_orbit(iss_tle, iss_tle.epoch_jd, 0.0)
        assert state.error_code == 0
        r_km = float(np.linalg.norm(state.position_km))
        assert r_km > 6000.0


# ===========================================================================
# 6. SATELLITE STATE UNION TYPE
# ===========================================================================


class TestSatelliteStateUnion:

    def test_omm_is_valid_satellite_state(self, iss_omm):
        # SatelliteState is a Union — isinstance checks should pass
        assert isinstance(iss_omm, (SatelliteTLE, SatelliteOMM))

    def test_tle_is_valid_satellite_state(self, iss_tle):
        assert isinstance(iss_tle, (SatelliteTLE, SatelliteOMM))

    def test_top_level_exports_available(self):
        assert hasattr(astra, "SatelliteOMM")
        assert hasattr(astra, "SatelliteState")
        assert hasattr(astra, "parse_omm_json")
        assert hasattr(astra, "parse_omm_record")
        assert hasattr(astra, "load_omm_file")
        assert hasattr(astra, "validate_omm")


# ===========================================================================
# 7. load_omm_file() — Local File Ingestion
# ===========================================================================


class TestLoadOMMFile:

    def test_load_valid_json_file(self, tmp_path):
        f = tmp_path / "catalog.json"
        f.write_text(ISS_OMM_JSON, encoding="utf-8")
        results = astra.load_omm_file(str(f))
        assert len(results) == 1
        assert isinstance(results[0], SatelliteOMM)
        assert results[0].norad_id == "25544"

    def test_load_file_with_multiple_records(self, tmp_path):
        record2 = dict(ISS_OMM_RECORD)
        record2["NORAD_CAT_ID"] = "44235"
        record2["OBJECT_NAME"] = "STARLINK-1"
        payload = json.dumps([ISS_OMM_RECORD, record2])
        f = tmp_path / "two_sats.json"
        f.write_text(payload, encoding="utf-8")
        results = astra.load_omm_file(str(f))
        assert len(results) == 2

    def test_nonexistent_file_raises_astra_error(self):
        with pytest.raises(AstraError) as exc_info:
            astra.load_omm_file("definitely_does_not_exist.json")
        assert "not found" in str(exc_info.value).lower()

    def test_invalid_json_file_raises_astra_error(self, tmp_path):
        f = tmp_path / "bad.json"
        f.write_text("this is not json", encoding="utf-8")
        with pytest.raises(AstraError):
            astra.load_omm_file(str(f))

    def test_load_empty_array_returns_empty_list(self, tmp_path):
        f = tmp_path / "empty.json"
        f.write_text("[]", encoding="utf-8")
        results = astra.load_omm_file(str(f))
        assert results == []

    def test_loaded_omm_is_propagatable(self, tmp_path):
        f = tmp_path / "iss.json"
        f.write_text(ISS_OMM_JSON, encoding="utf-8")
        results = astra.load_omm_file(str(f))
        omm = results[0]
        state = astra.propagate_orbit(omm, omm.epoch_jd, 0.0)
        assert state.error_code == 0


# ===========================================================================
# 8. validate_omm() — Pre-parse Sanity Checking
# ===========================================================================


class TestValidateOMM:

    def test_valid_full_record_returns_true(self):
        assert astra.validate_omm(ISS_OMM_RECORD) is True

    def test_missing_inclination_returns_false(self):
        bad = dict(ISS_OMM_RECORD)
        del bad["INCLINATION"]
        assert astra.validate_omm(bad) is False

    def test_missing_mean_motion_returns_false(self):
        bad = dict(ISS_OMM_RECORD)
        del bad["MEAN_MOTION"]
        assert astra.validate_omm(bad) is False

    def test_missing_epoch_returns_false(self):
        bad = dict(ISS_OMM_RECORD)
        del bad["EPOCH"]
        assert astra.validate_omm(bad) is False

    def test_eccentricity_out_of_range_returns_false(self):
        bad = dict(ISS_OMM_RECORD)
        bad["ECCENTRICITY"] = "1.5"  # must be < 1
        assert astra.validate_omm(bad) is False

    def test_negative_eccentricity_returns_false(self):
        bad = dict(ISS_OMM_RECORD)
        bad["ECCENTRICITY"] = "-0.1"
        assert astra.validate_omm(bad) is False

    def test_zero_mean_motion_returns_false(self):
        bad = dict(ISS_OMM_RECORD)
        bad["MEAN_MOTION"] = "0.0"
        assert astra.validate_omm(bad) is False

    def test_negative_mean_motion_returns_false(self):
        bad = dict(ISS_OMM_RECORD)
        bad["MEAN_MOTION"] = "-5.0"
        assert astra.validate_omm(bad) is False

    def test_inclination_over_180_returns_false(self):
        bad = dict(ISS_OMM_RECORD)
        bad["INCLINATION"] = "200.0"
        assert astra.validate_omm(bad) is False

    def test_unparseable_epoch_returns_false(self):
        bad = dict(ISS_OMM_RECORD)
        bad["EPOCH"] = "not-a-date"
        assert astra.validate_omm(bad) is False

    def test_empty_dict_returns_false(self):
        assert astra.validate_omm({}) is False

    def test_filter_with_validate_omm_works(self):
        """Real-world pattern: pre-filter a batch before parsing."""
        bad = {"OBJECT_NAME": "BROKEN", "NORAD_CAT_ID": "99999"}
        records = [ISS_OMM_RECORD, bad]
        valid = [r for r in records if astra.validate_omm(r)]
        assert len(valid) == 1
        assert valid[0]["NORAD_CAT_ID"] == "25544"
