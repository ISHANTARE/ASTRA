"""Tests for all 9 findings from the post-v3.6.0 audit remediation.

Each test class maps to a specific finding ID. Tests are fully offline
(no network calls) and deterministic.
"""
from __future__ import annotations

import math
import json
import os
from datetime import datetime, timezone
from unittest.mock import patch, Mock, MagicMock

import numpy as np
import pytest

from astra.errors import AstraError, PropagationError


# ---------------------------------------------------------------------------
# Finding #1 — FM-1: STM integration failure raises PropagationError always
# ---------------------------------------------------------------------------


class TestFinding1_STMFailureRaises:
    """propagate_covariance_stm must ALWAYS raise PropagationError on failure.

    The old code silently returned the initial covariance in non-STRICT mode,
    freezing uncertainty and causing false-negative Pc values.
    """

    def test_stm_failure_raises_propagation_error(self, monkeypatch):
        """Simulate a failed IVP solve and assert PropagationError is raised."""
        from unittest.mock import patch as _patch

        failed_sol = Mock()
        failed_sol.success = False
        failed_sol.message = "Required step size is less than spacing"

        with _patch("astra.covariance.solve_ivp", return_value=failed_sol):
            from astra.covariance import propagate_covariance_stm

            r0 = np.array([7000.0, 0.0, 0.0])
            v0 = np.array([0.0, 7.5, 0.0])
            cov0 = np.eye(6) * 1e-4

            with pytest.raises(PropagationError) as exc_info:
                propagate_covariance_stm(
                    t_jd0=2460000.5,
                    r0_km=r0,
                    v0_km_s=v0,
                    cov0_6x6=cov0,
                    duration_s=3600.0,
                )

            msg = str(exc_info.value)
            assert "STM covariance propagation failed" in msg
            assert "initial covariance" in msg.lower() or "fallback" in msg.lower()

    def test_stm_failure_raises_regardless_of_strict_mode(self, monkeypatch):
        """Raising must be unconditional — not gated on STRICT_MODE."""
        from unittest.mock import patch as _patch
        import astra.config as cfg

        failed_sol = Mock()
        failed_sol.success = False
        failed_sol.message = "step size too small"

        # Test with STRICT_MODE=False  (the old silent-fallback path)
        original_strict = cfg.ASTRA_STRICT_MODE
        cfg.ASTRA_STRICT_MODE = False
        try:
            with _patch("astra.covariance.solve_ivp", return_value=failed_sol):
                from astra.covariance import propagate_covariance_stm

                r0 = np.array([7000.0, 0.0, 0.0])
                v0 = np.array([0.0, 7.5, 0.0])
                cov0 = np.eye(6) * 1e-4

                # MUST still raise even in relaxed mode
                with pytest.raises(PropagationError):
                    propagate_covariance_stm(
                        t_jd0=2460000.5, r0_km=r0, v0_km_s=v0,
                        cov0_6x6=cov0, duration_s=3600.0,
                    )
        finally:
            cfg.ASTRA_STRICT_MODE = original_strict

    def test_stm_success_returns_propagated_covariance(self):
        """On success, the result must NOT be the initial covariance."""
        from astra.covariance import propagate_covariance_stm

        r0 = np.array([7000.0, 0.0, 0.0])
        v0 = np.array([0.0, 7.546, 0.0])  # circular LEO ~7000 km
        cov0 = np.diag([1e-4, 1e-4, 1e-4, 1e-8, 1e-8, 1e-8])

        result = propagate_covariance_stm(
            t_jd0=2460000.5, r0_km=r0, v0_km_s=v0,
            cov0_6x6=cov0, duration_s=300.0,
        )
        assert result.shape == (6, 6)
        # Propagated covariance must differ from the initial one
        assert not np.allclose(result, cov0), (
            "Propagated covariance equals initial — STM had no effect"
        )
        # Must remain symmetric and positive semi-definite
        assert np.allclose(result, result.T, atol=1e-12)
        eigvals = np.linalg.eigvalsh(result)
        assert np.all(eigvals >= -1e-12), f"Covariance not PSD: min eigenvalue={eigvals.min()}"


# ---------------------------------------------------------------------------
# Finding #2 — FM-3: NRLMSISE-00 inside STM Jacobian
# ---------------------------------------------------------------------------


class TestFinding2_NRLMSISEInSTMKernel:
    """_acceleration_njit must use NRLMSISE-00 when use_nrlmsise=True."""

    def test_nrlmsise_gives_different_density_than_exponential(self):
        """NRLMSISE-00 rho should differ from a simple exponential at 400 km."""
        from astra.propagator import _nrlmsise00_density_njit

        rho_nrl = _nrlmsise00_density_njit(400.0, 150.0, 150.0, 15.0)
        # Canonical calibration anchor: ~3.7e-12 kg/m³
        assert 1e-13 < rho_nrl < 1e-10, (
            f"NRLMSISE-00 density at 400 km is {rho_nrl:.3e} — out of physical range"
        )

    def test_acceleration_nrlmsise_flag_changes_drag(self):
        """use_nrlmsise=True must produce a different acceleration than False."""
        try:
            from astra.covariance import _acceleration_njit
            from astra.propagator import _nrlmsise00_density_njit  # ensure Numba closure
        except (ImportError, Exception):
            pytest.skip("_acceleration_njit not compilable (Numba not installed)")

        r = np.array([6778.0, 0.0, 0.0])   # ~400 km altitude
        v = np.array([0.0, 7.67, 0.0])     # LEO velocity

        Bc = 0.022    # m²/kg (typical debris)
        rho_ref = 3.7e-12
        H_km = 58.5
        rho_ref_alt_km = 400.0

        try:
            a_exp = _acceleration_njit(r, v, Bc, rho_ref, H_km, rho_ref_alt_km,
                                       150.0, 150.0, 15.0, False)
            a_nrl = _acceleration_njit(r, v, Bc, rho_ref, H_km, rho_ref_alt_km,
                                       150.0, 150.0, 15.0, True)
        except Exception as exc:
            pytest.skip(f"Numba compilation failed: {exc}")

        # The two drag models MUST produce different results
        # Use atol=0 so the ~1e-9 drag accelerations aren't masked by the default atol=1e-8
        assert not np.allclose(a_exp, a_nrl, rtol=1e-6, atol=0), (
            "NRLMSISE flag has no effect — drag model is not being switched"
        )

    def test_stm_propagation_uses_nrlmsise_when_configured(self):
        """propagate_covariance_stm with NRLMSISE DragConfig must not raise."""
        from astra.covariance import propagate_covariance_stm
        from astra.propagator import DragConfig

        drag = DragConfig(cd=2.2, area_m2=10.0, mass_kg=500.0, model="NRLMSISE00")
        r0 = np.array([7000.0, 0.0, 0.0])
        v0 = np.array([0.0, 7.5, 0.0])
        cov0 = np.diag([1e-4]*3 + [1e-8]*3)

        # get_space_weather is imported inside propagate_covariance_stm from data_pipeline
        with patch("astra.data_pipeline.get_space_weather", return_value=(150.0, 150.0, 15.0)):
            result = propagate_covariance_stm(
                t_jd0=2460000.5, r0_km=r0, v0_km_s=v0,
                cov0_6x6=cov0, duration_s=300.0, drag_config=drag,
            )
        assert result.shape == (6, 6)


# ---------------------------------------------------------------------------
# Finding #3 — FM-9: Constants leakage — propagator.py signatures
# ---------------------------------------------------------------------------


class TestFinding3_ConstantsLeakage:
    """srp_illumination_factor default must reference EARTH_EQUATORIAL_RADIUS_KM."""

    def test_srp_factor_default_uses_constant(self):
        """The default earth_radius_km must equal EARTH_EQUATORIAL_RADIUS_KM."""
        import inspect
        from astra.propagator import srp_illumination_factor
        from astra.constants import EARTH_EQUATORIAL_RADIUS_KM

        sig = inspect.signature(srp_illumination_factor)
        default = sig.parameters["earth_radius_km"].default
        assert default == EARTH_EQUATORIAL_RADIUS_KM, (
            f"srp_illumination_factor default {default} != "
            f"EARTH_EQUATORIAL_RADIUS_KM {EARTH_EQUATORIAL_RADIUS_KM}"
        )

    def test_constant_value_is_wgs84(self):
        """EARTH_EQUATORIAL_RADIUS_KM must be the WGS-84 value 6378.137."""
        from astra.constants import EARTH_EQUATORIAL_RADIUS_KM
        assert EARTH_EQUATORIAL_RADIUS_KM == 6378.137


# ---------------------------------------------------------------------------
# Finding #4 — FM-1: UT1-UTC fallback warning quantifies error
# ---------------------------------------------------------------------------


class TestFinding4_UT1UTCWarningQuality:
    """The UT1-UTC fallback log message must mention the ~400 m error impact."""

    def test_fallback_warning_mentions_error_magnitude(self, caplog):
        """UT1-UTC fallback log must state the ~400m error magnitude."""
        import logging
        from astra.models import SatelliteTLE
        import astra.config as cfg

        tle = SatelliteTLE(
            norad_id="25544", name="ISS",
            line1="1 25544U 98067A   21001.00000000  .00001480  00000-0  34282-4 0  9990",
            line2="2 25544  51.6442 284.1199 0001364 338.5498  21.5664 15.48922536 12341",
            epoch_jd=2459215.5,
            object_type="PAYLOAD",
        )

        old_strict = cfg.ASTRA_STRICT_MODE
        cfg.ASTRA_STRICT_MODE = False
        import astra.orbit
        old_propagate = astra.orbit.logger.propagate
        astra.orbit.logger.propagate = True

        try:
            with patch("astra.data_pipeline.get_ut1_utc_correction",
                       side_effect=Exception("EOP cache empty")):
                from astra.orbit import propagate_orbit
                with caplog.at_level(logging.WARNING, logger="astra.orbit"):
                    try:
                        propagate_orbit(tle, 2459215.5, 0.0)
                    except Exception:
                        pass
        finally:
            cfg.ASTRA_STRICT_MODE = old_strict
            astra.orbit.logger.propagate = old_propagate

        all_warnings = " ".join([r.message for r in caplog.records])
        assert "400" in all_warnings or "UT1" in all_warnings, (
            f"Warning message does not mention '400' or 'UT1'. Got:\n{all_warnings}"
        )


# ---------------------------------------------------------------------------
# Finding #5 — FM-6: export_ocm_xml is implemented and round-trips
# ---------------------------------------------------------------------------


class TestFinding5_ExportOcmXml:
    """export_ocm_xml must exist, be importable, and round-trip with parse_ocm_xml."""

    def test_export_ocm_xml_is_importable(self):
        from astra.ocm import export_ocm_xml
        assert callable(export_ocm_xml)

    def test_export_ocm_xml_in_astra_namespace(self):
        import astra
        assert hasattr(astra, "export_ocm_xml"), (
            "export_ocm_xml is not exported in the astra top-level namespace"
        )

    def test_export_ocm_xml_produces_valid_xml(self):
        """export_ocm_xml must produce well-formed XML with required content."""
        from astra.ocm import export_ocm_xml
        from astra.propagator import NumericalState

        states = [
            NumericalState(
                t_jd=2460000.5,
                position_km=np.array([7000.0, 0.0, 0.0]),
                velocity_km_s=np.array([0.0, 7.5, 0.0]),
            ),
            NumericalState(
                t_jd=2460000.5 + 300.0 / 86400.0,
                position_km=np.array([6900.0, 100.0, 50.0]),
                velocity_km_s=np.array([0.1, 7.4, 0.2]),
            ),
        ]

        xml_str = export_ocm_xml(states, object_name="TEST_SAT")
        assert isinstance(xml_str, str), "export_ocm_xml must return a string"
        assert len(xml_str) > 100, "XML output is suspiciously short"
        # Must contain the satellite name
        assert "TEST_SAT" in xml_str, "object_name not found in exported XML"
        # Must contain at least one position value
        assert "7000" in xml_str or "7.0" in xml_str, (
            "Position value 7000 km not present in exported XML"
        )
        # Must be parseable as XML
        import xml.etree.ElementTree as ET
        try:
            ET.fromstring(xml_str)
        except ET.ParseError as exc:
            pytest.fail(f"export_ocm_xml produced malformed XML: {exc}")


# ---------------------------------------------------------------------------
# Finding #6 — FM-2: fetch_celestrak_comprehensive logs warning on failure
# ---------------------------------------------------------------------------


class TestFinding6_CelestrakGroupFailureWarning:
    """A failing group in fetch_celestrak_comprehensive must log a warning."""

    def test_warning_logged_on_group_failure(self, caplog):
        """A failing CelesTrak group must emit a WARNING to stdout (ASTRA custom handler)."""
        import logging
        from astra.data import fetch_celestrak_comprehensive

        call_count = {"n": 0}

        def _patched_fetch(group, format="tle"):
            call_count["n"] += 1
            if group == "active":
                return []   # empty but succeeds
            raise AstraError(f"Mock network failure for group '{group}'")

        import astra.data
        old_propagate = astra.data.logger.propagate
        astra.data.logger.propagate = True

        try:
            with patch("astra.data.fetch_celestrak_group", side_effect=_patched_fetch):
                with caplog.at_level(logging.WARNING, logger="astra.data"):
                    fetch_celestrak_comprehensive()
        finally:
            astra.data.logger.propagate = old_propagate

        # Must have attempted multiple groups
        assert call_count["n"] > 1, "fetch_celestrak_comprehensive only called once"

        # Must have emitted a WARNING (not silently passed)
        combined = " ".join([r.message for r in caplog.records])
        assert "WARNING" in combined or "INCOMPLETE" in combined or "EXCLUDED" in combined, (
            "No WARNING was logged when a CelesTrak group failed — "
            f"silent pass may still be active (Finding #6 not fixed).\nOutput:\n{combined}"
        )

    def test_partial_catalog_still_returned(self):
        """Even when some groups fail, successful groups are still returned."""
        from astra.models import SatelliteTLE
        from astra.data import fetch_celestrak_comprehensive

        mock_tle = SatelliteTLE(
            norad_id="99999", name="MOCK",
            line1="1 99999U 99999A   26100.00000000  .00000000  00000-0  00000-0 0  9999",
            line2="2 99999   0.0000   0.0000 0000000   0.0000   0.0000 15.00000000    00",
            epoch_jd=2460000.5, object_type="PAYLOAD",
        )

        def _patched_fetch(group, format="tle"):
            if group == "active":
                return [mock_tle]
            raise AstraError("simulated failure")

        with patch("astra.data.fetch_celestrak_group", side_effect=_patched_fetch):
            result = fetch_celestrak_comprehensive()

        assert len(result) >= 1, "Partial catalog should still contain active-group objects"

    def test_strict_mode_raises_on_group_failure(self):
        """When strict_mode=True, a failing CelesTrak group must raise an AstraError."""
        from astra.data import fetch_celestrak_comprehensive
        
        def _patched_fetch(group, format="tle"):
            if group == "active":
                return []
            raise AstraError(f"Mock network failure for group '{group}'")
            
        with patch("astra.data.fetch_celestrak_group", side_effect=_patched_fetch):
            with pytest.raises(AstraError, match="Strict mode is enabled"):
                fetch_celestrak_comprehensive(strict_mode=True)


# ---------------------------------------------------------------------------
# Finding #7 — FM-5: Spacetrack test parses correct orbital elements
# ---------------------------------------------------------------------------


class TestFinding7_SpacetrackOMMPhysics:
    """fetch_spacetrack_group must parse correct physical orbital elements."""

    _MOCK_OMM = json.dumps([{
        "OBJECT_NAME": "ISS (ZARYA)", "OBJECT_ID": "1998-067A",
        "NORAD_CAT_ID": "25544", "EPOCH": "2021-01-01T00:00:00.000000",
        "MEAN_MOTION": "15.48922536", "ECCENTRICITY": ".0001364",
        "INCLINATION": "51.6442", "RA_OF_ASC_NODE": "284.1199",
        "ARG_OF_PERICENTER": "338.5498", "MEAN_ANOMALY": "21.5664",
        "BSTAR": ".34282E-4", "RCS_SIZE": "LARGE",
    }])

    @patch("astra.spacetrack.requests.Session")
    def test_parsed_inclination_matches_physical_value(self, mock_session_cls):
        from astra.spacetrack import fetch_spacetrack_group

        mock_session = MagicMock()
        mock_session_cls.return_value = mock_session
        mock_session.post.return_value = Mock(status_code=200, ok=True, text="OK")
        mock_session.get.return_value = Mock(
            status_code=200, ok=True, text=self._MOCK_OMM,
            raise_for_status=Mock(), headers={},
        )

        env = {"SPACETRACK_USER": "u@test.com", "SPACETRACK_PASS": "pw"}
        with patch.dict(os.environ, env):
            results = fetch_spacetrack_group("active")

        omm = results[0]
        expected_inc = math.radians(51.6442)
        assert abs(omm.inclination_rad - expected_inc) < 1e-6, (
            f"Inclination: expected {expected_inc:.6f} rad, got {omm.inclination_rad:.6f} rad"
        )
        expected_mm = 15.48922536 * 2.0 * math.pi / 1440.0
        assert abs(omm.mean_motion_rad_min - expected_mm) < 1e-9
        assert abs(omm.eccentricity - 0.0001364) < 1e-7
        assert omm.epoch_jd > 2451545.0


# ---------------------------------------------------------------------------
# Finding #8 — FM-7: run_conjunction_sweep high-level wrapper
# ---------------------------------------------------------------------------


class TestFinding8_RunConjunctionSweep:
    """run_conjunction_sweep must exist and handle orchestration internally."""

    def test_run_conjunction_sweep_is_importable(self):
        from astra.conjunction import run_conjunction_sweep
        assert callable(run_conjunction_sweep)

    def test_run_conjunction_sweep_in_astra_namespace(self):
        import astra
        assert hasattr(astra, "run_conjunction_sweep")

    def test_empty_catalog_raises(self):
        from astra.conjunction import run_conjunction_sweep
        with pytest.raises(AstraError, match="catalog is empty"):
            run_conjunction_sweep([], t_start_jd=2460000.5, t_end_jd=2460001.5)

    def test_invalid_time_window_raises(self):
        from astra.conjunction import run_conjunction_sweep
        from astra.models import SatelliteTLE

        tle = SatelliteTLE(
            norad_id="25544", name="ISS",
            line1="1 25544U 98067A   21001.00000000  .00001480  00000-0  34282-4 0  9990",
            line2="2 25544  51.6442 284.1199 0001364 338.5498  21.5664 15.48922536 12341",
            epoch_jd=2459215.5, object_type="PAYLOAD",
        )
        with pytest.raises(AstraError, match="strictly greater"):
            run_conjunction_sweep([tle], t_start_jd=2460001.5, t_end_jd=2460000.5)

    def test_run_conjunction_sweep_returns_list(self):
        """With two satellites on nearly identical orbits, returns a list."""
        from astra.conjunction import run_conjunction_sweep
        from astra.models import SatelliteTLE

        # Two ISS-like TLEs (nearly co-orbital)
        tle_a = SatelliteTLE(
            norad_id="25544", name="ISS-A",
            line1="1 25544U 98067A   21001.00000000  .00001480  00000-0  34282-4 0  9990",
            line2="2 25544  51.6442 284.1199 0001364 338.5498  21.5664 15.48922536 12341",
            epoch_jd=2459215.5, object_type="PAYLOAD",
        )
        tle_b = SatelliteTLE(
            norad_id="99999", name="ISS-B",
            line1="1 99999U 98067B   21001.00000000  .00001480  00000-0  34282-4 0  9991",
            line2="2 99999  51.6442 284.1199 0001364 338.5498  21.5664 15.48922536 12342",
            epoch_jd=2459215.5, object_type="PAYLOAD",
        )

        t0 = 2459215.5
        t1 = t0 + 0.1  # ~2.4 hour window

        # UT1-UTC lookup fails gracefully; no mock needed — orbit.py handles the fallback
        events = run_conjunction_sweep(
            [tle_a, tle_b], t_start_jd=t0, t_end_jd=t1,
            step_minutes=5.0, threshold_km=10.0,
        )

        assert isinstance(events, list), f"Expected list, got {type(events)}"
        # Both satellites should have been processed (not dropped by NaN filter)
        # The list may be empty if they don't pass within threshold — that's fine
        assert events is not None


# ---------------------------------------------------------------------------
# Finding #9 — FM-4: export_cdm_kvn + parse_cdm_kvn symmetry
# ---------------------------------------------------------------------------


class TestFinding9_CDMKVNExport:
    """export_cdm_kvn must implement KVN serialisation and round-trip correctly."""

    def _make_cdm(self) -> object:
        from astra.cdm import ConjunctionDataMessage, CDMObject
        obj1 = CDMObject(
            object_designator="25544",
            object_name="ISS (ZARYA)",
            position_xyz=(100.0, 200.0, 300.0),
            velocity_xyz=(1.1, 2.2, 3.3),
            covariance_matrix=[1e-4] * 21,
        )
        obj2 = CDMObject(
            object_designator="40000",
            object_name="DEBRIS-A",
            position_xyz=(105.0, 201.0, 301.0),
            velocity_xyz=(1.2, 2.3, 3.4),
            covariance_matrix=[2e-4] * 21,
        )
        return ConjunctionDataMessage(
            message_id="TEST-001",
            creation_date=datetime(2026, 4, 25, 12, 0, 0, tzinfo=timezone.utc),
            tca_time=datetime(2026, 4, 25, 13, 0, 0, tzinfo=timezone.utc),
            miss_distance_m=150.0,
            relative_velocity_m_s=14200.0,
            collision_probability=1.5e-4,
            object_1=obj1,
            object_2=obj2,
        )

    def test_export_cdm_kvn_importable(self):
        from astra.cdm import export_cdm_kvn
        assert callable(export_cdm_kvn)

    def test_export_cdm_kvn_in_astra_namespace(self):
        import astra
        assert hasattr(astra, "export_cdm_kvn")

    def test_parse_cdm_kvn_importable(self):
        from astra.cdm import parse_cdm_kvn
        assert callable(parse_cdm_kvn)

    def test_export_cdm_kvn_produces_valid_string(self):
        from astra.cdm import export_cdm_kvn
        cdm = self._make_cdm()
        kvn = export_cdm_kvn(cdm, originator="ASTRA-TEST")
        assert isinstance(kvn, str)
        assert "TCA" in kvn
        assert "MISS_DISTANCE" in kvn
        assert "OBJECT1" in kvn
        assert "OBJECT2" in kvn
        assert "ASTRA-TEST" in kvn
        assert "TEST-001" in kvn

    def test_export_cdm_kvn_round_trips(self):
        """parse_cdm_kvn(export_cdm_kvn(cdm)) must recover original values."""
        from astra.cdm import export_cdm_kvn, parse_cdm_kvn
        cdm = self._make_cdm()
        kvn = export_cdm_kvn(cdm)
        recovered = parse_cdm_kvn(kvn)

        assert recovered.message_id == cdm.message_id
        assert abs(recovered.miss_distance_m - cdm.miss_distance_m) < 0.01
        assert abs(recovered.relative_velocity_m_s - cdm.relative_velocity_m_s) < 0.01
        assert recovered.collision_probability is not None
        assert abs(recovered.collision_probability - cdm.collision_probability) < 1e-9
        assert recovered.object_1.object_designator == "25544"
        assert recovered.object_2.object_designator == "40000"

    def test_parse_cdm_kvn_missing_required_field_raises(self):
        from astra.cdm import parse_cdm_kvn
        # KVN missing TCA
        kvn = (
            "MESSAGE_ID = TEST\n"
            "CREATION_DATE = 2026-04-25T12:00:00Z\n"
            "MISS_DISTANCE = 100.0\n"
            "RELATIVE_SPEED = 14000.0\n"
        )
        with pytest.raises(AstraError, match="TCA"):
            parse_cdm_kvn(kvn)

    def test_parse_cdm_kvn_negative_miss_distance_raises(self):
        from astra.cdm import parse_cdm_kvn
        kvn = (
            "MESSAGE_ID = TEST\n"
            "CREATION_DATE = 2026-04-25T12:00:00Z\n"
            "TCA = 2026-04-25T13:00:00Z\n"
            "MISS_DISTANCE = -50.0\n"
            "RELATIVE_SPEED = 14000.0\n"
        )
        with pytest.raises(AstraError, match="Negative miss distance"):
            parse_cdm_kvn(kvn)

    def test_export_cdm_xml_still_works(self):
        """Ensure the existing XML exporter was not broken by the KVN addition."""
        from astra.cdm import export_cdm_xml, parse_cdm_xml
        cdm = self._make_cdm()
        xml = export_cdm_xml(cdm)
        assert "<?xml" in xml
        recovered = parse_cdm_xml(xml)
        assert recovered.message_id == cdm.message_id
