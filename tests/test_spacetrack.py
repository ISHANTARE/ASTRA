# tests/test_spacetrack.py
"""Test suite for the Space-Track.org authenticated data ingestion module.
All tests use mocked network sessions — no real Space-Track credentials
or network connections are required to run this suite.
"""
import json
import os
from datetime import datetime, timedelta, timezone
import pytest
import requests
from unittest.mock import patch, Mock, MagicMock
from astra.spacetrack import (
    fetch_spacetrack_group,
    fetch_spacetrack_active,
    _get_credentials,
    _create_session,
)
from astra.errors import AstraError
from astra.models import SatelliteOMM, SatelliteTLE
@pytest.fixture(autouse=True)
def clear_session_cache():
    from astra.spacetrack import _SESSION_CACHE
    _SESSION_CACHE.clear()
    yield
# ---------------------------------------------------------------------------
# Fixture: realistic OMM JSON payload
# ---------------------------------------------------------------------------
_MOCK_OMM_PAYLOAD = json.dumps(
    [
        {
            "OBJECT_NAME": "ISS (ZARYA)",
            "OBJECT_ID": "1998-067A",
            "NORAD_CAT_ID": "25544",
            "OBJECT_TYPE": "PAYLOAD",
            "EPOCH": "2021-01-01T00:00:00.000000",
            "MEAN_MOTION": "15.48922536",
            "ECCENTRICITY": ".0001364",
            "INCLINATION": "51.6442",
            "RA_OF_ASC_NODE": "284.1199",
            "ARG_OF_PERICENTER": "338.5498",
            "MEAN_ANOMALY": "21.5664",
            "BSTAR": ".34282E-4",
            "RCS_SIZE": "LARGE",
        }
    ]
)
_MOCK_TLE_PAYLOAD = (
    "ISS (ZARYA)\n"
    "1 25544U 98067A   21001.00000000  .00001480  00000-0  34282-4 0  9990\n"
    "2 25544  51.6442 284.1199 0001364 338.5498  21.5664 15.48922536 12341\n"
)
# ---------------------------------------------------------------------------
# 1. Credential Handling
# ---------------------------------------------------------------------------
class TestCredentials:
    def test_raises_when_both_env_vars_missing(self):
        with patch.dict(os.environ, {}, clear=True):
            # Ensure neither variable is set
            os.environ.pop("SPACETRACK_USER", None)
            os.environ.pop("SPACETRACK_PASS", None)
            with pytest.raises(AstraError) as exc_info:
                _get_credentials()
            # Error message must include instructions
            assert "SPACETRACK_USER" in str(exc_info.value)
            assert "SPACETRACK_PASS" in str(exc_info.value)
    def test_raises_when_only_user_missing(self):
        with patch.dict(os.environ, {"SPACETRACK_PASS": "secret"}, clear=False):
            os.environ.pop("SPACETRACK_USER", None)
            with pytest.raises(AstraError):
                _get_credentials()
    def test_raises_when_only_pass_missing(self):
        with patch.dict(os.environ, {"SPACETRACK_USER": "test@test.com"}, clear=False):
            os.environ.pop("SPACETRACK_PASS", None)
            with pytest.raises(AstraError):
                _get_credentials()
    def test_returns_credentials_when_both_set(self):
        env = {"SPACETRACK_USER": "test@example.com", "SPACETRACK_PASS": "mypassword"}
        with patch.dict(os.environ, env):
            user, password = _get_credentials()
            assert user == "test@example.com"
            assert password == "mypassword"
    def test_error_message_contains_setx_instructions(self):
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("SPACETRACK_USER", None)
            os.environ.pop("SPACETRACK_PASS", None)
            with pytest.raises(AstraError) as exc_info:
                _get_credentials()
            msg = str(exc_info.value)
            assert "setx" in msg or "export" in msg
# ---------------------------------------------------------------------------
# 2. Authentication
# ---------------------------------------------------------------------------
class TestAuthentication:
    @patch("astra.spacetrack.requests.Session")
    def test_successful_auth_creates_session(self, mock_session_cls):
        mock_session = MagicMock()
        mock_session_cls.return_value = mock_session
        mock_login_resp = Mock()
        mock_login_resp.status_code = 200
        mock_login_resp.ok = True
        mock_login_resp.text = "Login Successful"
        mock_session.post.return_value = mock_login_resp
        session = _create_session("user@test.com", "pass")
        assert session is mock_session
        mock_session.post.assert_called_once()
    @patch("astra.spacetrack.requests.Session")
    def test_bad_credentials_raises_astra_error(self, mock_session_cls):
        mock_session = MagicMock()
        mock_session_cls.return_value = mock_session
        mock_session.post.return_value = Mock(
            status_code=401, ok=False, text="Failed Login"
        )
        with pytest.raises(AstraError) as exc_info:
            _create_session("bad@user.com", "wrongpass")
        assert "authentication failed" in str(exc_info.value).lower()
    @patch("astra.spacetrack.requests.Session")
    def test_network_error_raises_astra_error(self, mock_session_cls):
        import requests as req_lib
        mock_session = MagicMock()
        mock_session_cls.return_value = mock_session
        mock_session.post.side_effect = req_lib.RequestException("Connection timeout")
        with pytest.raises(AstraError) as exc_info:
            _create_session("user@test.com", "pass")
        assert "connect" in str(exc_info.value).lower()
# ---------------------------------------------------------------------------
# 3. Data Fetching — OMM format (default)
# ---------------------------------------------------------------------------
class TestFetchSpacetrackOMM:
    @patch("astra.spacetrack.requests.Session")
    def test_fetch_group_returns_omm_list(self, mock_session_cls):
        mock_session = MagicMock()
        mock_session_cls.return_value = mock_session
        # Login succeeds
        mock_session.post.return_value = Mock(status_code=200, ok=True, text="OK")
        # Data query succeeds
        mock_session.get.return_value = Mock(
            status_code=200,
            ok=True,
            text=_MOCK_OMM_PAYLOAD,
            raise_for_status=Mock(),
            headers={},
        )
        env = {"SPACETRACK_USER": "u@test.com", "SPACETRACK_PASS": "pw"}
        with patch.dict(os.environ, env):
            results = fetch_spacetrack_group("active")
        assert len(results) == 1
        omm = results[0]
        assert isinstance(omm, SatelliteOMM), (
            f"Expected SatelliteOMM, got {type(omm).__name__}"
        )
        # --- Fix: assert specific parsed orbital elements ---
        # A function returning an empty or zeroed SatelliteOMM must fail here.
        assert omm.norad_id == "25544", (
            f"NORAD ID mismatch: expected '25544', got {omm.norad_id!r}"
        )
        import math
        # Inclination: 51.6442° → radians
        expected_inc_rad = math.radians(51.6442)
        assert abs(omm.inclination_rad - expected_inc_rad) < 1e-6, (
            f"Inclination mismatch: expected {expected_inc_rad:.6f} rad, "
            f"got {omm.inclination_rad:.6f} rad"
        )
        # Mean motion: 15.48922536 rev/day → rad/min
        expected_mm_rad_min = 15.48922536 * 2.0 * math.pi / 1440.0
        assert abs(omm.mean_motion_rad_min - expected_mm_rad_min) < 1e-9, (
            f"Mean motion mismatch: expected {expected_mm_rad_min:.9f} rad/min, "
            f"got {omm.mean_motion_rad_min:.9f} rad/min"
        )
        # Eccentricity: 0.0001364
        assert abs(omm.eccentricity - 0.0001364) < 1e-7, (
            f"Eccentricity mismatch: expected 0.0001364, got {omm.eccentricity}"
        )
        # Epoch must be a valid Julian Date (> J2000)
        assert omm.epoch_jd > 2451545.0, (
            f"Epoch JD looks wrong: {omm.epoch_jd} (expected > J2000=2451545.0)"
        )
    @patch("astra.spacetrack.requests.Session")
    def test_fetch_active_returns_omm_list(self, mock_session_cls):
        mock_session = MagicMock()
        mock_session_cls.return_value = mock_session
        mock_session.post.return_value = Mock(status_code=200, ok=True, text="OK")
        mock_session.get.return_value = Mock(
            status_code=200,
            ok=True,
            text=_MOCK_OMM_PAYLOAD,
            raise_for_status=Mock(),
            headers={},
        )
        env = {"SPACETRACK_USER": "u@test.com", "SPACETRACK_PASS": "pw"}
        with patch.dict(os.environ, env):
            results = fetch_spacetrack_active()
        omm = results[0]
        assert isinstance(omm, SatelliteOMM), (
            f"Expected SatelliteOMM, got {type(omm).__name__}"
        )
        # --- Fix: assert specific orbital elements ---
        assert omm.norad_id == "25544", (
            f"fetch_spacetrack_active NORAD ID mismatch: got {omm.norad_id!r}"
        )
        import math
        expected_inc_rad = math.radians(51.6442)
        assert abs(omm.inclination_rad - expected_inc_rad) < 1e-6, (
            f"fetch_spacetrack_active inclination mismatch: "
            f"expected {expected_inc_rad:.6f} rad, got {omm.inclination_rad:.6f} rad"
        )
# ---------------------------------------------------------------------------
# 4. Data Fetching — TLE format
# ---------------------------------------------------------------------------
class TestFetchSpacetrackTLE:
    @patch("astra.spacetrack.requests.Session")
    def test_fetch_group_tle_returns_satellite_tle_list(self, mock_session_cls):
        mock_session = MagicMock()
        mock_session_cls.return_value = mock_session
        mock_session.post.return_value = Mock(status_code=200, ok=True, text="OK")
        mock_session.get.return_value = Mock(
            status_code=200,
            ok=True,
            text=_MOCK_TLE_PAYLOAD,
            raise_for_status=Mock(),
            headers={},
        )
        env = {"SPACETRACK_USER": "u@test.com", "SPACETRACK_PASS": "pw"}
        with patch.dict(os.environ, env):
            results = fetch_spacetrack_group("active", format="tle")
        assert len(results) >= 1
        tle = results[0]
        assert isinstance(tle, SatelliteTLE), (
            f"Expected SatelliteTLE, got {type(tle).__name__}"
        )
        # --- Fix: assert specific parsed TLE fields ---
        assert tle.norad_id == "25544", (
            f"TLE NORAD ID mismatch: expected '25544', got {tle.norad_id!r}"
        )
        assert tle.name.strip() == "ISS (ZARYA)", (
            f"TLE name mismatch: expected 'ISS (ZARYA)', got {tle.name!r}"
        )
        # Epoch JD must be valid (> J2000)
        assert tle.epoch_jd > 2451545.0, (
            f"TLE epoch_jd looks wrong: {tle.epoch_jd}"
        )
        # BSTAR drag term: 0.34282e-4 (from line 1 field)
        assert abs(tle.bstar - 0.34282e-4) < 1e-9, (
            f"TLE BSTAR mismatch: expected 3.4282e-5, got {tle.bstar}"
        )
# ---------------------------------------------------------------------------
# 5. Error Handling — Fetch failures
# ---------------------------------------------------------------------------
class TestFetchSpacetrackErrors:
    def test_fetch_without_credentials_raises(self):
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("SPACETRACK_USER", None)
            os.environ.pop("SPACETRACK_PASS", None)
            with pytest.raises(AstraError) as exc_info:
                fetch_spacetrack_group("active")
            assert "SPACETRACK_USER" in str(exc_info.value)
    @patch("astra.spacetrack.requests.Session")
    def test_empty_response_raises_astra_error(self, mock_session_cls):
        mock_session = MagicMock()
        mock_session_cls.return_value = mock_session
        mock_session.post.return_value = Mock(status_code=200, ok=True, text="OK")
        mock_session.get.return_value = Mock(
            status_code=200, ok=True, text="   ", raise_for_status=Mock(), headers={}
        )
        env = {"SPACETRACK_USER": "u@test.com", "SPACETRACK_PASS": "pw"}
        with patch.dict(os.environ, env):
            with pytest.raises(AstraError) as exc_info:
                fetch_spacetrack_group("nonexistent_group")
            assert "empty" in str(exc_info.value).lower()
    @patch("astra.spacetrack.requests.Session")
    def test_http_error_on_data_fetch_raises(self, mock_session_cls):
        import requests as req_lib
        mock_session = MagicMock()
        mock_session_cls.return_value = mock_session
        mock_session.post.return_value = Mock(status_code=200, ok=True, text="OK")
        mock_session.get.return_value = Mock(
            raise_for_status=Mock(side_effect=req_lib.HTTPError("503"))
        )
        env = {"SPACETRACK_USER": "u@test.com", "SPACETRACK_PASS": "pw"}
        with patch.dict(os.environ, env):
            with pytest.raises(AstraError):
                fetch_spacetrack_group("active")
# ---------------------------------------------------------------------------
# 6. Top-level namespace exports
# ---------------------------------------------------------------------------
class TestNamespaceExports:
    def test_spacetrack_functions_exported_from_astra(self):
        import astra
        assert hasattr(astra, "fetch_spacetrack_group")
        assert hasattr(astra, "fetch_spacetrack_active")
    def test_celestrak_omm_siblings_exported(self):
        import astra
        assert hasattr(astra, "fetch_celestrak_group_omm")
        assert hasattr(astra, "fetch_celestrak_active_omm")
        assert hasattr(astra, "fetch_celestrak_comprehensive_omm")
# ---------------------------------------------------------------------------
# 7. Session Cache & Logout
# ---------------------------------------------------------------------------
class TestSessionCacheAndLogout:
    @patch("astra.spacetrack.requests.Session")
    def test_session_caching(self, mock_session_cls):
        from astra.spacetrack import _SESSION_CACHE, _create_session
        _SESSION_CACHE.clear()
        mock_session = MagicMock()
        mock_session_cls.return_value = mock_session
        mock_login_resp = Mock(status_code=200, ok=True, text="Login Successful")
        mock_session.post.return_value = mock_login_resp
        # First call should hit the network
        session1 = _create_session("cache_user@test.com", "pass")
        mock_session.post.assert_called_once()
        # Second call should use cache
        mock_session.post.reset_mock()
        session2 = _create_session("cache_user@test.com", "pass")
        mock_session.post.assert_not_called()
        assert session1 is session2
        assert "cache_user@test.com" in _SESSION_CACHE
        _SESSION_CACHE.clear()
    @patch("astra.spacetrack.requests.Session")
    def test_session_cache_ttl_refreshes_expired_session(self, mock_session_cls):
        from astra.spacetrack import _SESSION_CACHE, _create_session
        _SESSION_CACHE.clear()
        old_session = MagicMock()
        old_session.post.return_value = Mock(status_code=200, ok=True, text="OK")
        _SESSION_CACHE["expired@test.com"] = (
            old_session,
            datetime.now(timezone.utc) - timedelta(hours=1),
        )
        new_session = MagicMock()
        new_session.post.return_value = Mock(status_code=200, ok=True, text="OK")
        mock_session_cls.return_value = new_session
        session = _create_session("expired@test.com", "pw")
        assert session is new_session
        old_session.close.assert_called_once()
    @patch("astra.spacetrack.requests.Session")
    def test_spacetrack_logout(self, mock_session_cls):
        from astra.spacetrack import spacetrack_logout, _SESSION_CACHE
        _SESSION_CACHE.clear()
        mock_session = MagicMock()
        _SESSION_CACHE["dummy@test.com"] = (mock_session, datetime.now(timezone.utc))
        env = {"SPACETRACK_USER": "dummy@test.com", "SPACETRACK_PASS": "pw"}
        with patch.dict(os.environ, env):
            spacetrack_logout()
        # Session should be removed from cache
        assert "dummy@test.com" not in _SESSION_CACHE
        mock_session.get.assert_called_once()  # Should call logout URL
        # Logout when no session should not crash
        with patch.dict(os.environ, env):
            spacetrack_logout()  # safe to call
# ---------------------------------------------------------------------------
# 8. SATCAT
# ---------------------------------------------------------------------------
class TestSATCAT:
    @patch("astra.spacetrack.requests.Session")
    def test_fetch_satcat_success(self, mock_session_cls):
        from astra.spacetrack import fetch_spacetrack_satcat
        mock_session = MagicMock()
        mock_session_cls.return_value = mock_session
        mock_session.post.return_value = Mock(status_code=200, ok=True, text="OK")
        mock_resp = Mock(
            status_code=200,
            ok=True,
            text='[{"SATNAME": "ISS", "NORAD_CAT_ID": "25544"}]',
        )
        mock_resp.headers = {}
        mock_session.get.return_value = mock_resp
        env = {"SPACETRACK_USER": "u@test.com", "SPACETRACK_PASS": "pw"}
        with patch.dict(os.environ, env):
            results = fetch_spacetrack_satcat(["25544"])
        assert len(results) == 1
        assert results[0]["SATNAME"] == "ISS"
        mock_session.get.assert_called_once()
        assert "NORAD_CAT_ID/25544" in mock_session.get.call_args[0][0]
# ---------------------------------------------------------------------------
# 9. Rate Limit & Pagination Guard
# ---------------------------------------------------------------------------
class TestRateLimitAndPagination:
    @patch("astra.spacetrack.requests.Session")
    @patch("astra.spacetrack.logger")
    def test_rate_limit_warning(self, mock_logger, mock_session_cls):
        mock_session = MagicMock()
        mock_session_cls.return_value = mock_session
        mock_session.post.return_value = Mock(status_code=200, ok=True, text="OK")
        mock_resp = Mock(status_code=200, ok=True, text=_MOCK_OMM_PAYLOAD)
        mock_resp.headers = {"X-RateLimit-Remaining": "5"}
        mock_session.get.return_value = mock_resp
        env = {"SPACETRACK_USER": "u@test.com", "SPACETRACK_PASS": "pw"}
        with patch.dict(os.environ, env):
            fetch_spacetrack_group("active", format="json")
        mock_logger.warning.assert_called()
        assert any(
            "rate limit low" in call[0][0]
            for call in mock_logger.warning.call_args_list
        )
    @patch("astra.spacetrack.requests.Session")
    @patch("astra.spacetrack.logger")
    def test_pagination_warning(self, mock_logger, mock_session_cls):
        mock_session = MagicMock()
        mock_session_cls.return_value = mock_session
        mock_session.post.return_value = Mock(status_code=200, ok=True, text="OK")
        # Create a large response string
        large_json = "[\n" + ",\n".join(["{}" for _ in range(50005)]) + "\n]"
        mock_resp = Mock(status_code=200, ok=True, text=large_json)
        mock_resp.headers = {}
        mock_session.get.return_value = mock_resp
        env = {"SPACETRACK_USER": "u@test.com", "SPACETRACK_PASS": "pw"}
        # We need to mock json.loads or parse_omm_json since the fake JSON won't parse completely correctly
        with (
            patch.dict(os.environ, env),
            patch("astra.omm.parse_omm_json", return_value=[]),
        ):
            fetch_spacetrack_group("active", format="json")
        mock_logger.warning.assert_called()
        assert any(
            "truncated by Space-Track API caps" in call[0][0]
            for call in mock_logger.warning.call_args_list
        )
class TestSessionReauthentication:
    @patch("astra.spacetrack.requests.Session")
    def test_fetch_group_reauthenticates_once_on_401(self, mock_session_cls):
        first_session = MagicMock()
        second_session = MagicMock()
        mock_session_cls.side_effect = [first_session, second_session]
        first_session.post.return_value = Mock(status_code=200, ok=True, text="OK")
        second_session.post.return_value = Mock(status_code=200, ok=True, text="OK")
        http_401 = Mock(status_code=401)
        http_401.raise_for_status.side_effect = requests.HTTPError(response=http_401)
        http_401.headers = {}
        first_session.get.return_value = http_401
        good = Mock(status_code=200, ok=True, text=_MOCK_OMM_PAYLOAD)
        good.raise_for_status = Mock()
        good.headers = {}
        second_session.get.return_value = good
        env = {"SPACETRACK_USER": "u@test.com", "SPACETRACK_PASS": "pw"}
        with patch.dict(os.environ, env):
            results = fetch_spacetrack_group("active")
        assert len(results) == 1
        assert isinstance(results[0], SatelliteOMM)
        first_session.close.assert_called_once()
