"""Tests for astra/data.py — CelesTrak TLE and OMM ingestion."""

import json
import pytest
from unittest.mock import patch, Mock

from astra.data import (
    _HEADERS,
    fetch_celestrak_active,
    fetch_celestrak_group,
    fetch_celestrak_active_omm,
    fetch_celestrak_group_omm,
)
from astra.models import SatelliteTLE, SatelliteOMM
from astra.errors import AstraError

# ---------------------------------------------------------------------------
# Mock payloads
# ---------------------------------------------------------------------------

TLE_MOCK_DATA = """\
ISS (ZARYA)
1 25544U 98067A   21001.00000000  .00001480  00000-0  34282-4 0  9990
2 25544  51.6442 284.1199 0001364 338.5498  21.5664 15.48922536 12341
STARLINK-1
1 44235U 19029A   21001.00000000  .00001480  00000-0  34282-4 0  9999
2 44235  53.0500 284.1199 0001364 338.5498  21.5664 15.48922536 12340"""

OMM_MOCK_DATA = json.dumps(
    [
        {
            "OBJECT_NAME": "ISS (ZARYA)",
            "NORAD_CAT_ID": "25544",
            "OBJECT_TYPE": "PAYLOAD",
            "EPOCH": "2021-01-01T00:00:00.000000",
            "MEAN_MOTION": "15.48922536",
            "ECCENTRICITY": "0.0001364",
            "INCLINATION": "51.6442",
            "RA_OF_ASC_NODE": "284.1199",
            "ARG_OF_PERICENTER": "338.5498",
            "MEAN_ANOMALY": "21.5664",
            "BSTAR": "0.000034282",
            "RCS_SIZE": "LARGE",
        },
        {
            "OBJECT_NAME": "STARLINK-1",
            "NORAD_CAT_ID": "44235",
            "OBJECT_TYPE": "PAYLOAD",
            "EPOCH": "2021-01-01T00:00:00.000000",
            "MEAN_MOTION": "15.48922537",
            "ECCENTRICITY": "0.0001364",
            "INCLINATION": "53.0500",
            "RA_OF_ASC_NODE": "284.1199",
            "ARG_OF_PERICENTER": "338.5498",
            "MEAN_ANOMALY": "21.5664",
            "BSTAR": "0.000034282",
            "RCS_SIZE": "SMALL",
        },
    ]
)

# ---------------------------------------------------------------------------
# TLE Tests
# ---------------------------------------------------------------------------


@patch("astra.data._session.get")
def test_fetch_celestrak_active_tle(mock_get):
    """Active catalog in TLE format returns SatelliteTLE list."""
    mock_resp = Mock()
    mock_resp.text = TLE_MOCK_DATA
    mock_resp.status_code = 200
    mock_resp.raise_for_status = Mock()
    mock_get.return_value = mock_resp

    catalog = fetch_celestrak_active()

    assert len(catalog) == 2
    assert isinstance(catalog[0], SatelliteTLE)
    assert catalog[0].name == "ISS (ZARYA)"
    assert catalog[0].norad_id == "25544"


@patch("astra.data._session.get")
def test_fetch_celestrak_group_tle(mock_get):
    """Group fetch in TLE format calls correct URL and returns SatelliteTLE list."""
    mock_resp = Mock()
    mock_resp.text = TLE_MOCK_DATA
    mock_resp.status_code = 200
    mock_resp.raise_for_status = Mock()
    mock_get.return_value = mock_resp

    catalog = fetch_celestrak_group("starlink")

    assert len(catalog) == 2
    assert isinstance(catalog[0], SatelliteTLE)
    mock_get.assert_called_with(
        "https://celestrak.org/NORAD/elements/gp.php?GROUP=starlink&FORMAT=tle",
        headers=_HEADERS,
        timeout=20.0,
        verify=True,
    )


# ---------------------------------------------------------------------------
# OMM Tests
# ---------------------------------------------------------------------------


@patch("astra.data._session.get")
def test_fetch_celestrak_active_omm(mock_get):
    """Active catalog in OMM JSON format returns SatelliteOMM list."""
    mock_resp = Mock()
    mock_resp.text = OMM_MOCK_DATA
    mock_resp.status_code = 200
    mock_resp.raise_for_status = Mock()
    mock_get.return_value = mock_resp

    catalog = fetch_celestrak_active_omm()

    assert len(catalog) == 2
    assert isinstance(catalog[0], SatelliteOMM)
    assert catalog[0].name == "ISS (ZARYA)"
    assert catalog[0].norad_id == "25544"
    assert catalog[0].rcs_m2 == 10.0  # LARGE → 10.0 m²


@patch("astra.data._session.get")
def test_fetch_celestrak_group_omm(mock_get):
    """Group fetch in OMM JSON format calls correct URL and returns SatelliteOMM list."""
    mock_resp = Mock()
    mock_resp.text = OMM_MOCK_DATA
    mock_resp.status_code = 200
    mock_resp.raise_for_status = Mock()
    mock_get.return_value = mock_resp

    catalog = fetch_celestrak_group_omm("starlink")

    assert len(catalog) == 2
    assert isinstance(catalog[0], SatelliteOMM)
    mock_get.assert_called_with(
        "https://celestrak.org/NORAD/elements/gp.php?GROUP=starlink&FORMAT=json",
        headers=_HEADERS,
        timeout=20.0,
        verify=True,
    )


@patch("astra.data._session.get")
def test_omm_fields_populated(mock_get):
    """OMM objects carry correct orbital element conversions from JSON values."""
    import math

    mock_resp = Mock()
    mock_resp.text = OMM_MOCK_DATA
    mock_resp.status_code = 200
    mock_resp.raise_for_status = Mock()
    mock_get.return_value = mock_resp

    catalog = fetch_celestrak_active_omm()
    iss = catalog[0]

    # Angles must be converted from degrees → radians
    assert abs(iss.inclination_rad - math.radians(51.6442)) < 1e-9
    assert abs(iss.raan_rad - math.radians(284.1199)) < 1e-9
    # Mean motion must be converted from rev/day → rad/min
    expected_mm = 15.48922536 * (2.0 * math.pi) / 1440.0
    assert abs(iss.mean_motion_rad_min - expected_mm) < 1e-12
    # Object type
    assert iss.object_type == "PAYLOAD"


# ---------------------------------------------------------------------------
# Error Handling Tests
# ---------------------------------------------------------------------------


@patch("astra.data._session.get")
def test_fetch_celestrak_network_error(mock_get):
    """Network failure raises AstraError with a helpful message."""
    import requests

    mock_get.side_effect = requests.RequestException("API down")

    with pytest.raises(AstraError, match="Failed to fetch CelesTrak"):
        fetch_celestrak_active()


@patch("astra.data._session.get")
def test_fetch_celestrak_rate_limit(mock_get):
    """Celestrak 403 rate-limit message raises AstraError with actionable text."""
    mock_resp = Mock()
    mock_resp.status_code = 403
    mock_resp.text = "GP data has not updated since your last successful download of GROUP=starlink at 2026-04-01 00:00:00 UTC. Data is updated once every 2 hours."
    mock_resp.raise_for_status = Mock()
    mock_get.return_value = mock_resp

    with pytest.raises(AstraError, match="rate limit"):
        fetch_celestrak_group("starlink")


@patch("astra.data._session.get")
def test_fetch_celestrak_fallback_to_supplemental_after_500(mock_get):
    """Legacy gp.php 503 triggers supplemental sup-gp.php with FILE=group."""
    bad = Mock()
    bad.status_code = 503
    bad.text = ""

    good = Mock()
    good.status_code = 200
    good.text = TLE_MOCK_DATA
    good.raise_for_status = Mock()

    mock_get.side_effect = [bad, good]

    catalog = fetch_celestrak_group("starlink")
    assert len(catalog) == 2
    assert mock_get.call_count == 2
    assert mock_get.call_args_list[0].args[0] == (
        "https://celestrak.org/NORAD/elements/gp.php?GROUP=starlink&FORMAT=tle"
    )
    assert mock_get.call_args_list[1].args[0] == (
        "https://celestrak.org/NORAD/elements/supplemental/sup-gp.php"
    )
    kwargs = mock_get.call_args_list[1].kwargs
    assert kwargs["params"] == {"FILE": "starlink", "FORMAT": "TLE"}
    assert kwargs["headers"] == _HEADERS


@patch("astra.data._session.get")
def test_fetch_celestrak_gps_ops_fallback_uses_gps_a_source(mock_get):
    """gps-ops supplemental mapping uses SOURCE=GPS-A (not FILE=gps-ops)."""
    bad = Mock()
    bad.status_code = 500
    bad.text = ""

    good = Mock()
    good.status_code = 200
    good.text = TLE_MOCK_DATA
    good.raise_for_status = Mock()

    mock_get.side_effect = [bad, good]

    fetch_celestrak_group("gps-ops")
    assert mock_get.call_args_list[1].kwargs["params"] == {
        "SOURCE": "GPS-A",
        "FORMAT": "TLE",
    }


@patch("astra.data._session.get")
def test_fetch_celestrak_active_no_supplemental_mapping_on_legacy_failure(mock_get):
    """GROUP=active has no sup-gp mapping; legacy failure surfaces clearly."""
    bad = Mock()
    bad.status_code = 500
    bad.text = ""

    mock_get.return_value = bad

    with pytest.raises(AstraError, match="no supplemental sup-gp"):
        fetch_celestrak_active()
