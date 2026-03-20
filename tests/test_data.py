import pytest
from unittest.mock import patch, Mock

from astra.data import fetch_celestrak_active, fetch_celestrak_group
from astra.errors import AstraError

CEL_MOCK_DATA = """ISS (ZARYA)
1 25544U 98067A   21001.00000000  .00001480  00000-0  34282-4 0  9990
2 25544  51.6442 284.1199 0001364 338.5498  21.5664 15.48922536 12345
STARLINK-1
1 44235U 19029A   21001.00000000  .00001480  00000-0  34282-4 0  9990
2 44235  53.0500 284.1199 0001364 338.5498  21.5664 15.48922536 12345"""

@patch("requests.get")
def test_fetch_celestrak_active(mock_get):
    mock_resp = Mock()
    mock_resp.text = CEL_MOCK_DATA
    mock_resp.raise_for_status = Mock()
    mock_get.return_value = mock_resp
    
    catalog = fetch_celestrak_active()
    assert len(catalog) == 2
    assert catalog[0].name == "ISS (ZARYA)"

@patch("requests.get")
def test_fetch_celestrak_group(mock_get):
    mock_resp = Mock()
    mock_resp.text = CEL_MOCK_DATA
    mock_resp.raise_for_status = Mock()
    mock_get.return_value = mock_resp
    
    catalog = fetch_celestrak_group("starlink")
    assert len(catalog) == 2
    mock_get.assert_called_with("https://celestrak.org/NORAD/elements/gp.php?GROUP=starlink&FORMAT=tle", timeout=15.0)

@patch("requests.get")
def test_fetch_celestrak_error(mock_get):
    import requests
    mock_get.side_effect = requests.RequestException("API down")
    
    with pytest.raises(AstraError):
        fetch_celestrak_active()
