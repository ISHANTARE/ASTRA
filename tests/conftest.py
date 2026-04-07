import pytest
import numpy as np
from datetime import datetime, timezone

import astra

ISS_NAME = "ISS (ZARYA)"
ISS_LINE1 = "1 25544U 98067A   21001.00000000  .00001480  00000-0  34282-4 0  9990"
ISS_LINE2 = "2 25544  51.6442 284.1199 0001364 338.5498  21.5664 15.48922536 12341"
ISS_NORAD = "25544"

TEST_JD_START = 2460676.5  # 2025-01-01T00:00:00 UTC
TEST_JD_END   = 2460677.5  # 2025-01-02T00:00:00 UTC

@pytest.fixture
def iss_tle():
    return astra.parse_tle(ISS_NAME, ISS_LINE1, ISS_LINE2)


@pytest.fixture
def iss_omm():
    """ISS-like OMM record (same regime as ``iss_tle``) for format-agnostic API tests."""
    from astra.omm import parse_omm_record

    rec = {
        "OBJECT_NAME": ISS_NAME,
        "OBJECT_ID": "1998-067A",
        "NORAD_CAT_ID": ISS_NORAD,
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
        "MASS": "419725",
    }
    return parse_omm_record(rec)

@pytest.fixture
def sample_observer():
    from astra.models import Observer
    return Observer(
        name="Bangalore",
        latitude_deg=12.97,
        longitude_deg=77.59,
        elevation_m=920.0,
        min_elevation_deg=10.0
    )

@pytest.fixture
def observer():
    """Alias for sample_observer used by visibility tests."""
    from astra.models import Observer
    return Observer(
        name="Bangalore",
        latitude_deg=12.97,
        longitude_deg=77.59,
        elevation_m=920.0,
        min_elevation_deg=10.0
    )

@pytest.fixture
def time_steps():
    return np.arange(0.0, 24 * 60, 5.0)  # 288 steps

@pytest.fixture
def small_catalog():
    lines = [
        "ISS (ZARYA)",
        "1 25544U 98067A   21001.00000000  .00001480  00000-0  34282-4 0  9990",
        "2 25544  51.6442 284.1199 0001364 338.5498  21.5664 15.48922536 12341",
        "TIRUPATI",
        "1 44383U 19038A   21001.00000000  .00001480  00000-0  34282-4 0  9993",
        "2 44383  97.6442 284.1199 0001364 338.5498  21.5664 14.48922536 12342",
        "DEBRIS X",
        "1 99999D 98067A   21001.00000000  .00001480  00000-0  34282-4 0  9995",
        "2 99999  10.6442 284.1199 0001364 338.5498  21.5664  2.48922536 12347",
    ]
    return astra.load_tle_catalog(lines)
