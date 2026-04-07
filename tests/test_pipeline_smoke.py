"""End-to-end smoke tests: OMM → propagate → debris → filter → conjunction."""

from __future__ import annotations

import json

import pytest

from astra.conjunction import find_conjunctions
from astra.debris import filter_altitude, make_debris_object
from astra.omm import parse_omm_json
from astra.orbit import propagate_orbit


# Reuse ISS-like OMM from test_omm (physically self-consistent LEO).
_ISS_OMM = {
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
    "MASS": "419725",
}


def test_pipeline_omm_parse_propagate_debris_filter():
    """Mock-fetch analogue: JSON → OMM → SGP4 point → DebrisObject → altitude filter."""
    payload = json.dumps([_ISS_OMM])
    sats = parse_omm_json(payload)
    assert len(sats) == 1
    sat = sats[0]

    state = propagate_orbit(satellite=sat, epoch_jd=sat.epoch_jd, t_since_minutes=0.0)
    assert state.position_km.shape == (3,)
    assert state.velocity_km_s.shape == (3,)

    obj = make_debris_object(sat)
    kept = filter_altitude([obj], min_km=200.0, max_km=600.0)
    assert len(kept) == 1
    assert kept[0].source.norad_id == "25544"


@pytest.fixture
def _two_omm_payload():
    second = dict(_ISS_OMM)
    second["NORAD_CAT_ID"] = "99998"
    second["OBJECT_NAME"] = "CHASE"
    return json.dumps([_ISS_OMM, second])


def test_pipeline_two_omm_conjunction_screening_smoke(_two_omm_payload):
    """Two OMM objects → propagated trajectories → find_conjunctions (no close event required)."""
    import numpy as np

    from astra.omm import parse_omm_json
    from astra.orbit import propagate_many

    sats = parse_omm_json(_two_omm_payload)
    assert len(sats) == 2

    times = np.linspace(sats[0].epoch_jd, sats[0].epoch_jd + 0.05, 32)
    trajs, _vels = propagate_many(sats, times)

    elements = {sat.norad_id: make_debris_object(sat) for sat in sats}

    events = find_conjunctions(trajs, times, elements, threshold_km=5.0)
    assert isinstance(events, list)
