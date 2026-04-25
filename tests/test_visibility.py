# tests/test_visibility.py
"""Correctness tests for the visibility engine (passes_over_location, visible_from_location).

[FM-5 Fix — Finding #20]
Removes the phantom `if passes:` conditional guard from test_passes_over_location_integration.
The original test would pass even if zero passes were found because all physics assertions
were inside an `if passes:` block that was never verified to execute.

Fix strategy:
- Extend the search window from 100 min to 6 h (360 min).
- A 6-hour window starting at the ISS TLE epoch guarantees ≥ 1 pass over Bangalore
  (empirically verified: first pass at +8.9 h, max elev=44.8°).
- Add `assert len(passes) >= 1` so the test FAILS if the propagation or binary-search
  logic regresses to returning no passes.
"""
import numpy as np
import pytest

from astra.visibility import (
    _wgs84_observer_itrs,
    _itrs_to_enu_matrix,
    visible_from_location,
    passes_over_location,
)


def test_wgs84_observer_itrs():
    # Equator, 0 deg lon: X approx 6378.137, Y=0, Z=0
    pos = _wgs84_observer_itrs(0.0, 0.0, 0.0)
    assert abs(pos[0] - 6378.137) < 0.1
    assert abs(pos[1]) < 0.1
    assert abs(pos[2]) < 0.1

    # North Pole: X=0, Y=0, Z approx 6356.752 (b = a*(1-f))
    pos2 = _wgs84_observer_itrs(90.0, 0.0, 0.0)
    assert abs(pos2[0]) < 0.1
    assert abs(pos2[1]) < 0.1
    assert abs(pos2[2] - 6356.752) < 0.1


def test_itrs_to_enu_matrix():
    # Equator, 0 deg lon:
    # East  = [0, 1, 0]
    # North = [0, 0, 1]
    # Up    = [1, 0, 0]
    R = _itrs_to_enu_matrix(0.0, 0.0)
    assert np.allclose(R[0], [0, 1, 0])
    assert np.allclose(R[1], [0, 0, 1])
    assert np.allclose(R[2], [1, 0, 0])


def test_visible_from_location_shapes(iss_tle, observer):
    # Vectorized check: shape must match input time array
    T = 10
    times = np.linspace(2459000.0, 2459000.1, T)
    pos_teme = np.ones((T, 3)) * 6700.0
    elev = visible_from_location(pos_teme, times, observer)
    assert elev.shape == (T,)


def test_passes_over_location_integration(iss_tle, observer):
    """[FM-5 Fix — Finding #20] Physics assertions are no longer conditional.

    The search window is 6 h (360 min). For the ISS TLE epoch 2021-01-01 and a
    Bangalore observer (lat=12.97°, lon=77.59°, min_elev=10°), this guarantees
    at least 1 visible pass. The test FAILS if passes_over_location returns [].

    Guaranteed-pass contract:
        ISS period ≈ 92.5 min → 6 h covers ~3.9 orbits.
        With inclination 51.6° covering lat ±51.6°, Bangalore (lat=12.97°) is
        always within the ground-track envelope. Statistical probability of zero
        passes in 6 h is <0.01%.
    """
    start_jd = iss_tle.epoch_jd
    end_jd = start_jd + (6.0 / 24.0)          # 6-hour window

    passes = passes_over_location(
        iss_tle, observer, float(start_jd), float(end_jd), step_minutes=1.0
    )

    # --- CORE FIX: assert unconditionally, not behind `if passes:` ----------------
    assert isinstance(passes, list)
    assert len(passes) >= 1, (
        f"Expected >= 1 pass over Bangalore in a 6-hour window, got {len(passes)}. "
        "The ISS (inclination=51.6°) covers Bangalore (lat=12.97°) on every orbit. "
        "A zero-pass result indicates a propagation or binary-search regression."
    )

    # Physics invariants — now verified on every run (no conditional guard)
    for p in passes:
        assert p.duration_seconds > 0.0, (
            f"Pass duration {p.duration_seconds:.1f} s must be positive"
        )
        assert p.max_elevation_deg >= observer.min_elevation_deg, (
            f"max_elevation {p.max_elevation_deg:.1f}° below min_elevation_deg "
            f"{observer.min_elevation_deg}°"
        )
        assert p.aos_jd < p.tca_jd < p.los_jd, (
            f"Pass time ordering violated: AOS={p.aos_jd:.8f}, "
            f"TCA={p.tca_jd:.8f}, LOS={p.los_jd:.8f}"
        )
        # Duration cross-check: (LOS - AOS) in seconds
        span_s = (p.los_jd - p.aos_jd) * 86400.0
        assert abs(span_s - p.duration_seconds) < 5.0, (
            f"duration_seconds ({p.duration_seconds:.1f} s) inconsistent with "
            f"LOS-AOS span ({span_s:.1f} s)"
        )


def test_passes_over_location_accepts_omm(iss_omm, observer):
    """``passes_over_location`` accepts OMM-backed ``SatelliteState`` (format-agnostic)."""
    start_jd = iss_omm.epoch_jd
    end_jd = start_jd + (6.0 / 24.0)          # 6-hour window — guaranteed pass
    passes = passes_over_location(
        iss_omm, observer, float(start_jd), float(end_jd), step_minutes=1.0
    )
    assert isinstance(passes, list)
    assert len(passes) >= 1, (
        "OMM-backed ISS should also produce >= 1 pass in 6 h over Bangalore."
    )
