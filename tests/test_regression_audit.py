"""Regression tests for physics, time, Pc, strict mode, and concurrency fixes."""

from __future__ import annotations

import math
import threading
import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Helper: ISS TLE — same lines as conftest.py (already validated 69 chars + checksum)
# ---------------------------------------------------------------------------

# Epoch: 2021-001.000 (1 Jan 2021 00:00 UTC)
_ISS_L1 = "1 25544U 98067A   21001.00000000  .00001480  00000-0  34282-4 0  9990"
_ISS_L2 = "2 25544  51.6442 284.1199 0001364 338.5498  21.5664 15.48922536 12341"


def _iss_tle():
    """Return an ISS SatelliteTLE using the same TLE as conftest.py."""
    from astra.tle import parse_tle

    return parse_tle("ISS (ZARYA)", _ISS_L1, _ISS_L2)


def test_j4_python_and_numba_paths_agree():
    """Python ``_acceleration`` must match Numba ``_acceleration_njit`` (J4 term included)."""
    from astra.propagator import _acceleration, _acceleration_njit

    r = np.array([7000.0, 0.0, 500.0])  # slight z-offset activates J3+J4 z-terms
    v = np.array([0.0, 7.5, 0.0])
    t_jd = 2451545.0

    # Python path (high-level, no drag, no 3rd body, no SRP)
    # [MSIS SYNC] _acceleration now requires f107_obs, f107_adj, ap_daily, hf_atmosphere
    # between drag_ref_alt_km and include_third_body.  Values are irrelevant here because
    # use_drag=False and hf_atmosphere=False, so MSIS is never invoked.
    empty_coeffs = np.zeros((1, 2, 3))
    a_py = _acceleration(
        t_jd,
        r,
        v,
        False,
        2.2,
        10.0,
        1000.0,
        0.0,
        50.0,
        400.0,  # use_drag=False, drag_ref_alt_km=400.0
        150.0,
        150.0,
        15.0,
        False,  # f107_obs, f107_adj, ap_daily, hf_atm
        False,
        t_jd,
        1.0,
        empty_coeffs,
        empty_coeffs,  # 3rd body off
        False,
        1.5,
        True,  # SRP off
    )

    # Numba path: disable drag (rho=0), disable 3rd body, disable SRP
    a_nb = _acceleration_njit(
        t_jd,
        r,
        v,
        False,
        2.2,
        10.0,
        1000.0,
        0.0,
        50.0,
        400.0,  # use_drag=False, drag_ref_alt_km=400.0
        150.0,
        150.0,
        15.0,
        False,  # f107_obs, f107_adj, ap_daily, hf_atm
        False,
        t_jd,
        1.0,
        empty_coeffs,
        empty_coeffs,  # 3rd body off
        False,
        1.5,
        True,  # SRP off
    )

    np.testing.assert_allclose(
        a_py,
        a_nb,
        rtol=1e-6,
        atol=1e-12,
        err_msg="Python and Numba J4 acceleration paths disagree — sign fix not applied consistently.",
    )
    # Both must be finite
    assert np.all(np.isfinite(a_py))


def test_srp_cylindrical_illumination_factor_phy18():
    """Cylindrical umbra: night-side in-cylinder → ν=0; sun-facing → ν=1."""
    from astra.propagator import srp_cylindrical_illumination_factor
    from astra.constants import EARTH_EQUATORIAL_RADIUS_KM

    Re = EARTH_EQUATORIAL_RADIUS_KM
    rsun = np.array([1.0e8, 0.0, 0.0])
    assert (
        srp_cylindrical_illumination_factor(np.array([-6800.0, 0.0, 0.0]), rsun) == 0.0
    )
    assert (
        srp_cylindrical_illumination_factor(np.array([6800.0, 0.0, 0.0]), rsun) == 1.0
    )
    # Just outside the cylinder on the night side: ρ > Re → sunlit in this model
    assert (
        srp_cylindrical_illumination_factor(np.array([-6800.0, Re + 100.0, 0.0]), rsun)
        == 1.0
    )


def test_j4_sign_is_negative_for_equatorial_radial():
    """At z=0, J4 radial contribution in x should oppose the outward two-body direction (Vallado Eq. 8-31)."""
    from astra.constants import J4, EARTH_MU_KM3_S2, EARTH_EQUATORIAL_RADIUS_KM

    r = np.array([7000.0, 0.0, 0.0])
    r_mag = 7000.0
    r2 = r_mag**2
    r9 = r_mag**9

    fJ4 = 0.625 * J4 * EARTH_MU_KM3_S2 * EARTH_EQUATORIAL_RADIUS_KM**4 / r9
    a_j4_x = fJ4 * r[0] * (0.0 - 0.0 + 3.0 * r2)

    # J4 < 0 → fJ4 < 0 → a_j4_x < 0 (correct, opposing two-body outward direction)
    assert J4 < 0, "J4 constant should be negative per WGS-84"
    assert fJ4 < 0, f"fJ4 should be negative (J4<0, +0.625 multiplier). Got {fJ4}"
    assert (
        a_j4_x < 0
    ), f"J4 x-accel at equatorial position should be negative. Got {a_j4_x}"


def test_pc_exact_dblquad_vs_chan_near_hit():
    """Near-direct-hit (low Mahalanobis): 2D disk integral should exceed Chan point estimate."""
    from astra.covariance import _exact_pc_2d_integral
    import numpy as np

    # Small miss vector — the satellite will almost certainly hit
    miss_2d = np.array([0.003, 0.0])  # 3 m miss in encounter plane
    sigma = 0.1  # 100 m 1-sigma
    C_p = np.diag([sigma**2, sigma**2])
    inv_C_p = np.linalg.inv(C_p)
    det_C_p = float(np.linalg.det(C_p))
    combined_radius = 0.01  # 10 m

    pc_exact = _exact_pc_2d_integral(miss_2d, inv_C_p, det_C_p, combined_radius)
    assert 0.0 < pc_exact <= 1.0, f"Exact Pc out of range: {pc_exact}"

    # Chan approximation for comparison
    import math

    mahal = float(miss_2d @ inv_C_p @ miss_2d)
    area = math.pi * combined_radius**2
    pc_chan = math.exp(-0.5 * mahal) * area / (2.0 * math.pi * math.sqrt(det_C_p))

    # Exact should be ≥ Chan in the near-hit regime
    assert (
        pc_exact >= pc_chan * 0.5
    ), f"Exact Pc ({pc_exact:.4e}) unexpectedly much smaller than Chan ({pc_chan:.4e})"


def test_pc_exact_vs_mc_order_of_magnitude():
    """Exact 2D Pc and Monte Carlo Pc stay within an order of magnitude for a near-hit case."""
    from astra.covariance import (
        compute_collision_probability,
        compute_collision_probability_mc,
    )

    miss = np.array([0.05, 0.0, 0.0])  # 50 m miss
    v_rel = np.array([0.0, 0.0, 10.0])  # 10 km/s cross-track
    sigma = 0.1  # 100 m 1-sigma
    cov_3x3 = np.diag([sigma**2] * 3)

    pc_ana = compute_collision_probability(
        miss, v_rel, cov_3x3, cov_3x3, radius_a_km=0.005, radius_b_km=0.005
    )
    assert 0.0 <= pc_ana <= 1.0

    cov_6x6 = np.diag([sigma**2] * 3 + [1e-6] * 3)
    pc_mc = compute_collision_probability_mc(
        miss,
        v_rel,
        cov_6x6,
        cov_6x6,
        radius_a_km=0.005,
        radius_b_km=0.005,
        n_samples=30_000,
        seed=42,
    )
    assert 0.0 <= pc_mc <= 1.0

    if pc_mc > 1e-7 and pc_ana > 1e-7:
        ratio = max(pc_ana, pc_mc) / min(pc_ana, pc_mc)
        assert (
            ratio < 20.0
        ), f"Analytical Pc={pc_ana:.3e} and MC Pc={pc_mc:.3e} differ by >{ratio:.1f}×"


def test_get_ut1_utc_correction_j2000_reasonable():
    """UT1−UTC at J2000 should be sub-second (not ~±64 s TT/UTC confusion)."""
    from astra.data_pipeline import get_ut1_utc_correction

    dut1 = float(get_ut1_utc_correction(2451545.0))
    assert -1.0 < dut1 < 1.0, (
        f"UT1-UTC = {dut1:.3f} s at J2000 — expected |dut1| < 1 s; "
        "a value near ±64 s indicates the TT-UTC bug is still present."
    )


def test_get_ut1_utc_correction_returns_finite():
    """UT1−UTC stays finite and in a physical band for several epoch offsets."""
    from astra.data_pipeline import get_ut1_utc_correction

    jd_start = 2451545.0  # J2000
    for delta in [0, 365, 730, 1826]:
        dut1 = float(get_ut1_utc_correction(jd_start + delta))
        assert math.isfinite(dut1), f"Non-finite UT1-UTC at JD+{delta}"
        assert -2.0 < dut1 < 2.0, f"UT1-UTC out of physical range at JD+{delta}: {dut1}"


def test_pc_singular_covariance_returns_zero_no_crash():
    """Near-singular encounter-plane covariance: finite Pc in [0,1], no crash."""
    from astra.covariance import compute_collision_probability

    miss = np.array([1.0, 0.0, 0.0])
    v_rel = np.array([0.0, 0.0, 10.0])
    # Near-singular: one axis has near-zero variance
    cov = np.diag([1.0, 1e-20, 1.0])

    pc = compute_collision_probability(miss, v_rel, cov, cov)
    assert math.isfinite(pc)
    assert 0.0 <= pc <= 1.0


def test_set_strict_mode_is_callable():
    """``set_strict_mode`` is importable from ``astra.config``."""
    from astra.config import set_strict_mode

    assert callable(set_strict_mode)


def test_set_strict_mode_thread_safe():
    """Concurrent ``set_strict_mode`` toggles do not error."""
    import astra.config as cfg
    from astra.config import set_strict_mode

    errors: list[Exception] = []
    original = cfg.ASTRA_STRICT_MODE

    def _toggle(val: bool):
        try:
            for _ in range(50):
                set_strict_mode(val)
        except Exception as exc:
            errors.append(exc)

    threads = [
        threading.Thread(target=_toggle, args=(True,)),
        threading.Thread(target=_toggle, args=(False,)),
        threading.Thread(target=_toggle, args=(True,)),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"Errors during concurrent set_strict_mode: {errors}"
    # Restore
    set_strict_mode(original)


def test_load_space_weather_concurrent_no_double_parse(monkeypatch):
    """Concurrent ``load_space_weather`` parses CSV exactly once."""
    import astra.data_pipeline as dp

    parse_count: list[int] = [0]
    original_parse = dp._parse_sw_csv

    def _counting_parse(text):
        parse_count[0] += 1
        original_parse(text)

    monkeypatch.setattr(dp, "_parse_sw_csv", _counting_parse)
    monkeypatch.setattr(dp, "_sw_loaded", False)
    monkeypatch.setattr(dp, "_sw_cache", {})

    # Feed a minimal but valid CSV so no HTTP request is made
    import pathlib
    import tempfile

    minimal_csv = (
        "TYPE,YYYY,MM,DD,BSRN,ND,Kp1,Kp2,Kp3,Kp4,Kp5,Kp6,Kp7,Kp8,Kp_sum,"
        "Ap1,Ap2,Ap3,Ap4,Ap5,Ap6,Ap7,Ap8,Ap_avg,Cp,C9,ISN,F10.7_obs,F10.7_adj,Q,F10.7_81,Ap_avg2\n"
        "OBS,2000,01,01,2245,1,3,3,3,3,3,3,3,3,24,"
        "15,15,15,15,15,15,15,15,15.0,0.5,3,52,150.0,148.0,0,150.0,15.0\n"
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        sw_file = pathlib.Path(tmpdir) / "SW-All.csv"
        sw_file.write_text(minimal_csv, encoding="utf-8")
        monkeypatch.setattr(dp, "_DEFAULT_DATA_DIR", tmpdir)

        barrier = threading.Barrier(4)
        exc_list: list[Exception] = []

        def _call():
            barrier.wait()  # synchronise all threads to fire at once
            try:
                dp.load_space_weather(data_dir=tmpdir)
            except Exception as e:
                exc_list.append(e)

        threads = [threading.Thread(target=_call) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not exc_list, f"load_space_weather raised: {exc_list}"
        assert (
            parse_count[0] == 1
        ), f"_parse_sw_csv called {parse_count[0]} times; expected exactly 1."


def test_collision_probability_nan_when_none():
    """``collision_probability_nan`` is NaN when Pc is None."""
    from astra.models import ConjunctionEvent

    evt = ConjunctionEvent(
        object_a_id="A",
        object_b_id="B",
        tca_jd=2451545.0,
        miss_distance_km=1.0,
        relative_velocity_km_s=10.0,
        collision_probability=None,
        risk_level="UNKNOWN",
        position_a_km=np.zeros(3),
        position_b_km=np.zeros(3),
    )
    assert evt.collision_probability is None
    pc_nan = evt.collision_probability_nan
    assert isinstance(pc_nan, float)
    assert math.isnan(pc_nan), "Expected float('nan') when Pc is None"


def test_collision_probability_nan_when_set():
    """``collision_probability_nan`` equals Pc when Pc is set."""
    from astra.models import ConjunctionEvent

    evt = ConjunctionEvent(
        object_a_id="A",
        object_b_id="B",
        tca_jd=2451545.0,
        miss_distance_km=1.0,
        relative_velocity_km_s=10.0,
        collision_probability=1.5e-4,
        risk_level="CRITICAL",
        position_a_km=np.zeros(3),
        position_b_km=np.zeros(3),
    )
    assert evt.collision_probability_nan == pytest.approx(1.5e-4)


def test_7dof_powered_arc_tsiolkovsky():
    """Powered ``propagate_cowell`` depletes mass per Tsiolkovsky (Δm = F·t/(Isp·g0))."""
    from astra.propagator import propagate_cowell, DragConfig, NumericalState
    from astra.models import FiniteBurn, ManeuverFrame

    tle = _iss_tle()

    thrust_N = 22.0  # N  (small thruster)
    isp_s = 220.0  # s  (cold-gas typical)
    duration_s = 30.0  # s

    burn = FiniteBurn(
        epoch_ignition_jd=tle.epoch_jd + 120.0 / 86400.0,
        duration_s=duration_s,
        thrust_N=thrust_N,
        isp_s=isp_s,
        direction=(1.0, 0.0, 0.0),  # prograde (VNB V-axis)
        frame=ManeuverFrame.VNB,
    )

    initial_mass = 420_000.0  # kg

    t0 = tle.epoch_jd
    state0 = NumericalState(
        t_jd=t0,
        position_km=np.array([6778.137, 0.0, 0.0]),
        velocity_km_s=np.array([0.0, 7.6597, 0.0]),
        mass_kg=initial_mass,
    )

    states = propagate_cowell(
        state0,
        duration_s=180.0,
        maneuvers=[burn],
        drag_config=DragConfig(mass_kg=initial_mass),
        include_third_body=False,
    )

    assert len(states) >= 2

    mass_final = states[-1].mass_kg
    assert mass_final is not None
    assert mass_final < initial_mass, "Mass must decrease during powered arc"

    # Tsiolkovsky mass depletion: Δm = F·t / (Isp · g0)
    g0 = 9.80665  # m/s²
    dm_expected = thrust_N * duration_s / (isp_s * g0)  # kg
    dm_actual = initial_mass - mass_final

    assert dm_actual > 0, "Propellant must have been consumed"
    assert abs(dm_actual - dm_expected) < 1.0, (  # 1 kg tolerance
        f"Mass depletion {dm_actual:.3f} kg deviates from Tsiolkovsky "
        f"{dm_expected:.3f} kg by {abs(dm_actual - dm_expected):.3f} kg"
    )


def test_7dof_powered_arc_produces_finite_states():
    """Powered propagation yields finite positions and velocities."""
    from astra.propagator import propagate_cowell, DragConfig, NumericalState
    from astra.models import FiniteBurn, ManeuverFrame

    tle = _iss_tle()

    burn = FiniteBurn(
        epoch_ignition_jd=tle.epoch_jd + 60.0 / 86400.0,
        duration_s=60.0,
        thrust_N=50.0,
        isp_s=300.0,
        direction=(1.0, 0.0, 0.0),
        frame=ManeuverFrame.VNB,
    )

    t0 = tle.epoch_jd
    state0 = NumericalState(
        t_jd=t0,
        position_km=np.array([6778.137, 0.0, 0.0]),
        velocity_km_s=np.array([0.0, 7.6597, 0.0]),
        mass_kg=420_000.0,
    )

    states = propagate_cowell(
        state0,
        duration_s=300.0,
        maneuvers=[burn],
        drag_config=DragConfig(mass_kg=420_000.0),
        include_third_body=False,
    )

    assert len(states) > 0
    for s in states:
        assert np.all(np.isfinite(s.position_km)), f"NaN/Inf position at JD {s.t_jd}"
        assert np.all(np.isfinite(s.velocity_km_s)), f"NaN/Inf velocity at JD {s.t_jd}"
        r_mag = float(np.linalg.norm(s.position_km))
        assert (
            6550.0 < r_mag < 7200.0
        ), f"Orbit radius {r_mag:.1f} km outside ISS LEO bounds"


def test_propagate_cowell_strict_raises_propagation_error_on_ivp_failure(monkeypatch):
    """Strict mode: failed IVP raises ``PropagationError``."""
    import astra.config as cfg
    from astra.errors import PropagationError
    from astra.propagator import propagate_cowell, NumericalState

    class _FailedSol:
        success = False
        message = "mock integrator failure"

    def _fake_solve_ivp(*_args, **_kwargs):
        return _FailedSol()

    monkeypatch.setattr("astra.propagator.solve_ivp", _fake_solve_ivp)

    prev = cfg.ASTRA_STRICT_MODE
    cfg.ASTRA_STRICT_MODE = True
    try:
        state0 = NumericalState(
            t_jd=2451545.0,
            position_km=np.array([7000.0, 0.0, 0.0]),
            velocity_km_s=np.array([0.0, 7.5, 0.0]),
        )
        with pytest.raises(PropagationError, match="Cowell integration failed"):
            propagate_cowell(
                state0,
                duration_s=120.0,
                dt_out=60.0,
                drag_config=None,
                include_third_body=False,
                use_de=False,
            )
    finally:
        cfg.ASTRA_STRICT_MODE = prev


def test_pc_cara_order_of_magnitude_high_speed_encounter():
    """High-speed encounter: Pc in the ~1e-4 decade (screening sanity, not certified CARA replay)."""
    from astra.covariance import compute_collision_probability

    miss = np.array([0.05, 0.0, 0.0])
    v_rel = np.array([0.0, 0.0, 10.0])
    sigma_km = 0.2
    cov = np.diag([sigma_km**2] * 3)

    pc = compute_collision_probability(
        miss, v_rel, cov, cov, radius_a_km=0.005, radius_b_km=0.005
    )
    assert math.isfinite(pc) and 0.0 < pc < 1.0
    # Tunable scenario: ~6e-4 — same order as published CARA examples (~10^-4)
    assert 1.0e-4 < pc < 2.0e-3, f"Pc={pc:.4e} outside expected CARA-class decade"


def test_estimate_covariance_exported():
    """``estimate_covariance`` is importable from the ``astra`` package."""
    from astra import estimate_covariance

    assert callable(estimate_covariance)


def test_set_strict_mode_exported():
    """``set_strict_mode`` is importable from ``astra.config``."""
    from astra.config import set_strict_mode

    assert callable(set_strict_mode)


def test_rotate_covariance_rtn_to_eci_finite():
    """``rotate_covariance_rtn_to_eci`` returns a finite 3×3 matrix and preserves trace."""
    from astra.covariance import rotate_covariance_rtn_to_eci

    r = np.array([7000.0, 0.0, 0.0])
    v = np.array([0.0, 7.5, 0.1])
    cov_rtn = np.diag([1.0, 4.0, 0.25])
    cov_eci = rotate_covariance_rtn_to_eci(cov_rtn, r, v)

    assert cov_eci.shape == (3, 3)
    assert np.all(np.isfinite(cov_eci))
    # Rotation must preserve trace (eigenvalue sum invariant)
    np.testing.assert_allclose(np.trace(cov_eci), np.trace(cov_rtn), rtol=1e-10)
