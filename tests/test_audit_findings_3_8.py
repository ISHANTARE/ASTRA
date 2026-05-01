import logging

import numpy as np
import pytest


def test_lowering_hohmann_burns_are_retrograde_in_vnb_and_rtn():
    from astra.constants import EARTH_EQUATORIAL_RADIUS_KM
    from astra.maneuver import plan_hohmann
    from astra.models import ManeuverFrame

    r_high = EARTH_EQUATORIAL_RADIUS_KM + 700.0
    r_low = EARTH_EQUATORIAL_RADIUS_KM + 400.0

    burns_vnb = plan_hohmann(
        r_high, r_low, isp_s=300.0, mass_kg=1000.0, thrust_N=10.0,
        t_ignition_jd=2460000.5, frame=ManeuverFrame.VNB,
    )
    assert burns_vnb[0].direction == (-1.0, 0.0, 0.0)
    assert burns_vnb[1].direction == (-1.0, 0.0, 0.0)

    burns_rtn = plan_hohmann(
        r_high, r_low, isp_s=300.0, mass_kg=1000.0, thrust_N=10.0,
        t_ignition_jd=2460000.5, frame=ManeuverFrame.RTN,
    )
    assert burns_rtn[0].direction == (0.0, -1.0, 0.0)
    assert burns_rtn[1].direction == (0.0, -1.0, 0.0)


def test_powered_python_fallback_rotation_matches_numba_kernel():
    from astra.propagator import _acceleration, _powered_derivative, _powered_derivative_njit

    t_jd = 2460000.5
    y = np.array([7000.0, 0.0, 0.0, 0.0, 7.5, 0.0, 1000.0])
    empty_coeffs = np.zeros((1, 2, 3))
    common_args = (
        0.0, y, t_jd, False, 2.2, 10.0, 1000.0, 0.0, 50.0, 400.0,
        150.0, 150.0, 15.0, False, False, t_jd, 1.0,
        empty_coeffs, empty_coeffs, False, 1.5, True,
        10.0, 300.0, np.array([1.0, 0.0, 0.0]), 0, 10.0,
    )

    dy_py = _powered_derivative(*common_args)
    dy_nb = _powered_derivative_njit(*common_args)

    np.testing.assert_allclose(dy_py, dy_nb, rtol=1e-12, atol=1e-15)
    a_grav = _acceleration(
        2460000.5, y[:3], y[3:6], False, 2.2, 10.0, 1000.0, 0.0, 50.0, 400.0,
        150.0, 150.0, 15.0, False, False, 2460000.5, 1.0, empty_coeffs, empty_coeffs,
        False, 1.5, True, 10.0,
    )
    thrust_component = dy_py[3:6] - a_grav
    np.testing.assert_allclose(thrust_component, np.array([0.0, 1e-5, 0.0]), atol=1e-15)


def test_srp_area_is_independent_from_drag_area():
    from astra.propagator import _acceleration
    from astra.constants import AU_KM

    t_jd = 2460000.5
    r = np.array([7000.0, 0.0, 0.0])
    v = np.array([0.0, 7.5, 0.0])
    sun_coeffs = np.zeros((1, 2, 3))
    moon_coeffs = np.zeros((1, 2, 3))
    sun_coeffs[0, 0, :] = np.array([AU_KM, 0.0, 0.0])

    base_args = (
        t_jd, r, v, False, 2.2, 0.0, 1000.0, 0.0, 50.0, 400.0,
        150.0, 150.0, 15.0, False, True, t_jd, 1.0,
        sun_coeffs, moon_coeffs,
    )
    without_srp = _acceleration(*base_args, False, 1.5, False, 20.0)
    with_srp = _acceleration(*base_args, True, 1.5, False, 20.0)

    srp_delta = with_srp - without_srp
    assert np.linalg.norm(srp_delta) > 0.0
    assert srp_delta[0] < 0.0


def test_hcw_missing_mean_motion_warns_relaxed_and_fails_strict(caplog):
    import astra.config as cfg
    from astra.covariance import compute_collision_probability_mc

    prev = cfg.ASTRA_STRICT_MODE
    miss = np.array([0.02, 0.0, 0.0])
    rel_vel = np.array([0.0, 0.001, 0.0])
    cov = np.eye(6) * 1e-8
    try:
        cfg.ASTRA_STRICT_MODE = False
        caplog.set_level(logging.WARNING)
        pc = compute_collision_probability_mc(
            miss, rel_vel, cov, cov, n_samples=200, seed=7,
        )
        assert 0.0 <= pc <= 1.0
        assert "requires mean_motion_rad_s" in caplog.text

        cfg.ASTRA_STRICT_MODE = True
        with pytest.raises(ValueError, match="ASTRA STRICT"):
            compute_collision_probability_mc(
                miss, rel_vel, cov, cov, n_samples=10, seed=7,
            )
    finally:
        cfg.ASTRA_STRICT_MODE = prev


def test_snc_process_noise_is_carried_across_maneuver_segments():
    from astra.models import FiniteBurn, ManeuverFrame
    from astra.propagator import NumericalState, SNCConfig, propagate_cowell

    t_jd = 2460000.5
    state0 = NumericalState(
        t_jd=t_jd,
        position_km=np.array([7000.0, 0.0, 0.0]),
        velocity_km_s=np.array([0.0, 7.5, 0.0]),
        mass_kg=1000.0,
        covariance_km2=np.eye(6) * 1e-9,
    )
    burn = FiniteBurn(
        epoch_ignition_jd=t_jd + 60.0 / 86400.0,
        duration_s=20.0,
        thrust_N=0.1,
        isp_s=300.0,
        direction=(1.0, 0.0, 0.0),
        frame=ManeuverFrame.VNB,
    )

    states = propagate_cowell(
        state0,
        duration_s=180.0,
        dt_out=60.0,
        include_third_body=False,
        include_stm=True,
        maneuvers=[burn],
        snc_config=SNCConfig(mode="white_noise", q_psd_m2_s3=1e-3),
    )

    traces = [float(np.trace(s.covariance_km2)) for s in states if s.covariance_km2 is not None]
    assert len(traces) >= 4
    assert traces[-1] > 2.0 * traces[1]
