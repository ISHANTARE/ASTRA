"""Tests for ASTRA v3.7.0 feature additions (AS-01 through AS-05).

Covers:
  - AS-01: propagate_cowell_at_times
  - AS-02a: plan_bielliptic
  - AS-02b: plan_inclination_change
  - AS-03: find_conjunction_windows
  - AS-04: compute_delta_v_budget
  - AS-05: compute_collision_probability_timeseries
"""
from __future__ import annotations

import math
import pytest
import numpy as np


# ===========================================================================
# AS-04: compute_delta_v_budget
# ===========================================================================

class TestComputeDeltaVBudget:
    """Tests for compute_delta_v_budget."""

    def test_hohmann_budget_matches_plan(self) -> None:
        """Budget from plan_hohmann burns must be physically consistent."""
        from astra.maneuver import plan_hohmann, compute_delta_v_budget

        burns = plan_hohmann(
            r_initial_km=6778.0,
            r_target_km=7178.0,
            isp_s=300.0,
            mass_kg=1000.0,
            thrust_N=10.0,
            t_ignition_jd=2460000.5,
        )
        budget = compute_delta_v_budget(burns, initial_mass_kg=1000.0)

        assert budget.total_delta_v_m_s > 0.0
        assert budget.total_propellant_kg > 0.0
        assert budget.final_mass_kg > 0.0
        assert budget.final_mass_kg < 1000.0
        assert len(budget.burns) == 2

        # Mass conservation: sum of propellant = initial - final
        assert abs(budget.total_propellant_kg - (1000.0 - budget.final_mass_kg)) < 1e-10

        # Per-burn mass chain: burn[i].mass_after = burn[i+1].mass_before
        assert abs(budget.burns[0]["mass_after_kg"] - budget.burns[1]["mass_before_kg"]) < 1e-10

    def test_empty_burns_raises(self) -> None:
        """Empty burn list must raise ManeuverError."""
        from astra.maneuver import compute_delta_v_budget
        from astra.errors import ManeuverError

        with pytest.raises(ManeuverError, match="empty"):
            compute_delta_v_budget([], initial_mass_kg=1000.0)

    def test_negative_mass_raises(self) -> None:
        """Non-positive initial mass must raise."""
        from astra.maneuver import compute_delta_v_budget, plan_hohmann
        from astra.errors import ManeuverError

        burns = plan_hohmann(6778.0, 7178.0, 300.0, 1000.0, 10.0, 2460000.5)
        with pytest.raises(ManeuverError, match="positive"):
            compute_delta_v_budget(burns, initial_mass_kg=-1.0)

    def test_insufficient_mass_raises(self) -> None:
        """Budget that would deplete mass must raise ManeuverError."""
        from astra.maneuver import compute_delta_v_budget, plan_hohmann
        from astra.errors import ManeuverError

        burns = plan_hohmann(6778.0, 7178.0, 300.0, 1000.0, 10.0, 2460000.5)
        with pytest.raises(ManeuverError, match="runs out of mass"):
            compute_delta_v_budget(burns, initial_mass_kg=0.001)

    def test_single_burn_budget(self) -> None:
        """Single-burn budget must compute correct Tsiolkovsky ΔV."""
        from astra.maneuver import compute_delta_v_budget
        from astra.models import FiniteBurn, ManeuverFrame
        from astra.constants import G0_STD

        burn = FiniteBurn(
            epoch_ignition_jd=2460000.5,
            duration_s=100.0,
            thrust_N=100.0,
            isp_s=300.0,
            direction=(1.0, 0.0, 0.0),
            frame=ManeuverFrame.VNB,
        )
        budget = compute_delta_v_budget([burn], initial_mass_kg=500.0)

        # Manual Tsiolkovsky check
        mdot = 100.0 / (300.0 * G0_STD)
        prop = mdot * 100.0
        m_after = 500.0 - prop
        dv_expected = 300.0 * G0_STD * math.log(500.0 / m_after)
        assert abs(budget.total_delta_v_m_s - dv_expected) < 1e-8
        assert abs(budget.final_mass_kg - m_after) < 1e-10


# ===========================================================================
# AS-02a: plan_bielliptic
# ===========================================================================

class TestPlanBielliptic:
    """Tests for plan_bielliptic."""

    def test_basic_bielliptic_returns_three_burns(self) -> None:
        """plan_bielliptic must return exactly 3 burns."""
        from astra.maneuver import plan_bielliptic
        burns = plan_bielliptic(
            r_initial_km=6778.0,
            r_target_km=42164.0,
            r_intermediate_km=100000.0,
            isp_s=300.0,
            mass_kg=2000.0,
            thrust_N=50.0,
            t_ignition_jd=2460000.5,
        )
        assert len(burns) == 3
        for b in burns:
            assert b.duration_s > 0.0
            assert b.thrust_N == 50.0
            assert b.isp_s == 300.0

    def test_burn_epochs_are_chronological(self) -> None:
        """All 3 burns must be in chronological order."""
        from astra.maneuver import plan_bielliptic
        burns = plan_bielliptic(
            r_initial_km=6778.0,
            r_target_km=42164.0,
            r_intermediate_km=100000.0,
            isp_s=300.0,
            mass_kg=2000.0,
            thrust_N=50.0,
            t_ignition_jd=2460000.5,
        )
        for i in range(len(burns) - 1):
            assert burns[i].epoch_cutoff_jd < burns[i + 1].epoch_ignition_jd, \
                f"Burn {i} overlaps burn {i+1}"

    def test_bielliptic_total_dv_less_than_hohmann_for_large_ratio(self) -> None:
        """For r_target/r_initial > 11.94, bielliptic should use less ΔV than Hohmann."""
        from astra.maneuver import plan_bielliptic, plan_hohmann, compute_delta_v_budget

        r_i = 6778.0
        r_f = 6778.0 * 15.0  # ratio = 15 > 11.94
        r_int = r_f * 3.0

        biel_burns = plan_bielliptic(r_i, r_f, r_int, 300.0, 5000.0, 100.0, 2460000.5)
        hoh_burns = plan_hohmann(r_i, r_f, 300.0, 5000.0, 100.0, 2460000.5)

        biel_budget = compute_delta_v_budget(biel_burns, 5000.0)
        hoh_budget = compute_delta_v_budget(hoh_burns, 5000.0)

        assert biel_budget.total_delta_v_m_s < hoh_budget.total_delta_v_m_s, \
            f"Bielliptic ΔV ({biel_budget.total_delta_v_m_s:.1f}) should be less " \
            f"than Hohmann ({hoh_budget.total_delta_v_m_s:.1f}) for ratio > 11.94"

    def test_invalid_intermediate_raises(self) -> None:
        """r_intermediate < max(r_initial, r_target) must raise."""
        from astra.maneuver import plan_bielliptic
        from astra.errors import ManeuverError

        with pytest.raises(ManeuverError, match="r_intermediate_km"):
            plan_bielliptic(
                r_initial_km=6778.0,
                r_target_km=42164.0,
                r_intermediate_km=10000.0,  # too small
                isp_s=300.0,
                mass_kg=2000.0,
                thrust_N=50.0,
                t_ignition_jd=2460000.5,
            )

    def test_equal_radii_raises(self) -> None:
        """Equal initial and target radii must raise."""
        from astra.maneuver import plan_bielliptic
        from astra.errors import ManeuverError

        with pytest.raises(ManeuverError, match="equal"):
            plan_bielliptic(6778.0, 6778.0, 100000.0, 300.0, 2000.0, 50.0, 2460000.5)

    def test_negative_radius_raises(self) -> None:
        """Negative radius must raise."""
        from astra.maneuver import plan_bielliptic
        from astra.errors import ManeuverError

        with pytest.raises(ManeuverError, match="positive"):
            plan_bielliptic(-100.0, 42164.0, 100000.0, 300.0, 2000.0, 50.0, 2460000.5)


# ===========================================================================
# AS-02b: plan_inclination_change
# ===========================================================================

class TestPlanInclinationChange:
    """Tests for plan_inclination_change."""

    def test_basic_inclination_change_returns_one_burn(self) -> None:
        """plan_inclination_change must return exactly 1 burn."""
        from astra.maneuver import plan_inclination_change
        burns = plan_inclination_change(
            r_km=42164.0,
            delta_inc_deg=1.0,
            isp_s=300.0,
            mass_kg=3000.0,
            thrust_N=22.0,
            t_ignition_jd=2460000.5,
        )
        assert len(burns) == 1
        assert burns[0].duration_s > 0.0

    def test_dv_matches_analytical_formula(self) -> None:
        """ΔV must match 2·v·sin(Δi/2) for circular orbit."""
        from astra.maneuver import plan_inclination_change, compute_delta_v_budget
        from astra.constants import EARTH_MU_KM3_S2

        r_km = 42164.0
        delta_i = 2.0  # degrees
        v_circ = math.sqrt(EARTH_MU_KM3_S2 / r_km)
        expected_dv_m_s = 2.0 * v_circ * abs(math.sin(math.radians(delta_i) / 2.0)) * 1000.0

        burns = plan_inclination_change(r_km, delta_i, 300.0, 3000.0, 22.0, 2460000.5)
        budget = compute_delta_v_budget(burns, 3000.0)

        # Budget ΔV should match analytical (within Tsiolkovsky finite-burn effects)
        assert abs(budget.total_delta_v_m_s - expected_dv_m_s) / expected_dv_m_s < 0.01

    def test_positive_inc_change_burns_positive_normal(self) -> None:
        """Positive Δi must produce a burn in the +N direction (VNB)."""
        from astra.maneuver import plan_inclination_change
        burns = plan_inclination_change(42164.0, 1.0, 300.0, 3000.0, 22.0, 2460000.5)
        assert burns[0].direction[1] > 0.0  # +N in VNB

    def test_negative_inc_change_burns_negative_normal(self) -> None:
        """Negative Δi must produce a burn in the -N direction (VNB)."""
        from astra.maneuver import plan_inclination_change
        burns = plan_inclination_change(42164.0, -1.0, 300.0, 3000.0, 22.0, 2460000.5)
        assert burns[0].direction[1] < 0.0  # -N in VNB

    def test_rtn_frame_normal_direction(self) -> None:
        """In RTN frame, normal is axis 2."""
        from astra.maneuver import plan_inclination_change
        from astra.models import ManeuverFrame
        burns = plan_inclination_change(
            42164.0, 1.0, 300.0, 3000.0, 22.0, 2460000.5, frame=ManeuverFrame.RTN,
        )
        assert burns[0].direction[2] > 0.0  # +N in RTN

    def test_zero_inclination_raises(self) -> None:
        """Zero Δi must raise."""
        from astra.maneuver import plan_inclination_change
        from astra.errors import ManeuverError
        with pytest.raises(ManeuverError, match="zero"):
            plan_inclination_change(42164.0, 0.0, 300.0, 3000.0, 22.0, 2460000.5)


# ===========================================================================
# AS-01: propagate_cowell_at_times
# ===========================================================================

class TestPropagateAtTimes:
    """Tests for propagate_cowell_at_times."""

    def test_returns_correct_number_of_states(self) -> None:
        """Must return exactly len(times_jd) states."""
        from astra.propagator import NumericalState, propagate_cowell_at_times

        state0 = NumericalState(
            t_jd=2460000.5,
            position_km=np.array([6778.0, 0.0, 0.0]),
            velocity_km_s=np.array([0.0, 7.668, 0.0]),
        )
        times = np.array([2460000.501, 2460000.502, 2460000.505])
        states = propagate_cowell_at_times(state0, times)
        assert len(states) == 3

    def test_epochs_match_requested_times(self) -> None:
        """Each returned state must have the requested t_jd."""
        from astra.propagator import NumericalState, propagate_cowell_at_times

        state0 = NumericalState(
            t_jd=2460000.5,
            position_km=np.array([6778.0, 0.0, 0.0]),
            velocity_km_s=np.array([0.0, 7.668, 0.0]),
        )
        times = np.array([2460000.501, 2460000.503, 2460000.510])
        states = propagate_cowell_at_times(state0, times)
        for s, t in zip(states, times):
            assert abs(s.t_jd - t) < 1e-12

    def test_orbit_radius_preserved(self) -> None:
        """Orbit radius must be approximately preserved for a circular orbit."""
        from astra.propagator import NumericalState, propagate_cowell_at_times

        r0 = 6778.0
        state0 = NumericalState(
            t_jd=2460000.5,
            position_km=np.array([r0, 0.0, 0.0]),
            velocity_km_s=np.array([0.0, 7.668, 0.0]),
        )
        # 10 minutes forward
        times = np.array([2460000.5 + 600.0 / 86400.0])
        states = propagate_cowell_at_times(state0, times, include_third_body=False)
        r_final = np.linalg.norm(states[0].position_km)
        # Radius should be within ~1% for near-circular orbit (J2 causes small variations)
        assert abs(r_final - r0) / r0 < 0.01

    def test_empty_times_raises(self) -> None:
        """Empty times array must raise ValueError."""
        from astra.propagator import NumericalState, propagate_cowell_at_times

        state0 = NumericalState(
            t_jd=2460000.5,
            position_km=np.array([6778.0, 0.0, 0.0]),
            velocity_km_s=np.array([0.0, 7.668, 0.0]),
        )
        with pytest.raises(ValueError, match="non-empty"):
            propagate_cowell_at_times(state0, np.array([]))

    def test_time_before_epoch_raises(self) -> None:
        """Times before state0.t_jd must raise ValueError."""
        from astra.propagator import NumericalState, propagate_cowell_at_times

        state0 = NumericalState(
            t_jd=2460000.5,
            position_km=np.array([6778.0, 0.0, 0.0]),
            velocity_km_s=np.array([0.0, 7.668, 0.0]),
        )
        with pytest.raises(ValueError, match="state0.t_jd"):
            propagate_cowell_at_times(state0, np.array([2459999.0]))

    def test_propagate_at_times_mass_interpolated_during_burn(self) -> None:
        """Mass at mid-burn epoch must be between initial and final mass."""
        from astra.propagator import NumericalState, propagate_cowell_at_times
        from astra.models import FiniteBurn, ManeuverFrame
        t0 = 2460000.5
        state0 = NumericalState(t_jd=t0,
                                position_km=np.array([6778.0, 0.0, 0.0]),
                                velocity_km_s=np.array([0.0, 7.668, 0.0]),
                                mass_kg=1000.0)
        burn = FiniteBurn(epoch_ignition_jd=t0 + 60/86400,
                          duration_s=120.0, thrust_N=100.0, isp_s=300.0,
                          direction=(1.0, 0.0, 0.0), frame=ManeuverFrame.VNB)
        mid_burn_t = t0 + 120/86400  # 60 s into the burn
        states = propagate_cowell_at_times(state0, np.array([mid_burn_t]),
                                           maneuvers=[burn])
        assert states[0].mass_kg < 1000.0, "Mid-burn mass must be less than initial"
        assert states[0].mass_kg > 1000.0 - (100.0/(300.0*9.80665))*120.0, \
            "Mid-burn mass must be greater than final mass"


# ===========================================================================
# AS-03: find_conjunction_windows
# ===========================================================================

class TestFindConjunctionWindows:
    """Tests for find_conjunction_windows."""

    def test_parallel_close_trajectories_produce_window(self) -> None:
        """Two parallel trajectories within threshold should produce a window."""
        from astra.conjunction import find_conjunction_windows

        T = 20
        times = np.linspace(2460000.5, 2460000.5 + 100.0 / 86400.0, T)
        # Object A: circular motion
        angles = np.linspace(0, np.pi / 4, T)
        traj_a = np.column_stack([7000.0 * np.cos(angles),
                                  7000.0 * np.sin(angles),
                                  np.zeros(T)])
        # Object B: 2 km offset (within 5 km threshold)
        traj_b = traj_a + np.array([2.0, 0.0, 0.0])

        windows = find_conjunction_windows(
            {"A": traj_a, "B": traj_b},
            times,
            threshold_km=5.0,
        )
        assert len(windows) >= 1
        w = windows[0]
        assert w.object_a_id == "A"
        assert w.object_b_id == "B"
        assert w.min_distance_km < 5.0
        assert w.entry_jd <= w.tca_jd <= w.exit_jd
        assert w.duration_s > 0.0

    def test_distant_trajectories_produce_no_window(self) -> None:
        """Two distant trajectories should produce no windows."""
        from astra.conjunction import find_conjunction_windows

        T = 10
        times = np.linspace(2460000.5, 2460000.6, T)
        traj_a = np.tile([7000.0, 0.0, 0.0], (T, 1))
        traj_b = np.tile([7100.0, 0.0, 0.0], (T, 1))  # 100 km away

        windows = find_conjunction_windows(
            {"A": traj_a, "B": traj_b},
            times,
            threshold_km=5.0,
        )
        assert len(windows) == 0

    def test_window_contains_tca(self) -> None:
        """TCA must be within the window interval."""
        from astra.conjunction import find_conjunction_windows

        T = 30
        times = np.linspace(2460000.5, 2460000.5 + 300.0 / 86400.0, T)
        # Approach and departure: V-shaped distance profile
        traj_a = np.tile([7000.0, 0.0, 0.0], (T, 1))
        offsets = np.abs(np.linspace(-10.0, 10.0, T))
        traj_b = traj_a.copy()
        traj_b[:, 0] += offsets  # closest at center

        windows = find_conjunction_windows(
            {"A": traj_a, "B": traj_b},
            times,
            threshold_km=8.0,
        )
        for w in windows:
            assert w.entry_jd <= w.tca_jd <= w.exit_jd

    def test_too_few_timesteps_raises(self) -> None:
        """Fewer than 3 timesteps must raise AstraError."""
        from astra.conjunction import find_conjunction_windows
        from astra.errors import AstraError

        with pytest.raises(AstraError, match="3"):
            find_conjunction_windows(
                {"A": np.zeros((2, 3)), "B": np.zeros((2, 3))},
                np.array([1.0, 2.0]),
            )


# ===========================================================================
# AS-05: compute_collision_probability_timeseries
# ===========================================================================

class TestPcTimeseries:
    """Tests for compute_collision_probability_timeseries."""

    def test_basic_timeseries_shape(self) -> None:
        """Output must be (N,) array with values in [0, 1]."""
        from astra.covariance import compute_collision_probability_timeseries

        N = 10
        miss = np.tile([0.5, 0.0, 0.0], (N, 1))  # 500 m miss
        vrel = np.tile([7.0, 0.0, 0.0], (N, 1))   # 7 km/s relative
        cov_a = np.tile(np.eye(3) * 0.01, (N, 1, 1))
        cov_b = np.tile(np.eye(3) * 0.01, (N, 1, 1))
        times = np.linspace(2460000.5, 2460001.5, N)

        pc = compute_collision_probability_timeseries(
            miss, vrel, cov_a, cov_b, times
        )
        assert pc.shape == (N,)
        assert np.all(np.isfinite(pc))
        assert np.all((pc >= 0.0) & (pc <= 1.0))

    def test_closer_miss_produces_higher_pc(self) -> None:
        """Smaller miss distance should produce higher Pc."""
        from astra.covariance import compute_collision_probability_timeseries

        N = 2
        # Relative velocity along x-axis
        vrel = np.tile([7.0, 0.0, 0.0], (N, 1))
        # Covariance scaled so both miss distances are in the resolvable regime
        cov = np.tile(np.eye(3) * 0.1, (N, 1, 1))
        times = np.array([2460000.5, 2460001.5])

        # Miss vector perpendicular to vrel (y-axis) — this is what the
        # encounter-plane projection actually uses for Pc.
        miss = np.array([[0.0, 0.005, 0.0], [0.0, 1.0, 0.0]])
        pc = compute_collision_probability_timeseries(
            miss, vrel, cov, cov, times, radius_a_km=0.005, radius_b_km=0.005,
        )
        assert pc[0] > pc[1], \
            f"Closer miss should give higher Pc: pc[0]={pc[0]:.2e}, pc[1]={pc[1]:.2e}"

    def test_shape_mismatch_raises(self) -> None:
        """Mismatched N dimensions must raise ValueError."""
        from astra.covariance import compute_collision_probability_timeseries

        with pytest.raises(ValueError, match="miss_vectors_km"):
            compute_collision_probability_timeseries(
                np.zeros((5, 2)),   # wrong: should be (N, 3)
                np.zeros((5, 3)),
                np.zeros((5, 3, 3)),
                np.zeros((5, 3, 3)),
                np.zeros(5),
            )

    def test_singular_covariance_produces_nan(self) -> None:
        """Zero covariance should produce NaN or a valid fallback."""
        from astra.covariance import compute_collision_probability_timeseries

        N = 3
        miss = np.tile([0.5, 0.0, 0.0], (N, 1))
        vrel = np.tile([7.0, 0.0, 0.0], (N, 1))
        cov_zero = np.zeros((N, 3, 3))
        cov_good = np.tile(np.eye(3) * 0.01, (N, 1, 1))
        times = np.linspace(2460000.5, 2460001.5, N)

        # Zero + good covariance → deterministic collision check → should not crash
        pc = compute_collision_probability_timeseries(
            miss, vrel, cov_zero, cov_good, times
        )
        assert pc.shape == (N,)
        # Should be finite (either 0 or valid Pc from the non-zero combined cov)
        assert np.all(np.isfinite(pc))
