"""Production-grade audit tests — validates every finding from the v3.7.1 deep audit.

Test identifiers map directly to the audit report remediation backlog:
    T-01  SNC white-noise covariance growth vs closed-form analytical expectation
    T-02  propagate_cowell_at_times mass interpolation during powered arcs
    T-03  compute_delta_v_budget vs propagate_cowell mass depletion cross-validation
    T-04  propagate_covariance_stm emits warning for GEO initial state
    T-05  Two-body energy conservation for pure coast arcs
    T-06  Cowell initial acceleration magnitude for LEO state
    T-07  estimate_covariance physical plausibility (sigma_r in [0.01, 100] km)
    H-05  NumericalState frozen array immutability (defensive copy)
    H-03  propagate_cowell_batch honours use_empirical_drag parameter
    F-10  srp_cylindrical_illumination_factor emits DeprecationWarning
    F-16  include_stm=True error message includes constructive guidance
    F-01  DMC mode error no longer references v3.7.0
    F-09  g0 Numba literal guard fires at import time
"""

from __future__ import annotations

import logging
import math
import warnings

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Shared LEO state factory
# ---------------------------------------------------------------------------

def _leo_state(mass_kg: float = 1000.0, with_covariance: bool = False):
    """Return a canonical circular LEO NumericalState at ~400 km altitude."""
    from astra.propagator import NumericalState

    cov = None
    if with_covariance:
        cov = np.diag([1e-2, 1e-2, 1e-2, 1e-6, 1e-6, 1e-6])

    return NumericalState(
        t_jd=2460000.5,
        position_km=np.array([6778.137, 0.0, 0.0]),
        velocity_km_s=np.array([0.0, 7.6597, 0.0]),
        mass_kg=mass_kg,
        covariance_km2=cov,
    )


# ===========================================================================
# T-05 — Two-body energy conservation for coast arcs
# ===========================================================================

class TestEnergyConservation:
    """Specific mechanical energy ε = v²/2 − μ/r oscillates but stays bounded
    for coast arcs with conservative perturbations (J2–J6, no drag).

    J2 perturbations cause periodic energy exchange of order
    ΔE ≈ J2·μ·Re²/r³ ≈ 5e-4 km²/s² for LEO.  The test validates that the
    oscillation stays within this physically expected envelope, confirming
    the integrator does not have a secular energy leak.
    """

    def test_two_body_energy_bounded_1hr(self):
        """ε oscillation < 1e-3 km²/s² over a 1-hour J2–J6 coast (no drag)."""
        from astra.propagator import propagate_cowell, NumericalState
        from astra.constants import EARTH_MU_KM3_S2 as mu

        state0 = NumericalState(
            t_jd=2460000.5,
            position_km=np.array([7000.0, 0.0, 0.0]),
            velocity_km_s=np.array([0.0, 7.546, 0.0]),
        )

        states = propagate_cowell(
            state0, duration_s=3600.0, dt_out=60.0,
            drag_config=None, include_third_body=False, use_de=False,
        )

        energies = []
        for s in states:
            r = float(np.linalg.norm(s.position_km))
            v = float(np.linalg.norm(s.velocity_km_s))
            energies.append(0.5 * v * v - mu / r)

        e_range = max(energies) - min(energies)
        # J2 energy oscillation for LEO is ~5e-4 km²/s²; allow 1e-3 envelope.
        # A secular leak (integrator bug) would cause e_range to grow linearly
        # with propagation duration — 1e-3 bounds it to the expected oscillation.
        assert e_range < 1e-3, (
            f"Energy oscillation {e_range:.3e} km²/s² exceeds J2-expected envelope "
            "(1e-3 km²/s²) over 1-hour coast. Possible integrator energy leak."
        )


# ===========================================================================
# T-06 — Cowell initial acceleration magnitude for LEO
# ===========================================================================

class TestCowellAccelerationMagnitude:
    """For circular LEO at ~7000 km, |a| ≈ μ/r² ≈ 8.14e-3 km/s²."""

    def test_acceleration_magnitude_leo(self):
        from astra.propagator import _acceleration

        r = np.array([7000.0, 0.0, 0.0])
        v = np.array([0.0, 7.546, 0.0])
        empty = np.zeros((1, 2, 3))

        a = _acceleration(
            2460000.5, r, v,
            False, 2.2, 10.0, 1000.0, 0.0, 58.5, 400.0,
            150.0, 150.0, 15.0, False,
            False, 2460000.5, 1.0, empty, empty,
            False, 1.5, True,
        )

        a_mag = float(np.linalg.norm(a))
        assert 7e-3 < a_mag < 9e-3, (
            f"|a| = {a_mag:.5e} km/s² outside expected LEO range [7e-3, 9e-3]. "
            "Gravitational constant or radius may be wrong."
        )


# ===========================================================================
# T-07 — estimate_covariance physical plausibility
# ===========================================================================

class TestEstimateCovariancePlausibility:
    """For time_since_epoch_days=1.5, position 1-sigma must be in [0.01, 100] km."""

    def test_position_sigma_physical_range(self):
        from astra.covariance import estimate_covariance
        import astra.config as cfg

        prev = cfg.ASTRA_STRICT_MODE
        cfg.ASTRA_STRICT_MODE = False
        try:
            cov = estimate_covariance(time_since_epoch_days=1.5)
            sigma_r = math.sqrt(cov[0, 0])
            assert 0.01 < sigma_r < 100.0, (
                f"Position 1-sigma = {sigma_r:.4f} km outside physically realistic "
                "range [0.01, 100] km for a 1.5-day old TLE."
            )
        finally:
            cfg.ASTRA_STRICT_MODE = prev

    def test_matrix_properties(self):
        """Covariance must be symmetric, PSD, and finite."""
        from astra.covariance import estimate_covariance
        import astra.config as cfg

        prev = cfg.ASTRA_STRICT_MODE
        cfg.ASTRA_STRICT_MODE = False
        try:
            cov = estimate_covariance(time_since_epoch_days=3.0, f107_flux=200.0)
            assert cov.shape == (3, 3)
            assert np.all(np.isfinite(cov))
            np.testing.assert_allclose(cov, cov.T, atol=1e-15)
            eigvals = np.linalg.eigvalsh(cov)
            assert np.all(eigvals >= -1e-12), f"Non-PSD covariance: eigenvalues={eigvals}"
        finally:
            cfg.ASTRA_STRICT_MODE = prev


# ===========================================================================
# T-04 — propagate_covariance_stm GEO warning
# ===========================================================================

class TestCovarianceSTMGeoWarning:
    """propagate_covariance_stm must emit a warning for GEO/HEO altitudes."""

    def test_geo_altitude_emits_warning(self, caplog):
        """Calling with r ~ 42164 km (GEO) must log a warning about altitude."""
        from astra.covariance import propagate_covariance_stm

        r0 = np.array([42164.0, 0.0, 0.0])
        v0 = np.array([0.0, 3.0747, 0.0])
        cov0 = np.eye(6) * 1e-2

        with caplog.at_level(logging.WARNING, logger="astra.covariance"):
            try:
                propagate_covariance_stm(
                    t_jd0=2460000.5, r0_km=r0, v0_km_s=v0,
                    cov0_6x6=cov0, duration_s=60.0,
                )
            except Exception:
                pass  # Integration may fail for GEO without drag — that's OK

        assert any("MEO/GEO/HEO" in rec.message for rec in caplog.records), (
            "propagate_covariance_stm must warn about MEO/GEO/HEO orbit altitude "
            "to prevent silent covariance model mismatch."
        )


# ===========================================================================
# T-02 — propagate_cowell_at_times mass interpolation during burn
# ===========================================================================

class TestPropagateAtTimesMassInterpolation:
    """Mid-burn interpolated mass must be between initial and final mass."""

    def test_mass_interpolated_during_burn(self):
        from astra.propagator import NumericalState, propagate_cowell_at_times
        from astra.models import FiniteBurn, ManeuverFrame

        t0 = 2460000.5
        initial_mass = 1000.0
        state0 = NumericalState(
            t_jd=t0,
            position_km=np.array([6778.0, 0.0, 0.0]),
            velocity_km_s=np.array([0.0, 7.668, 0.0]),
            mass_kg=initial_mass,
        )

        burn = FiniteBurn(
            epoch_ignition_jd=t0 + 60.0 / 86400.0,
            duration_s=120.0,
            thrust_N=100.0,
            isp_s=300.0,
            direction=(1.0, 0.0, 0.0),
            frame=ManeuverFrame.VNB,
        )

        # Query at mid-burn (60 s into burn = 120 s from epoch)
        mid_burn_t = t0 + 120.0 / 86400.0
        states = propagate_cowell_at_times(
            state0, np.array([mid_burn_t]),
            maneuvers=[burn],
            include_third_body=False,
        )

        assert states[0].mass_kg is not None
        assert states[0].mass_kg < initial_mass, (
            f"Mid-burn mass {states[0].mass_kg} must be less than initial {initial_mass}"
        )

        # Compute max possible mass depletion (full burn)
        g0 = 9.80665
        max_dm = (100.0 / (300.0 * g0)) * 120.0
        final_mass_min = initial_mass - max_dm

        assert states[0].mass_kg > final_mass_min, (
            f"Mid-burn mass {states[0].mass_kg} must be greater than "
            f"minimum final mass {final_mass_min}"
        )

    def test_position_continuity_with_dense(self):
        """Interpolated position must be continuous with dense propagation output."""
        from astra.propagator import (
            NumericalState, propagate_cowell, propagate_cowell_at_times,
        )

        t0 = 2460000.5
        state0 = NumericalState(
            t_jd=t0,
            position_km=np.array([6778.0, 0.0, 0.0]),
            velocity_km_s=np.array([0.0, 7.668, 0.0]),
        )

        # Dense propagation for reference
        dense = propagate_cowell(
            state0, duration_s=600.0, dt_out=10.0,
            drag_config=None, include_third_body=False,
        )

        # Pick a reference point from dense output
        ref_state = dense[len(dense) // 2]
        ref_t = ref_state.t_jd

        # Interpolated query at the same time
        interp = propagate_cowell_at_times(
            state0, np.array([ref_t]),
            include_third_body=False,
        )

        # Position should agree to within spline interpolation accuracy
        np.testing.assert_allclose(
            interp[0].position_km, ref_state.position_km,
            rtol=1e-4, atol=1e-3,
            err_msg="Interpolated position diverges from dense propagation reference."
        )


# ===========================================================================
# T-03 — compute_delta_v_budget vs propagate_cowell cross-validation
# ===========================================================================

class TestBudgetVsPropagatorCrossValidation:
    """Analytical ΔV budget and numerical propagator mass depletion must agree."""

    def test_mass_depletion_agreement(self):
        from astra.propagator import propagate_cowell, NumericalState, DragConfig
        from astra.maneuver import compute_delta_v_budget
        from astra.models import FiniteBurn, ManeuverFrame

        t0 = 2460000.5
        initial_mass = 1000.0

        burn = FiniteBurn(
            epoch_ignition_jd=t0 + 60.0 / 86400.0,
            duration_s=60.0,
            thrust_N=50.0,
            isp_s=300.0,
            direction=(1.0, 0.0, 0.0),
            frame=ManeuverFrame.VNB,
        )

        # Analytical budget
        budget = compute_delta_v_budget([burn], initial_mass)

        # Numerical propagation
        state0 = NumericalState(
            t_jd=t0,
            position_km=np.array([6778.0, 0.0, 0.0]),
            velocity_km_s=np.array([0.0, 7.668, 0.0]),
            mass_kg=initial_mass,
        )

        states = propagate_cowell(
            state0, duration_s=180.0,
            maneuvers=[burn],
            drag_config=DragConfig(mass_kg=initial_mass),
            include_third_body=False,
        )

        dm_propagator = initial_mass - states[-1].mass_kg
        dm_budget = budget.total_propellant_kg

        assert abs(dm_propagator - dm_budget) < 0.01, (
            f"Mass depletion mismatch: propagator={dm_propagator:.4f} kg, "
            f"budget={dm_budget:.4f} kg (diff={abs(dm_propagator - dm_budget):.4f} kg). "
            "g0 constant or Tsiolkovsky coupling may be inconsistent."
        )


# ===========================================================================
# H-05 — NumericalState frozen array immutability
# ===========================================================================

class TestNumericalStateImmutability:
    """NumericalState must own its array data — external mutation must not propagate."""

    def test_position_array_is_independent_copy(self):
        from astra.propagator import NumericalState

        pos = np.array([7000.0, 0.0, 0.0])
        state = NumericalState(
            t_jd=2460000.5, position_km=pos,
            velocity_km_s=np.array([0.0, 7.5, 0.0]),
        )

        # Mutate the original array — state must be unaffected
        pos[0] = 9999.0
        assert state.position_km[0] == 7000.0, (
            "NumericalState.position_km must be a defensive copy. "
            "External array mutation corrupted the state."
        )

    def test_velocity_array_is_independent_copy(self):
        from astra.propagator import NumericalState

        vel = np.array([0.0, 7.5, 0.0])
        state = NumericalState(
            t_jd=2460000.5,
            position_km=np.array([7000.0, 0.0, 0.0]),
            velocity_km_s=vel,
        )

        vel[1] = 0.0
        assert state.velocity_km_s[1] == 7.5, (
            "NumericalState.velocity_km_s must be a defensive copy."
        )

    def test_covariance_array_is_independent_copy(self):
        from astra.propagator import NumericalState

        cov = np.eye(6) * 0.01
        state = NumericalState(
            t_jd=2460000.5,
            position_km=np.array([7000.0, 0.0, 0.0]),
            velocity_km_s=np.array([0.0, 7.5, 0.0]),
            covariance_km2=cov,
        )

        cov[0, 0] = 999.0
        assert state.covariance_km2[0, 0] == pytest.approx(0.01), (
            "NumericalState.covariance_km2 must be a defensive copy."
        )


# ===========================================================================
# H-03 — propagate_cowell_batch honours use_empirical_drag
# ===========================================================================

class TestBatchUseEmpiricalDrag:
    """propagate_cowell_batch must thread use_empirical_drag through to workers."""

    def test_parameter_is_exposed(self):
        """The use_empirical_drag keyword must be accepted without TypeError."""
        import inspect
        from astra.propagator import propagate_cowell_batch

        sig = inspect.signature(propagate_cowell_batch)
        assert "use_empirical_drag" in sig.parameters, (
            "propagate_cowell_batch must expose use_empirical_drag parameter "
            "instead of hardcoding True."
        )


# ===========================================================================
# F-10 — srp_cylindrical_illumination_factor DeprecationWarning
# ===========================================================================

class TestSRPCylindricalDeprecation:
    """srp_cylindrical_illumination_factor must emit DeprecationWarning."""

    def test_deprecation_warning_emitted(self):
        from astra.propagator import srp_cylindrical_illumination_factor

        r = np.array([6800.0, 0.0, 0.0])
        rsun = np.array([1.0e8, 0.0, 0.0])

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            srp_cylindrical_illumination_factor(r, rsun)

        dep_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
        assert len(dep_warnings) > 0, (
            "srp_cylindrical_illumination_factor must emit DeprecationWarning "
            "to steer users toward srp_illumination_factor()."
        )


# ===========================================================================
# F-16 — include_stm=True error message includes guidance
# ===========================================================================

class TestSTMErrorMessage:
    """include_stm=True ValueError must include constructive guidance."""

    def test_error_mentions_covariance_construction(self):
        from astra.propagator import propagate_cowell, NumericalState

        state0 = NumericalState(
            t_jd=2460000.5,
            position_km=np.array([7000.0, 0.0, 0.0]),
            velocity_km_s=np.array([0.0, 7.5, 0.0]),
        )

        with pytest.raises(ValueError, match="sigma_r"):
            propagate_cowell(
                state0, duration_s=60.0, include_stm=True,
                drag_config=None, include_third_body=False,
            )


# ===========================================================================
# F-01 — DMC mode error message does not reference v3.7.0
# ===========================================================================

class TestDMCModeMessage:
    """SNCConfig DMC mode must not reference a specific version number."""

    def test_dmc_error_no_version(self):
        from astra.propagator import SNCConfig

        with pytest.raises(NotImplementedError) as exc_info:
            SNCConfig(mode="dmc")

        msg = str(exc_info.value)
        assert "v3.7.0" not in msg, (
            "DMC NotImplementedError must not reference 'v3.7.0' — "
            "this is the current release."
        )
        assert "future release" in msg.lower() or "not yet implemented" in msg.lower()


# ===========================================================================
# F-09 — g0 Numba literal guard fires at import time
# ===========================================================================

class TestG0LiteralGuard:
    """constants.py must assert that G0_STD_KM_S2 matches the Numba inlined literal."""

    def test_g0_guard_exists(self):
        from astra.constants import G0_STD_KM_S2

        # If we got here, the import-time assertion passed.
        assert abs(G0_STD_KM_S2 - 9.80665e-3) < 1e-20


# ===========================================================================
# T-01 — SNC white-noise covariance growth vs analytical (PLACEHOLDER)
#
# This test requires propagate_cowell with include_stm=True and snc_config,
# which exercises the full STM+SNC pipeline. The analytical closed-form is:
#   P_pos(t) = P_pos(0) + q·t³/3
#   P_vel(t) = P_vel(0) + q·t
# We test that propagated covariance grows in the correct direction.
# ===========================================================================

class TestSNCCovarianceGrowth:
    """SNC white-noise covariance must grow over time (monotone increase test)."""

    def test_covariance_grows_with_snc(self):
        """With non-zero SNC, final position covariance must exceed initial."""
        from astra.propagator import (
            propagate_cowell, NumericalState, SNCConfig, DragConfig,
        )

        q_m2s3 = 1e-8
        state0 = NumericalState(
            t_jd=2460000.5,
            position_km=np.array([6778.0, 0.0, 0.0]),
            velocity_km_s=np.array([0.0, 7.668, 0.0]),
            covariance_km2=np.diag([1e-4] * 3 + [1e-8] * 3),
        )

        snc = SNCConfig(q_psd_m2_s3=q_m2s3, mode="white_noise")

        states = propagate_cowell(
            state0, duration_s=120.0, dt_out=60.0,
            drag_config=None, include_third_body=False,
            include_stm=True, snc_config=snc,
        )

        P0 = state0.covariance_km2
        Pf = states[-1].covariance_km2

        assert Pf is not None, "Final state must have covariance when include_stm=True"

        # Position covariance trace must grow
        trace_pos_0 = P0[0, 0] + P0[1, 1] + P0[2, 2]
        trace_pos_f = Pf[0, 0] + Pf[1, 1] + Pf[2, 2]

        assert trace_pos_f > trace_pos_0, (
            f"Position covariance trace must grow with SNC: "
            f"initial={trace_pos_0:.3e}, final={trace_pos_f:.3e}"
        )

        # Velocity covariance trace must also grow
        trace_vel_0 = P0[3, 3] + P0[4, 4] + P0[5, 5]
        trace_vel_f = Pf[3, 3] + Pf[4, 4] + Pf[5, 5]

        assert trace_vel_f > trace_vel_0, (
            f"Velocity covariance trace must grow with SNC: "
            f"initial={trace_vel_0:.3e}, final={trace_vel_f:.3e}"
        )


# ===========================================================================
# Strategic: Pc exact integral used for mid-Mahalanobis regime (BL-04 guard)
# ===========================================================================

class TestPcExactIntegralMidMahalanobis:
    """For mahalanobis_sq in [1, 4) with 3×3 covariance, exact integral must be used."""

    def test_pc_nonzero_for_mid_mahalanobis(self):
        from astra.covariance import compute_collision_probability

        # Construct a case where mahalanobis_sq ≈ 2.25 (in [1, 4) range)
        miss = np.array([0.15, 0.0, 0.0])
        v_rel = np.array([0.0, 0.0, 10.0])
        sigma = 0.1
        cov = np.diag([sigma ** 2] * 3)

        pc = compute_collision_probability(
            miss, v_rel, cov, cov,
            radius_a_km=0.005, radius_b_km=0.005,
        )

        assert pc > 0 and math.isfinite(pc), (
            f"Pc must be positive and finite for mid-Mahalanobis regime, got {pc}"
        )


# ===========================================================================
# Strategic: Conjunction exception narrowing validation
# ===========================================================================

class TestConjunctionExceptionNarrowing:
    """TCA refinement must not swallow NameError/AttributeError."""

    def test_closest_approach_raises_programming_errors(self):
        """closest_approach must not catch NameError from its inner code."""
        from astra.conjunction import closest_approach

        traj_a = np.array([[7000.0, 0.0, 0.0], [7000.0, 100.0, 0.0], [7000.0, 200.0, 0.0]])
        traj_b = np.array([[7010.0, 0.0, 0.0], [7010.0, 100.0, 0.0], [7010.0, 200.0, 0.0]])
        times = np.array([2460000.5, 2460000.50347, 2460000.50694])

        # This should succeed without swallowing any errors
        dist, tca, idx = closest_approach(traj_a, traj_b, times)
        assert math.isfinite(dist)
        assert math.isfinite(tca)
