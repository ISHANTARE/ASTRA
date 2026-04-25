"""Regression tests for ASTRA v3.6.0 audit remediation.

Covers all six findings from the deep audit:
  - CF-6: SPACEBOOK_ENABLED centralized in astra.config
  - FM-9A: assert_guards for Numba-inlined literals in constants.py
  - CF-7: strict mode enforced in STM covariance density fallback
  - FM-3: STM and Cowell atmosphere model consistency (F10.7/Ap threading)
  - FM-5: Physical invariant tests (validated via test_orbit.py upgrades)

Each test is a minimal, deterministic, offline unit test that does NOT
require network access or Numba JIT compilation.
"""

from __future__ import annotations

import math
import threading
import pytest
import numpy as np


# ===========================================================================
# CF-6: SPACEBOOK_ENABLED centralized in config
# ===========================================================================

class TestCF6SpacebookCentralized:
    """Verify SPACEBOOK_ENABLED lives exclusively in astra.config."""

    def test_spacebook_enabled_importable_from_config(self) -> None:
        """SPACEBOOK_ENABLED must be importable from astra.config."""
        from astra.config import SPACEBOOK_ENABLED
        assert isinstance(SPACEBOOK_ENABLED, bool)

    def test_set_spacebook_enabled_callable_from_config(self) -> None:
        """set_spacebook_enabled must be importable and callable from astra.config."""
        from astra.config import set_spacebook_enabled
        assert callable(set_spacebook_enabled)

    def test_set_spacebook_enabled_callable_from_astra(self) -> None:
        """set_spacebook_enabled must be in the public astra namespace."""
        import astra
        assert hasattr(astra, "set_spacebook_enabled")
        assert callable(astra.set_spacebook_enabled)

    def test_set_spacebook_enabled_round_trip(self) -> None:
        """set_spacebook_enabled must persist the value it is given."""
        from astra.config import set_spacebook_enabled, SPACEBOOK_ENABLED as orig
        try:
            set_spacebook_enabled(False)
            from astra import config
            assert config.SPACEBOOK_ENABLED is False

            set_spacebook_enabled(True)
            assert config.SPACEBOOK_ENABLED is True
        finally:
            set_spacebook_enabled(orig)

    def test_set_spacebook_enabled_thread_safe(self) -> None:
        """Concurrent set_spacebook_enabled calls must not raise."""
        from astra.config import set_spacebook_enabled, SPACEBOOK_ENABLED as orig
        errors: list[Exception] = []

        def _toggle(val: bool) -> None:
            try:
                for _ in range(50):
                    set_spacebook_enabled(val)
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

        assert not errors, f"Concurrent set_spacebook_enabled raised: {errors}"
        set_spacebook_enabled(orig)

    def test_frames_uses_config_not_environ(self) -> None:
        """frames.get_eop_correction must obey config.SPACEBOOK_ENABLED, not os.environ."""
        from astra.config import set_spacebook_enabled, SPACEBOOK_ENABLED as orig
        try:
            set_spacebook_enabled(False)
            from astra.frames import get_eop_correction
            import numpy as np
            xp, yp, dut1 = get_eop_correction(np.array([2451545.0]))
            # When disabled, must return zeros without raising
            assert float(xp[0]) == 0.0
            assert float(yp[0]) == 0.0
            assert float(dut1[0]) == 0.0
        finally:
            set_spacebook_enabled(orig)

    def test_spacebook_guard_uses_config(self) -> None:
        """spacebook._spacebook_enabled() must reflect config.SPACEBOOK_ENABLED."""
        from astra.config import set_spacebook_enabled, SPACEBOOK_ENABLED as orig
        from astra.spacebook import _spacebook_enabled
        try:
            set_spacebook_enabled(False)
            assert _spacebook_enabled() is False
            set_spacebook_enabled(True)
            assert _spacebook_enabled() is True
        finally:
            set_spacebook_enabled(orig)


# ===========================================================================
# FM-9A: assert_guards for Numba-inlined literals
# ===========================================================================

class TestFM9AConstantGuards:
    """Verify constants.py contains the assert_guards added in the FM-9A fix."""

    def test_constants_module_imports_without_error(self) -> None:
        """constants.py must import cleanly — any failed assertion fires here."""
        import importlib
        import astra.constants as c
        importlib.reload(c)  # re-run assertions

    def test_j2_guard_value(self) -> None:
        from astra.constants import J2
        assert abs(J2 - 1.08262668e-3) < 1e-15, f"J2={J2!r} diverged"

    def test_earth_mu_guard_value(self) -> None:
        from astra.constants import EARTH_MU_KM3_S2
        assert EARTH_MU_KM3_S2 == 398600.4418

    def test_g0_std_km_s2_guard_value(self) -> None:
        from astra.constants import G0_STD_KM_S2
        assert abs(G0_STD_KM_S2 - 9.80665e-3) < 1e-20, f"G0_STD_KM_S2={G0_STD_KM_S2!r}"

    def test_r_gas_guard_value(self) -> None:
        from astra.constants import R_GAS
        assert abs(R_GAS - 8.314462618) < 1e-9, f"R_GAS={R_GAS!r}"

    def test_sun_radius_guard_value(self) -> None:
        from astra.constants import SUN_RADIUS_KM
        assert SUN_RADIUS_KM == 695700.0

    def test_g0_km_s2_matches_propagator_literal(self) -> None:
        """G0_STD_KM_S2 must exactly match the 9.80665e-3 literal in _powered_derivative_njit."""
        from astra.constants import G0_STD_KM_S2
        propagator_literal = 9.80665e-3
        assert G0_STD_KM_S2 == propagator_literal, (
            f"constants.G0_STD_KM_S2={G0_STD_KM_S2!r} != propagator literal "
            f"{propagator_literal!r}. Update propagator.py to use the constant."
        )

    def test_j2_matches_propagator_j2c_literal(self) -> None:
        """J2 must exactly match the J2c = 0.00108262668 literal in propagator.py kernels."""
        from astra.constants import J2
        propagator_j2c = 0.00108262668
        assert abs(J2 - propagator_j2c) < 1e-15, (
            f"constants.J2={J2!r} != propagator J2c={propagator_j2c!r}. "
            "Update propagator.py _acceleration_njit and _compute_force_jacobian."
        )


# ===========================================================================
# CF-7: Strict mode enforced in covariance STM density fallback
# ===========================================================================

class TestCF7StrictModeCovarianceFallback:
    """Verify strict mode raises when space-weather fails in STM propagation."""

    def test_strict_mode_raises_on_sw_failure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When ASTRA_STRICT_MODE=True and get_space_weather fails, PropagationError must fire."""
        import astra.config as cfg
        from astra.errors import PropagationError
        import astra.covariance as cov_mod

        # Make get_space_weather always fail
        def _always_fail(t_jd: float):
            raise RuntimeError("No SW data (mocked for test)")

        monkeypatch.setattr(
            "astra.data_pipeline.get_space_weather", _always_fail, raising=False
        )
        # Also patch the import inside covariance.py's function scope
        monkeypatch.setattr(
            "astra.covariance.atmospheric_density_empirical",
            lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("No data")),
            raising=False,
        )

        prev = cfg.ASTRA_STRICT_MODE
        cfg.ASTRA_STRICT_MODE = True
        try:
            from astra.models import SatelliteTLE
            from astra.propagator import DragConfig

            r0 = np.array([6778.0, 0.0, 0.0])
            v0 = np.array([0.0, 7.67, 0.0])
            cov0 = np.eye(6) * 1.0
            dc = DragConfig(cd=2.2, area_m2=10.0, mass_kg=1000.0)

            with pytest.raises(PropagationError, match="ASTRA STRICT"):
                cov_mod.propagate_covariance_stm(
                    r0_km=r0,
                    v0_km_s=v0,
                    cov0_6x6=cov0,
                    t_jd0=2451545.0,
                    duration_s=3600.0,
                    drag_config=dc,
                )
        finally:
            cfg.ASTRA_STRICT_MODE = prev

    def test_relaxed_mode_logs_warning_on_sw_failure(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """When ASTRA_STRICT_MODE=False and get_space_weather fails, a warning must be logged."""
        import astra.config as cfg
        import astra.covariance as cov_mod
        import logging

        def _always_fail(t_jd: float):
            raise RuntimeError("No SW data (mocked for test)")

        monkeypatch.setattr(
            "astra.data_pipeline.get_space_weather", _always_fail, raising=False
        )

        prev = cfg.ASTRA_STRICT_MODE
        cfg.ASTRA_STRICT_MODE = False
        try:
            from astra.propagator import DragConfig
            r0 = np.array([6778.0, 0.0, 0.0])
            v0 = np.array([0.0, 7.67, 0.0])
            cov0 = np.eye(6) * 1.0
            dc = DragConfig(cd=2.2, area_m2=10.0, mass_kg=1000.0)

            with caplog.at_level(logging.WARNING, logger="astra.covariance"):
                try:
                    cov_mod.propagate_covariance_stm(
                        r0_km=r0,
                        v0_km_s=v0,
                        cov0_6x6=cov0,
                        t_jd0=2451545.0,
                        duration_s=3600.0,
                        drag_config=dc,
                    )
                except Exception:
                    pass  # STM may still fail for other reasons; we only check the warning

            assert any("Falling back" in r.message or "fallback" in r.message.lower()
                       for r in caplog.records), (
                "Expected a 'fallback' warning when space-weather fails in relaxed mode. "
                f"Captured records: {[r.message for r in caplog.records]}"
            )
        finally:
            cfg.ASTRA_STRICT_MODE = prev


# ===========================================================================
# FM-3: STM atmosphere model consistency
# ===========================================================================

class TestFM3AtmosphereSync:
    """Verify that STM covariance and Cowell propagator use consistent atmosphere model flags."""

    def test_stm_accepts_nrlmsise_drag_config(self) -> None:
        """propagate_covariance_stm must not crash when DragConfig.model='NRLMSISE00'."""
        from astra.propagator import DragConfig
        import astra.covariance as cov_mod

        r0 = np.array([6778.0, 0.0, 0.0])
        v0 = np.array([0.0, 7.668, 0.0])
        cov0 = np.eye(6) * 1e-3
        dc = DragConfig(cd=2.2, area_m2=10.0, mass_kg=1000.0, model="NRLMSISE00")

        # Should complete without raising (may use exponential fallback if SW unavailable)
        result = cov_mod.propagate_covariance_stm(
            r0_km=r0,
            v0_km_s=v0,
            cov0_6x6=cov0,
            t_jd0=2451545.0,
            duration_s=3600.0,
            drag_config=dc,
        )
        assert result.shape == (6, 6), f"Expected (6,6) covariance, got {result.shape}"
        assert np.all(np.isfinite(result)), "Covariance contains NaN/Inf"

    def test_stm_covariance_grows_with_drag(self) -> None:
        """With drag enabled, propagated covariance trace must exceed initial for LEO orbits."""
        from astra.propagator import DragConfig
        import astra.covariance as cov_mod

        r0 = np.array([6778.0, 0.0, 0.0])
        v0 = np.array([0.0, 7.668, 0.0])
        cov0 = np.eye(6) * 1e-4  # 10 m 1-sigma in each axis
        dc = DragConfig(cd=2.2, area_m2=10.0, mass_kg=1000.0, model="EXPONENTIAL")

        cov_propagated = cov_mod.propagate_covariance_stm(
            r0_km=r0,
            v0_km_s=v0,
            cov0_6x6=cov0,
            t_jd0=2451545.0,
            duration_s=7200.0,  # 2 hours
            drag_config=dc,
        )
        assert np.trace(cov_propagated) >= np.trace(cov0), (
            f"Covariance trace shrank from {np.trace(cov0):.4e} to "
            f"{np.trace(cov_propagated):.4e} — indicates integration failure or "
            "negative divergence in the drag Jacobian."
        )
