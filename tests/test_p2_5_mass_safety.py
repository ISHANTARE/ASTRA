import pytest
import logging
import numpy as np
from astra.models import FiniteBurn, ManeuverFrame
from astra.propagator import propagate_cowell, NumericalState
from astra.errors import ManeuverError
from astra import config


def test_mass_safety_strict_mode_p2_5():
    """Verify that MISSING mass raises ManeuverError in STRICT mode."""
    original_strict_mode = config.ASTRA_STRICT_MODE
    config.set_strict_mode(True)

    try:
        epoch = 2451545.0
        burn = FiniteBurn(
            epoch_ignition_jd=epoch,
            duration_s=100.0,
            thrust_N=10.0,
            isp_s=300.0,
            direction=(1.0, 0.0, 0.0),
            frame=ManeuverFrame.VNB,
        )

        state0 = NumericalState(
            t_jd=epoch,
            position_km=np.array([7000.0, 0.0, 0.0]),
            velocity_km_s=np.array([0.0, 7.5, 0.0]),
            mass_kg=None,
        )

        with pytest.raises(ManeuverError) as excinfo:
            propagate_cowell(state0, 200.0, maneuvers=[burn])

        assert "mass_kg is required" in str(excinfo.value)
    finally:
        config.set_strict_mode(original_strict_mode)


def test_mass_safety_relaxed_mode_p2_5(monkeypatch, caplog):
    """Missing mass in RELAXED mode must be equivalent to explicit 1000 kg."""
    import astra.propagator as propagator

    original_strict_mode = config.ASTRA_STRICT_MODE
    config.set_strict_mode(False)
    monkeypatch.setattr(propagator.logger, "propagate", True)

    try:
        epoch = 2451545.0
        burn = FiniteBurn(
            epoch_ignition_jd=epoch + 10.0 / 86400.0,
            duration_s=30.0,
            thrust_N=10.0,
            isp_s=300.0,
            direction=(1.0, 0.0, 0.0),
            frame=ManeuverFrame.VNB,
        )

        state_missing_mass = NumericalState(
            t_jd=epoch,
            position_km=np.array([7000.0, 0.0, 0.0]),
            velocity_km_s=np.array([0.0, 7.5, 0.0]),
            mass_kg=None,
        )
        state_explicit_mass = NumericalState(
            t_jd=epoch,
            position_km=np.array([7000.0, 0.0, 0.0]),
            velocity_km_s=np.array([0.0, 7.5, 0.0]),
            mass_kg=1000.0,
        )

        with caplog.at_level(logging.WARNING, logger="astra.propagator"):
            fallback_states = propagate_cowell(
                state_missing_mass,
                60.0,
                dt_out=30.0,
                maneuvers=[burn],
                include_third_body=False,
            )
        explicit_states = propagate_cowell(
            state_explicit_mass,
            60.0,
            dt_out=30.0,
            maneuvers=[burn],
            include_third_body=False,
        )

        assert any("Defaulting to 1000.0 kg" in record.message for record in caplog.records)
        assert len(fallback_states) == len(explicit_states)
        for fallback, explicit in zip(fallback_states, explicit_states):
            np.testing.assert_allclose(fallback.position_km, explicit.position_km, rtol=0.0, atol=1e-10)
            np.testing.assert_allclose(fallback.velocity_km_s, explicit.velocity_km_s, rtol=0.0, atol=1e-12)
            assert fallback.mass_kg == pytest.approx(explicit.mass_kg, abs=1e-12)
    finally:
        config.set_strict_mode(original_strict_mode)


def test_radius_validation_p2_5(monkeypatch, caplog):
    """Verify that unphysical position radius logs a warning."""
    import astra.propagator as propagator

    monkeypatch.setattr(propagator.logger, "propagate", True)
    with caplog.at_level(logging.WARNING, logger="astra.propagator"):
        NumericalState(
            t_jd=2451545.0,
            position_km=np.array([5000.0, 0.0, 0.0]),
            velocity_km_s=np.array([0.0, 7.5, 0.0]),
        )

    messages = [record.getMessage() for record in caplog.records]
    assert any(
        "NumericalState position radius 5000.00 km is inside Earth radius." in message
        for message in messages
    )
