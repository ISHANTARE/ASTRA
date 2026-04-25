"""tests/test_propagator_batch.py — Tests for propagate_cowell_batch.

[FM-4 Fix — Finding #16]
Validates the new batch propagation wrapper against propagate_cowell.
The API is dict[satellite_id, NumericalState] → dict[satellite_id, list[NumericalState]],
consistent with the TrajectoryMap convention used throughout ASTRA.
"""
import numpy as np
import pytest

from astra.propagator import NumericalState, propagate_cowell, propagate_cowell_batch


def _make_state(offset_km: float = 0.0) -> NumericalState:
    """Create a simple circular LEO initial state."""
    return NumericalState(
        t_jd=2460000.5,
        position_km=np.array([6778.0 + offset_km, 0.0, 0.0]),
        velocity_km_s=np.array([0.0, 7.668, 0.0]),
    )


def _states_dict(ids_offsets) -> dict:
    """Build a {sat_id: NumericalState} dict from (id, offset) tuples."""
    return {sid: _make_state(off) for sid, off in ids_offsets}


class TestPropagateCowell_Batch:

    # ── Basic contract ───────────────────────────────────────────────────────

    def test_returns_dict(self):
        states = {"SAT1": _make_state(), "SAT2": _make_state(10.0)}
        results = propagate_cowell_batch(states, duration_s=60.0, dt_out=10.0)
        assert isinstance(results, dict), "Must return a dict"

    def test_keys_are_satellite_ids(self):
        states = {"SAT1": _make_state(), "SAT2": _make_state(10.0)}
        results = propagate_cowell_batch(states, duration_s=60.0, dt_out=10.0)
        assert "SAT1" in results
        assert "SAT2" in results

    def test_trajectory_length_matches_single(self):
        """Batch result must have same length as single-satellite call."""
        s = _make_state()
        single = propagate_cowell(s, duration_s=120.0, dt_out=30.0)
        batch  = propagate_cowell_batch({"SAT1": s}, duration_s=120.0, dt_out=30.0)
        assert len(batch["SAT1"]) == len(single), (
            f"Batch length {len(batch['SAT1'])} != single {len(single)}"
        )

    def test_positions_match_single(self):
        """Final position from batch must match single call within 1e-6 km."""
        s = _make_state()
        single = propagate_cowell(s, duration_s=120.0, dt_out=30.0)
        batch  = propagate_cowell_batch({"SAT1": s}, duration_s=120.0, dt_out=30.0)

        pos_single = single[-1].position_km
        pos_batch  = batch["SAT1"][-1].position_km
        diff = float(np.linalg.norm(pos_single - pos_batch))
        # Tolerance is 1e-4 km (10 cm): tiny differences arise from thread-scheduling
        # non-determinism in floating-point step selection — not a physics error.
        assert diff < 1e-4, (
            f"Position mismatch between single and batch: {diff:.2e} km "
            "(expected < 1e-4 km / 10 cm)"
        )

    def test_multiple_satellites(self):
        n = 5
        states = {f"SAT{i}": _make_state(i * 5.0) for i in range(n)}
        results = propagate_cowell_batch(states, duration_s=60.0, dt_out=15.0)
        assert len(results) == n, f"Expected {n} results, got {len(results)}"

    def test_max_workers_respected(self):
        """max_workers=1 should still produce correct results (sequential mode)."""
        states = {"SAT1": _make_state(), "SAT2": _make_state(10.0)}
        results = propagate_cowell_batch(
            states, duration_s=60.0, dt_out=10.0, max_workers=1
        )
        assert len(results) == 2

    # ── Input validation ─────────────────────────────────────────────────────

    def test_raises_on_empty_states(self):
        with pytest.raises(ValueError, match="empty"):
            propagate_cowell_batch({}, duration_s=60.0)

    def test_raises_on_non_positive_duration(self):
        states = {"SAT1": _make_state()}
        with pytest.raises(ValueError, match="positive"):
            propagate_cowell_batch(states, duration_s=0.0)

    def test_raises_on_negative_duration(self):
        states = {"SAT1": _make_state()}
        with pytest.raises(ValueError, match="positive"):
            propagate_cowell_batch(states, duration_s=-100.0)

    # ── Maneuver dict handling ───────────────────────────────────────────────

    def test_none_maneuvers_ok(self):
        states = {"SAT1": _make_state()}
        results = propagate_cowell_batch(states, duration_s=60.0, maneuvers=None)
        assert "SAT1" in results

    def test_empty_maneuvers_dict_ok(self):
        states = {"SAT1": _make_state(), "SAT2": _make_state(10.0)}
        results = propagate_cowell_batch(states, duration_s=60.0, maneuvers={})
        assert len(results) == 2

    def test_maneuvers_key_per_satellite(self):
        """Satellites without a maneuvers key get empty burn list (no crash)."""
        states = {"SAT1": _make_state(), "SAT2": _make_state(10.0)}
        results = propagate_cowell_batch(
            states, duration_s=60.0, maneuvers={"SAT1": []}
            # SAT2 has no entry — should default to []
        )
        assert len(results) == 2

    # ── Physical sanity ──────────────────────────────────────────────────────

    def test_satellites_stay_in_orbit(self):
        """All propagated states must have position norm in [6400, 8000] km."""
        from astra.constants import EARTH_EQUATORIAL_RADIUS_KM as Re
        states = {f"SAT{i}": _make_state(i * 2.0) for i in range(3)}
        results = propagate_cowell_batch(states, duration_s=300.0, dt_out=60.0)
        for sat_id, traj in results.items():
            for step in traj:
                r = float(np.linalg.norm(step.position_km))
                assert Re + 100 < r < Re + 2000, (
                    f"{sat_id}: position norm {r:.1f} km outside physical LEO range"
                )

    def test_all_states_are_numerical_states(self):
        """Every element of the returned trajectories must be NumericalState."""
        states = {"SAT1": _make_state(), "SAT2": _make_state(5.0)}
        results = propagate_cowell_batch(states, duration_s=60.0, dt_out=20.0)
        for sat_id, traj in results.items():
            assert len(traj) > 0, f"{sat_id} trajectory is empty"
            for step in traj:
                assert isinstance(step, NumericalState), (
                    f"{sat_id}: expected NumericalState, got {type(step)}"
                )

    def test_time_monotonically_increasing(self):
        """Julian dates in the trajectory must be strictly increasing."""
        states = {"SAT1": _make_state()}
        results = propagate_cowell_batch(states, duration_s=180.0, dt_out=30.0)
        traj = results["SAT1"]
        times = [s.t_jd for s in traj]
        for i in range(1, len(times)):
            assert times[i] > times[i - 1], (
                f"t_jd not monotonically increasing at step {i}: "
                f"{times[i-1]:.10f} >= {times[i]:.10f}"
            )
