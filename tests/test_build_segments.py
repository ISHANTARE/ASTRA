"""TS-03: Edge-case tests for propagator._build_segments.

The audit identified that _build_segments was exercised only through happy-path
propagation calls. The segmentation boundary arithmetic is load-bearing: an
error here silently skips burns or produces zero-duration integrations, neither
of which raises an exception — they simply produce wrong trajectories.

Signature:  _build_segments(t_jd0, duration_s, burns)
Tests are offline and deterministic (no SGP4 or network calls).
"""
from __future__ import annotations

import math
import pytest
import numpy as np

from astra.propagator import _build_segments
from astra.models import FiniteBurn, ManeuverFrame


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_T_JD0 = 0.0  # anchor all burns relative to J2000 for simplicity


def _make_burn(start_s: float, duration_s: float, t_jd0: float = _T_JD0) -> FiniteBurn:
    """Create a minimal FiniteBurn anchored at t_jd0 + start_s."""
    ign_jd = t_jd0 + start_s / 86400.0
    return FiniteBurn(
        epoch_ignition_jd=ign_jd,
        duration_s=duration_s,
        thrust_N=10.0,
        isp_s=300.0,
        direction=(1.0, 0.0, 0.0),
        frame=ManeuverFrame.VNB,
    )


# ---------------------------------------------------------------------------
# Case 1: No burns → single coast segment spanning full duration
# ---------------------------------------------------------------------------


def test_no_burns_produces_single_coast():
    """Zero burns → exactly one coast segment from 0 to duration_s."""
    segs = _build_segments(_T_JD0, 3600.0, [])
    assert len(segs) == 1
    t_start, t_end, burn = segs[0]
    assert t_start == pytest.approx(0.0)
    assert t_end == pytest.approx(3600.0)
    assert burn is None


# ---------------------------------------------------------------------------
# Case 2: Burn in the middle → coast | powered | coast
# ---------------------------------------------------------------------------


def test_single_burn_middle_produces_three_segments():
    """Burn in the middle → pre-coast, powered arc, post-coast."""
    b = _make_burn(start_s=600.0, duration_s=60.0)
    segs = _build_segments(_T_JD0, 3600.0, [b])

    assert len(segs) == 3

    t0, t1, burn0 = segs[0]  # pre-coast
    assert burn0 is None
    assert t0 == pytest.approx(0.0)
    assert t1 == pytest.approx(600.0)

    t0, t1, burn1 = segs[1]  # powered arc
    assert burn1 is b
    assert t0 == pytest.approx(600.0)
    assert t1 == pytest.approx(660.0)

    t0, t1, burn2 = segs[2]  # post-coast
    assert burn2 is None
    assert t0 == pytest.approx(660.0)
    assert t1 == pytest.approx(3600.0)


# ---------------------------------------------------------------------------
# Case 3: Burn starting at exactly t=0 → no pre-coast segment
# ---------------------------------------------------------------------------


def test_burn_at_t0_has_no_pre_coast():
    """A burn igniting at t=0 must NOT produce a zero-duration pre-coast."""
    b = _make_burn(start_s=0.0, duration_s=120.0)
    segs = _build_segments(_T_JD0, 3600.0, [b])

    # Expect: powered arc [0, 120] then coast [120, 3600]
    assert len(segs) == 2

    t0, t1, burn0 = segs[0]
    assert burn0 is b
    assert t0 == pytest.approx(0.0)
    assert t1 == pytest.approx(120.0)

    t0, t1, burn1 = segs[1]
    assert burn1 is None
    assert t0 == pytest.approx(120.0)
    assert t1 == pytest.approx(3600.0)


# ---------------------------------------------------------------------------
# Case 4: Burn ending exactly at duration_s → no post-coast segment
# ---------------------------------------------------------------------------


def test_burn_ending_at_duration_has_no_post_coast():
    """Burn whose cutoff aligns with duration_s must NOT produce a zero-duration post-coast."""
    duration_s = 3600.0
    b = _make_burn(start_s=3540.0, duration_s=60.0)   # cutoff = 3600 exactly
    segs = _build_segments(_T_JD0, duration_s, [b])

    # Expect: coast [0, 3540] then powered arc [3540, 3600]
    assert len(segs) == 2

    t0, t1, burn0 = segs[0]
    assert burn0 is None
    assert t0 == pytest.approx(0.0)
    assert t1 == pytest.approx(3540.0)

    t0, t1, burn1 = segs[1]
    assert burn1 is b
    assert t0 == pytest.approx(3540.0)
    assert t1 == pytest.approx(3600.0)


# ---------------------------------------------------------------------------
# Case 5: Burn spanning the entire window → single powered arc, no coasts
# ---------------------------------------------------------------------------


def test_burn_spanning_entire_window():
    """A burn from 0 to duration_s → exactly one powered segment, no coasts."""
    duration_s = 3600.0
    b = _make_burn(start_s=0.0, duration_s=duration_s)
    segs = _build_segments(_T_JD0, duration_s, [b])

    assert len(segs) == 1
    t0, t1, burn0 = segs[0]
    assert burn0 is b
    assert t0 == pytest.approx(0.0)
    assert t1 == pytest.approx(duration_s)


# ---------------------------------------------------------------------------
# Case 6: Two adjacent burns with zero coast gap between them
# ---------------------------------------------------------------------------


def test_two_adjacent_burns_no_gap():
    """Burns with zero coast gap → coast | powered | powered | coast; no zero-duration gaps."""
    b1 = _make_burn(start_s=100.0, duration_s=50.0)   # [100, 150]
    b2 = _make_burn(start_s=150.0, duration_s=50.0)   # [150, 200]
    segs = _build_segments(_T_JD0, 3600.0, [b1, b2])

    # All segments must have strictly positive duration
    for t0, t1, _ in segs:
        assert t1 - t0 > 1e-9, f"Zero-duration segment found: [{t0}, {t1}]"

    # Exactly two powered segments
    powered = [(t0, t1, bk) for t0, t1, bk in segs if bk is not None]
    assert len(powered) == 2

    # Boundaries must be contiguous
    for i in range(len(segs) - 1):
        assert segs[i][1] == pytest.approx(segs[i + 1][0]), (
            f"Non-contiguous boundary between segs[{i}] and segs[{i+1}]"
        )


# ---------------------------------------------------------------------------
# Case 7: Burn entirely before the window → completely ignored
# ---------------------------------------------------------------------------


def test_burn_entirely_before_window_is_ignored():
    """A burn completing before the propagation window must be silently ignored."""
    # t_jd0 = 1000s into absolute time; burn ends at 60s abs = 940s before window
    t_jd0 = 1000.0 / 86400.0
    b = FiniteBurn(
        epoch_ignition_jd=0.0,   # 1000 s before window start
        duration_s=60.0,
        thrust_N=10.0,
        isp_s=300.0,
        direction=(1.0, 0.0, 0.0),
        frame=ManeuverFrame.VNB,
    )
    segs = _build_segments(t_jd0, 3600.0, [b])

    powered = [s for s in segs if s[2] is not None]
    assert len(powered) == 0, "Burn before window must produce no powered segment"
    assert len(segs) == 1 and segs[0][2] is None, "Should be one full-duration coast"


# ---------------------------------------------------------------------------
# Case 8: Burn straddling window start → clipped to start at 0
# ---------------------------------------------------------------------------


def test_burn_straddling_window_start_is_clipped():
    """A burn starting before window start but ending inside must be clipped to t=0."""
    # Window starts at abs time 500 s; burn ignites at abs 400 s (100 s before),
    # duration 300 s → cutoff at abs 700 s → 200 s into window.
    t_jd0 = 500.0 / 86400.0
    b = FiniteBurn(
        epoch_ignition_jd=400.0 / 86400.0,
        duration_s=300.0,
        thrust_N=10.0,
        isp_s=300.0,
        direction=(1.0, 0.0, 0.0),
        frame=ManeuverFrame.VNB,
    )
    segs = _build_segments(t_jd0, 3600.0, [b])

    powered = [(t0, t1, bk) for t0, t1, bk in segs if bk is not None]
    assert len(powered) == 1, "Clipped burn must produce exactly one powered segment"
    t0, t1, _ = powered[0]
    assert t0 == pytest.approx(0.0), "Clipped ignition must start at t=0"
    assert t1 == pytest.approx(200.0), "Cutoff = (700 - 500) = 200 s relative to window"


# ---------------------------------------------------------------------------
# Case 9: Burn straddling window end → cutoff clipped to duration_s
# ---------------------------------------------------------------------------


def test_burn_straddling_window_end_is_clipped():
    """A burn extending past duration_s must be clipped at the window boundary."""
    duration_s = 3600.0
    b = _make_burn(start_s=3500.0, duration_s=200.0)   # cutoff = 3700 > 3600
    segs = _build_segments(_T_JD0, duration_s, [b])

    powered = [(t0, t1, bk) for t0, t1, bk in segs if bk is not None]
    assert len(powered) == 1
    _, t1, _ = powered[0]
    assert t1 == pytest.approx(duration_s), (
        f"Cutoff must be clipped to duration_s={duration_s}, got {t1}"
    )


# ---------------------------------------------------------------------------
# Case 10: All segment boundaries are non-negative, ordered, and contiguous
# ---------------------------------------------------------------------------


def test_segment_endpoints_are_monotone_and_contiguous():
    """With multiple burns, all segment boundaries must be ≥ 0, ordered, and contiguous."""
    burns = [
        _make_burn(start_s=300.0,  duration_s=60.0),
        _make_burn(start_s=600.0,  duration_s=120.0),
        _make_burn(start_s=900.0,  duration_s=30.0),
    ]
    segs = _build_segments(_T_JD0, 3600.0, burns)

    prev_end = 0.0
    for i, (t0, t1, _) in enumerate(segs):
        assert t0 >= 0.0, f"seg[{i}]: start {t0} < 0"
        assert t1 > t0, f"seg[{i}]: end {t1} <= start {t0}"
        assert t0 == pytest.approx(prev_end, abs=1e-9), (
            f"seg[{i}]: gap at boundary; expected t0={prev_end}, got {t0}"
        )
        prev_end = t1

    assert prev_end == pytest.approx(3600.0), "Final segment must reach duration_s"


# ---------------------------------------------------------------------------
# Case 11: seg_duration < 1e-9 guard — sub-nanosecond segments never emitted
# ---------------------------------------------------------------------------


def test_no_sub_nanosecond_segments_produced():
    """_build_segments must never emit segments shorter than 1e-9 s."""
    b1 = _make_burn(start_s=0.0, duration_s=100.0)
    # Second burn ignites 5e-10 s (sub-ns) after b1 cutoff
    ign2_jd = _T_JD0 + (100.0 + 5e-10) / 86400.0
    b2 = FiniteBurn(
        epoch_ignition_jd=ign2_jd,
        duration_s=100.0,
        thrust_N=10.0,
        isp_s=300.0,
        direction=(0.0, 1.0, 0.0),
        frame=ManeuverFrame.VNB,
    )
    segs = _build_segments(_T_JD0, 3600.0, [b1, b2])
    for t0, t1, _ in segs:
        assert t1 - t0 >= 1e-9, (
            f"Sub-nanosecond segment emitted: [{t0:.15f}, {t1:.15f}] "
            f"(duration = {t1 - t0:.3e} s)"
        )


# ---------------------------------------------------------------------------
# Case 12: Overlap in relaxed mode → later burn skipped
# ---------------------------------------------------------------------------


def test_overlapping_burns_relaxed_mode_skips_later() -> None:
    """In relaxed mode, the second overlapping burn is skipped; b1 survives."""
    import astra.config as cfg

    prev = cfg.ASTRA_STRICT_MODE
    cfg.ASTRA_STRICT_MODE = False
    try:
        b1 = _make_burn(start_s=0.0, duration_s=200.0)    # [0, 200]
        b2 = _make_burn(start_s=100.0, duration_s=100.0)  # [100, 200] — overlaps b1

        segs = _build_segments(_T_JD0, 3600.0, [b1, b2])

        powered = [(t0, t1, bk) for t0, t1, bk in segs if bk is not None]
        assert len(powered) == 1, (
            f"Expected 1 powered segment (b2 skipped), got {len(powered)}"
        )
        assert powered[0][2] is b1, "Surviving burn must be b1 (the earlier one)"
    finally:
        cfg.ASTRA_STRICT_MODE = prev


# ---------------------------------------------------------------------------
# Case 13: Overlap in STRICT mode → ManeuverError raised immediately
# ---------------------------------------------------------------------------


def test_overlapping_burns_strict_mode_raises() -> None:
    """In STRICT mode, overlapping burns raise ManeuverError immediately."""
    import astra.config as cfg
    from astra.errors import ManeuverError

    prev = cfg.ASTRA_STRICT_MODE
    cfg.ASTRA_STRICT_MODE = True
    try:
        b1 = _make_burn(start_s=0.0,   duration_s=200.0)
        b2 = _make_burn(start_s=100.0, duration_s=100.0)

        with pytest.raises(ManeuverError, match="Temporal overlap detected"):
            _build_segments(_T_JD0, 3600.0, [b1, b2])
    finally:
        cfg.ASTRA_STRICT_MODE = prev
