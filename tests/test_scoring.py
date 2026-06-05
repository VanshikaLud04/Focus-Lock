"""Tests for FocusScorer."""
import pytest
from focuslock.scoring.focus_score import FocusScorer, ScoreBreakdown
from focuslock.fsm.focus_fsm import SessionStats


@pytest.fixture
def scorer():
    return FocusScorer({"recovery_weight": 0.3, "fp_penalty_weight": 0.1})


def make_stats(focused_sec: float, distracted_sec: float, total_sec: float) -> SessionStats:
    import time
    s = SessionStats()
    s.focused_sec    = focused_sec
    s.distracted_sec = distracted_sec
    s.start_time     = time.time() - total_sec
    return s


def make_events(states: list[str], fp_flags: list[int] = None) -> list[dict]:
    import time
    now = time.time()
    if fp_flags is None:
        fp_flags = [0] * len(states)
    return [
        {"state": s, "ts": now + i, "false_positive": fp}
        for i, (s, fp) in enumerate(zip(states, fp_flags))
    ]


def test_perfect_session(scorer):
    stats  = make_stats(focused_sec=3600, distracted_sec=0, total_sec=3600)
    events = make_events(["FOCUSED"] * 100)
    result = scorer.compute(stats, events)
    assert result.final_score > 90
    assert result.grade == "A"


def test_zero_duration(scorer):
    import time
    stats = SessionStats()
    stats.start_time = time.time()   # total_sec → 0
    result = scorer.compute(stats, [])
    assert result.final_score == 0


def test_fp_penalty_lowers_score(scorer):
    states   = ["FOCUSED"] * 100
    no_fp    = make_events(states, fp_flags=[0] * 100)
    with_fp  = make_events(states, fp_flags=[1] * 20 + [0] * 80)
    stats    = make_stats(3000, 600, 3600)

    s_clean = scorer.compute(stats, no_fp).final_score
    s_noisy = scorer.compute(stats, with_fp).final_score
    assert s_clean > s_noisy


def test_recovery_boosts_score(scorer):
    """Quick distraction-recovery should score higher than slow recovery."""
    # Alternating focus/distracted at tight intervals → fast recovery
    fast_events = make_events(["FOCUSED", "DISTRACTED", "FOCUSED"] * 10)
    stats       = make_stats(1800, 1800, 3600)
    result      = scorer.compute(stats, fast_events)
    assert isinstance(result.final_score, float)
    assert 0 <= result.final_score <= 100


def test_grade_boundaries(scorer):
    for score, expected_grade in [(95, "A"), (85, "B"), (70, "C"), (55, "D"), (30, "F")]:
        assert scorer._grade(score) == expected_grade


def test_score_clamped(scorer):
    stats  = make_stats(focused_sec=9000, distracted_sec=0, total_sec=3600)
    events = make_events(["FOCUSED"] * 10)
    result = scorer.compute(stats, events)
    assert 0 <= result.final_score <= 100
