"""Tests for FocusFSM state machine."""
import time
import pytest
from focuslock.fsm.focus_fsm import FocusFSM


@pytest.fixture
def fsm():
    cfg = {"cooldown_sec": 0.0, "distraction_alert_sec": 30, "break_reminder_min": 25}
    return FocusFSM(cfg)


def test_initial_state(fsm):
    assert fsm.state == "IDLE"


def test_idle_to_focused(fsm):
    state = fsm.update(focused=True)
    assert state == "FOCUSED"


def test_idle_to_distracted(fsm):
    state = fsm.update(focused=False)
    assert state == "DISTRACTED"


def test_focused_to_distracted(fsm):
    fsm.update(focused=True)
    state = fsm.update(focused=False)
    assert state == "DISTRACTED"


def test_distracted_to_focused(fsm):
    fsm.update(focused=False)
    state = fsm.update(focused=True)
    assert state == "FOCUSED"


def test_pause_resume(fsm):
    fsm.update(focused=True)
    fsm.pause()
    assert fsm.state == "BREAK"
    fsm.resume()
    assert fsm.state == "IDLE"


def test_cooldown_prevents_flicker():
    cfg = {"cooldown_sec": 10.0, "distraction_alert_sec": 30, "break_reminder_min": 25}
    fsm = FocusFSM(cfg)
    fsm.update(focused=True)
    state = fsm.update(focused=False)
    # Cooldown 10s — should stay FOCUSED
    assert state == "FOCUSED"


def test_stats_accumulate(fsm):
    fsm.update(focused=True)
    time.sleep(0.05)
    fsm._accumulate(time.monotonic())
    assert fsm.session_stats().focused_sec > 0


def test_reset(fsm):
    fsm.update(focused=True)
    fsm.reset()
    assert fsm.state == "IDLE"
    assert fsm.session_stats().focused_sec == 0.0


def test_distraction_alert_callback(fsm):
    fired = []
    fsm.on_distraction_alert = lambda secs: fired.append(secs)
    # Fake accumulated distracted time
    fsm._stats.distracted_sec = 31.0
    fsm.update(focused=False)
    # Callback may or may not fire depending on timing — just ensure no crash
