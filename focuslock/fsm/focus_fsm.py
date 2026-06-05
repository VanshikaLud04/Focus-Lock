"""
focus_fsm.py -- finite state machine for tracking focus state

States:
  IDLE        -- session started, waiting for a face
  FOCUSED     -- face detected and within gaze thresholds
  DISTRACTED  -- head out of range or phone detected
  BREAK       -- user paused, or break reminder triggered

Transitions:
  IDLE        -> FOCUSED     : face detected + focused
  IDLE        -> DISTRACTED  : face detected + not focused
  FOCUSED     -> DISTRACTED  : not focused for > cooldown
  DISTRACTED  -> FOCUSED     : focused for > cooldown
  ANY         -> BREAK       : user presses pause or timer fires
  BREAK       -> IDLE        : user resumes

A cooldown prevents rapid flicker between states.
"""

from __future__ import annotations
import time
from dataclasses import dataclass, field


STATES = ("IDLE", "FOCUSED", "DISTRACTED", "BREAK")


@dataclass
class SessionStats:
    """Accumulated time (seconds) per state for the current session."""
    focused_sec:     float = 0.0
    distracted_sec:  float = 0.0
    break_sec:       float = 0.0
    idle_sec:        float = 0.0
    start_time:      float = field(default_factory=time.time)
    transition_count: int  = 0

    @property
    def total_sec(self) -> float:
        return time.time() - self.start_time

    @property
    def focus_pct(self) -> float:
        t = self.total_sec
        return round((self.focused_sec / t) * 100, 1) if t > 0 else 0.0


class FocusFSM:
    """
    Finite State Machine that tracks user focus state.

    Usage
    -----
    fsm   = FocusFSM(cfg["fsm"])
    state = fsm.update(focused=True)
    stats = fsm.session_stats()
    fsm.pause()
    fsm.resume()
    """

    def __init__(self, cfg: dict) -> None:
        self._cooldown       = cfg.get("cooldown_sec",           1.5)
        self._alert_secs     = cfg.get("distraction_alert_sec",  30)
        self._break_min      = cfg.get("break_reminder_min",     25)

        self.state: str             = "IDLE"
        self._prev_state: str       = "IDLE"
        # Initialise _last_transition in the past so the first update() is
        # never blocked by the cooldown guard.
        _now = time.monotonic()
        self._last_transition: float = _now - self._cooldown
        self._state_start: float     = _now

        self._stats         = SessionStats()
        self._alert_fired   = False

        # Callbacks -- assign from outside to react to events
        self.on_distraction_alert = None
        self.on_break_reminder    = None
        self.on_state_change      = None

    def update(self, focused: bool) -> str:
        """
        Feed the latest focus decision from gaze.is_focused().
        Returns the (possibly unchanged) current state label.
        """
        now = time.monotonic()

        self._accumulate(now)

        if now - self._last_transition < self._cooldown:
            return self.state

        new_state = self._next_state(focused)
        if new_state != self.state:
            self._transition(new_state, now)

        if self.state == "DISTRACTED":
            if self._stats.distracted_sec >= self._alert_secs and not self._alert_fired:
                self._alert_fired = True
                if self.on_distraction_alert:
                    self.on_distraction_alert(self._stats.distracted_sec)

        if self._stats.focused_sec > 0 and self.state == "FOCUSED":
            if self._stats.focused_sec % (self._break_min * 60) < 2:
                if self.on_break_reminder:
                    self.on_break_reminder(self._stats.focused_sec)

        return self.state

    def _next_state(self, focused: bool) -> str:
        """Pure transition logic, no side effects."""
        if self.state in ("IDLE", "BREAK"):
            return "FOCUSED" if focused else "DISTRACTED"
        if self.state == "FOCUSED":
            return "FOCUSED" if focused else "DISTRACTED"
        if self.state == "DISTRACTED":
            return "FOCUSED" if focused else "DISTRACTED"
        return self.state

    def _transition(self, new_state: str, now: float) -> None:
        self._prev_state      = self.state
        self.state            = new_state
        self._last_transition = now
        self._state_start     = now
        self._alert_fired     = False
        self._stats.transition_count += 1
        if self.on_state_change:
            self.on_state_change(self._prev_state, new_state)

    def _accumulate(self, now: float) -> None:
        """Add elapsed time to the correct counter."""
        delta = now - self._state_start
        self._state_start = now
        if self.state == "FOCUSED":
            self._stats.focused_sec    += delta
        elif self.state == "DISTRACTED":
            self._stats.distracted_sec += delta
        elif self.state == "BREAK":
            self._stats.break_sec      += delta
        else:
            self._stats.idle_sec       += delta

    def pause(self) -> None:
        """Enter BREAK state."""
        self._transition("BREAK", time.monotonic())

    def resume(self) -> None:
        """Exit BREAK state back to IDLE."""
        self._transition("IDLE", time.monotonic())

    def session_stats(self) -> SessionStats:
        """Return accumulated session stats (flushes the current interval first)."""
        self._accumulate(time.monotonic())
        return self._stats

    def reset(self) -> None:
        """Start a new session."""
        _now = time.monotonic()
        self._stats           = SessionStats()
        self.state            = "IDLE"
        self._last_transition = _now - self._cooldown
        self._state_start     = _now
