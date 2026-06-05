"""
menubar.py -- macOS menu bar app (rumps)

Replaces the terminal window with a native macOS menu bar icon
showing current focus state, session stats, and controls.

State icons:
  green circle  FOCUSED
  red circle    DISTRACTED
  pause         BREAK / PAUSED
  white circle  IDLE

Requires:
    pip install rumps
    rumps only works in a non-sandboxed Terminal or packaged .app.
"""

from __future__ import annotations
from typing import Optional


STATE_ICONS = {
    "FOCUSED":    "( )",
    "DISTRACTED": "(x)",
    "BREAK":      "| |",
    "IDLE":       "( )",
}


class FocusLockMenuBarApp:
    """
    Wraps rumps.App to display Focus Lock state in the macOS menu bar.

    Usage
    -----
    app = FocusLockMenuBarApp(cfg)
    app.run()   # blocks until quit
    """

    def __init__(self, cfg: dict) -> None:
        self._cfg = cfg
        # TODO: implement with rumps
        # import rumps
        # self._app = rumps.App("--", quit_button="Quit Focus Lock")
        # self._setup_menu()

    def run(self) -> None:
        """Start the menu bar app event loop (blocking)."""
        # TODO: self._app.run()
        print("[MenuBar] rumps not yet wired up.")

    def update_state(self, state: str, stats) -> None:
        """
        Push a state update from the main session loop.

        Parameters
        ----------
        state : str          -- current FSM state label
        stats : SessionStats -- from FocusFSM.session_stats()
        """
        # TODO:
        # icon = STATE_ICONS.get(state, "--")
        # self._app.title = icon
        # self._app.menu["Session"].title = f"Session: {self._fmt_time(stats.total_sec)}"
        # self._app.menu["Focus"].title   = f"Focus: {stats.focus_pct}%"
        pass

    @staticmethod
    def _fmt_time(seconds: float) -> str:
        """Format seconds as MM:SS."""
        m, s = divmod(int(seconds), 60)
        return f"{m:02d}:{s:02d}"

    def _on_pause(self, sender) -> None:
        # TODO: fsm.pause() / fsm.resume() via shared state
        pass

    def _on_view_report(self, sender) -> None:
        # TODO: generate HTML report and open with subprocess
        pass
