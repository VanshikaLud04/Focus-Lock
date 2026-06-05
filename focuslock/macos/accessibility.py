"""
accessibility.py -- macOS Accessibility API integration

Uses pyobjc to:
  - Detect which app is frontmost (bundle ID + display name)
  - Estimate whether the user is actively typing (keystroke rate)
  - Apply per-app focus threshold overrides

Setup:
    pip install pyobjc-framework-Cocoa pyobjc-framework-Quartz
    Grant Accessibility permission in System Settings -> Privacy & Security -> Accessibility
"""

from __future__ import annotations
import threading
import time
from dataclasses import dataclass
from typing import Optional


@dataclass
class AppContext:
    """
    Snapshot of the current macOS application context.
    Passed to GazeEstimator.is_focused() for threshold overrides.
    """
    bundle_id:    str            = "unknown"
    app_name:     str            = "Unknown"
    keystrokes_per_min: float    = 0.0
    yaw_override: Optional[float] = None   # None means use config default


class AccessibilityMonitor:
    """
    Background thread that polls the macOS Accessibility API every N seconds.

    Usage
    -----
    monitor = AccessibilityMonitor(cfg["accessibility"])
    context = monitor.get_context()   # thread-safe read
    monitor.stop()
    """

    def __init__(self, cfg: dict) -> None:
        self._enabled        = cfg.get("enabled",            True)
        self._poll_interval  = cfg.get("poll_interval_sec",  5.0)
        self._app_overrides  = cfg.get("app_overrides",      {})

        self._context  = AppContext()
        self._lock     = threading.Lock()
        self._running  = False
        self._thread: Optional[threading.Thread] = None

        self._keystroke_count = 0
        self._keystroke_ts    = time.time()

        if self._enabled:
            self._start()

    def _start(self) -> None:
        self._running = True
        self._thread  = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

    def _poll_loop(self) -> None:
        while self._running:
            try:
                ctx = self._query_macos()
                with self._lock:
                    self._context = ctx
            except Exception as e:
                print(f"[AccessibilityMonitor] {e} -- grant Accessibility permission in System Settings.")
            time.sleep(self._poll_interval)

    def _query_macos(self) -> AppContext:
        """
        Query the macOS Accessibility API for the frontmost app.

        Uncomment below once pyobjc-framework-Cocoa is installed.
        """
        # from AppKit import NSWorkspace
        # ws  = NSWorkspace.sharedWorkspace()
        # app = ws.frontmostApplication()
        # if app is None:
        #     return AppContext()
        # bundle_id = app.bundleIdentifier() or "unknown"
        # app_name  = app.localizedName()     or "Unknown"
        # yaw_override = self._app_overrides.get(bundle_id, None)
        # kpm = self._get_keystroke_rate()
        # return AppContext(
        #     bundle_id            = bundle_id,
        #     app_name             = app_name,
        #     keystrokes_per_min   = kpm,
        #     yaw_override         = yaw_override,
        # )
        return AppContext()   # stub

    def _get_keystroke_rate(self) -> float:
        """
        Returns keystrokes per minute since last call.
        Counts events only -- never reads key content.
        """
        now       = time.time()
        elapsed   = now - self._keystroke_ts
        rate      = (self._keystroke_count / elapsed) * 60 if elapsed > 0 else 0.0
        self._keystroke_count = 0
        self._keystroke_ts    = now
        return rate

    def get_context(self) -> AppContext:
        """Thread-safe read of the latest app context."""
        if not self._enabled:
            return AppContext()
        with self._lock:
            return self._context

    def stop(self) -> None:
        """Stop the background polling thread."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
