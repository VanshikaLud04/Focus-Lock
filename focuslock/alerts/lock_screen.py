"""
lock_screen.py — Full-screen "punishment" overlay
==================================================
Uses tkinter to paint a borderless, always-on-top, full-screen
overlay the moment a phone-in-hand is detected.

The overlay:
  • Covers the entire screen (including the menu bar on macOS)
  • Cannot be clicked through or dismissed by the user
  • Auto-dismisses when the phone is no longer detected
  • Shows a live countdown of how long the lock has been active
  • Pulses a border to draw attention

Usage
-----
    lock = LockScreenOverlay()
    lock.show(reason="Phone in hand")   # from detection thread
    lock.hide()                          # when phone is put down
"""

from __future__ import annotations
import threading
import time
import tkinter as tk
from tkinter import font as tkfont


import subprocess
import sys
import os

class LockScreenOverlay:
    """
    Full-screen lock overlay.
    Spawns a subprocess to ensure tkinter runs on the main thread,
    avoiding macOS NSWindow cross-thread crashes.
    """
    def __init__(self) -> None:
        self._active = False
        self._proc: subprocess.Popen | None = None

    def show(self, reason: str = "Phone detected") -> None:
        """Show the overlay (no-op if already visible)."""
        if self._active:
            return
        self._active = True
        script_path = os.path.abspath(__file__)
        self._proc = subprocess.Popen([sys.executable, script_path, reason])

    def hide(self) -> None:
        """Dismiss the overlay."""
        self._active = False
        if self._proc:
            self._proc.terminate()
            self._proc = None

    @property
    def is_visible(self) -> bool:
        return self._active


def _run_ui_process(reason: str) -> None:
    # Warm dark palette (matches dashboard)
    BG       = "#0d0b08"
    ACCENT   = "#c4987a"    # warm peach
    TITLE_FG = "#e4d8cc"
    SUB_FG   = "#7a7068"
    BORDER   = "#2a2420"

    root = tk.Tk()
    lock_start = time.monotonic()

    # ── Window setup ─────────────────────────────────────────
    root.configure(bg=BG)
    root.attributes("-fullscreen",  True)
    root.attributes("-topmost",     True)
    root.attributes("-alpha",       0.96)
    root.overrideredirect(True)          # no title bar / borders
    root.lift()
    root.focus_force()

    root.bind_all("<Key>",    lambda e: "break")
    root.bind_all("<Button>", lambda e: "break")

    border_frame = tk.Frame(root, bg=BORDER, bd=0)
    border_frame.place(relx=0.5, rely=0.5, anchor="center",
                       width=460, height=340)

    inner = tk.Frame(border_frame, bg=BG, bd=0)
    inner.place(x=2, y=2, width=456, height=336)

    try:
        emoji_font = tkfont.Font(family="Apple Color Emoji", size=56)
    except Exception:
        emoji_font = tkfont.Font(size=48)

    tk.Label(inner, text="🔒", font=emoji_font,
             bg=BG, fg=ACCENT).pack(pady=(36, 4))

    tk.Label(inner, text="Focus Lock — Screen Locked",
             font=("SF Pro Display", 20, "bold"),
             bg=BG, fg=TITLE_FG).pack()

    reason_lbl = tk.Label(inner, text=reason,
                          font=("SF Pro", 14),
                          bg=BG, fg=ACCENT)
    reason_lbl.pack(pady=(6, 0))

    tk.Label(inner,
             text="Put your phone down to unlock.",
             font=("SF Pro", 13),
             bg=BG, fg=SUB_FG).pack(pady=(4, 0))

    timer_lbl = tk.Label(inner, text="Locked for 0s",
                         font=("SF Mono", 12),
                         bg=BG, fg=SUB_FG)
    timer_lbl.pack(pady=(14, 0))

    def _tick():
        elapsed = int(time.monotonic() - lock_start)
        m, s    = divmod(elapsed, 60)
        timer_lbl.config(
            text = f"Locked for {m}m {s:02d}s" if m else f"Locked for {s}s"
        )
        root.after(1000, _tick)

    colours = [BORDER, "#3a2820", "#2a2420", "#3a2820"]
    state   = [0]

    def _step():
        border_frame.config(bg=colours[state[0] % len(colours)])
        state[0] += 1
        root.after(600, _step)

    root.after(1000, _tick)
    root.after(600, _step)

    root.mainloop()

if __name__ == "__main__":
    reason_arg = sys.argv[1] if len(sys.argv) > 1 else "Phone detected"
    _run_ui_process(reason_arg)
