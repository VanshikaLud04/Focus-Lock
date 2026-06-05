"""
overlay.py -- OpenCV HUD overlay

Draws state, angles, FPS, and session stats onto the live camera frame.
"""

from __future__ import annotations
import time
from typing import Optional
import cv2
import numpy as np

# Colour palette (BGR)
COLOURS = {
    "FOCUSED":    (80, 210, 100),    # green
    "DISTRACTED": (60,  80, 240),    # red
    "BREAK":      (180, 160, 50),    # blue
    "IDLE":       (120, 120, 120),   # grey
    "white":      (240, 240, 240),
    "muted":      (100, 100, 100),
}

STATE_LABELS = {
    "FOCUSED":    "FOCUSED",
    "DISTRACTED": "DISTRACTED",
    "BREAK":      "BREAK",
    "IDLE":       "IDLE",
}


class HUDOverlay:
    """
    Draws the live HUD onto each frame.

    Usage
    -----
    hud = HUDOverlay(cfg)
    annotated = hud.draw(frame, state, stats, gaze_result, detections)
    """

    def __init__(self, cfg: dict) -> None:
        self._privacy_blur = cfg.get("privacy", {}).get("blur_face_in_hud", False)
        self._fps_history  = []
        self._last_ts      = time.monotonic()

    def draw(
        self,
        frame,
        state:       str,
        stats,
        gaze_result  = None,
        detections   = None,
    ) -> np.ndarray:
        """
        Overlay the HUD onto a copy of the frame and return it.

        Parameters
        ----------
        frame      : np.ndarray      BGR camera frame
        state      : str             current FSM state label
        stats      : SessionStats
        gaze_result: GazeResult | None
        detections : DetectionResult | None
        """
        out   = frame.copy()
        h, w  = out.shape[:2]
        color = COLOURS.get(state, COLOURS["IDLE"])

        # border
        thickness = 6
        cv2.rectangle(out, (0, 0), (w - 1, h - 1), color, thickness)

        # state badge (top-left)
        label = STATE_LABELS.get(state, state)
        self._put_badge(out, label, x=20, y=20, color=color)

        # FPS (top-right)
        fps = self._compute_fps()
        self._put_text(out, f"FPS: {fps:.0f}", x=w - 120, y=30, color=COLOURS["muted"])

        # gaze angles (bottom-left)
        if gaze_result and gaze_result.face_found:
            self._put_text(out, f"Yaw:   {gaze_result.yaw:+.1f}",   x=20, y=h - 80)
            self._put_text(out, f"Pitch: {gaze_result.pitch:+.1f}", x=20, y=h - 55)
        else:
            self._put_text(out, "User face not detected", x=20, y=h - 55, color=(60, 60, 200))

        # session stats (bottom-right)
        if stats:
            self._put_text(out, f"Session: {self._fmt(stats.total_sec)}",   x=w - 200, y=h - 80)
            self._put_text(out, f"Focus:   {stats.focus_pct:.0f}%",         x=w - 200, y=h - 55)

        # phone warning
        if detections and detections.has_phone:
            self._put_text(out, "Phone detected", x=20, y=70, color=(40, 80, 220))

        return out

    def _put_badge(
        self, frame, text: str, x: int, y: int,
        color: tuple, font_scale: float = 0.7,
    ) -> None:
        font      = cv2.FONT_HERSHEY_DUPLEX
        thickness = 1
        (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
        pad = 8
        cv2.rectangle(frame, (x - pad, y - pad), (x + tw + pad, y + th + pad), color, -1)
        cv2.putText(frame, text, (x, y + th), font, font_scale, (10, 10, 10), thickness, cv2.LINE_AA)

    def _put_text(
        self, frame, text: str, x: int, y: int,
        color: Optional[tuple] = None, font_scale: float = 0.55,
    ) -> None:
        color = color or COLOURS["white"]
        cv2.putText(frame, text, (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                    (10, 10, 10), 3, cv2.LINE_AA)
        cv2.putText(frame, text, (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                    color, 1, cv2.LINE_AA)

    def _compute_fps(self) -> float:
        """Rolling 30-frame FPS estimate."""
        now = time.monotonic()
        self._fps_history.append(now - self._last_ts)
        self._last_ts = now
        if len(self._fps_history) > 30:
            self._fps_history.pop(0)
        avg_dt = sum(self._fps_history) / len(self._fps_history)
        return 1.0 / avg_dt if avg_dt > 0 else 0.0

    @staticmethod
    def _fmt(seconds: float) -> str:
        m, s = divmod(int(seconds), 60)
        return f"{m:02d}:{s:02d}"
