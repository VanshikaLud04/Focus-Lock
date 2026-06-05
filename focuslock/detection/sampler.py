"""
sampler.py -- adaptive sampling via motion entropy

Instead of a fixed sample rate, the interval adjusts based on how much
the scene is changing. High motion -> sample faster. Static scene -> slow down.

Algorithm:
  1. Compute frame diff:  diff = |frame_t - frame_{t-1}|
  2. Build histogram of diff pixel values.
  3. Compute Shannon entropy of the histogram.
  4. Map entropy to a sampling interval.
  5. Smooth with an EMA to avoid jitter from a single noisy frame.
"""

from __future__ import annotations
import time
import numpy as np
from scipy.stats import entropy as scipy_entropy


class AdaptiveSampler:
    """
    Decides whether the current frame should trigger full inference.

    Usage
    -----
    sampler = AdaptiveSampler(cfg["adaptive_sampler"])

    for frame in camera.stream():
        should_infer, interval_ms = sampler.tick(frame, prev_frame)
        if should_infer:
            run_yolo_and_gaze(frame)
    """

    def __init__(self, cfg: dict) -> None:
        self._enabled  = cfg.get("enabled",          True)
        self._min_int  = cfg.get("min_interval_sec", 0.2)
        self._max_int  = cfg.get("max_interval_sec", 3.0)
        self._alpha    = cfg.get("ema_alpha",         0.25)

        self._ema_interval: float    = self._min_int
        self._last_sample_ts: float  = 0.0

    @staticmethod
    def _frame_diff(prev: np.ndarray, curr: np.ndarray) -> np.ndarray:
        """Absolute per-pixel difference between two BGR frames."""
        import cv2
        return cv2.absdiff(prev, curr)

    @staticmethod
    def _motion_entropy(diff: np.ndarray) -> float:
        """
        Shannon entropy of the diff-image histogram.

        Returns roughly in [0, ~4]:
          0   -> completely static frame
          ~4  -> high motion / chaotic frame
        """
        hist, _ = np.histogram(diff.flatten(), bins=64, range=(0, 256))
        hist    = hist.astype(np.float64)
        hist   += 1e-9
        hist   /= hist.sum()
        return float(scipy_entropy(hist))

    def _entropy_to_interval(self, entropy_val: float) -> float:
        """
        Map entropy to a sampling interval.

        High entropy -> short interval (sample faster).
        Low entropy  -> long interval (save CPU).
        """
        ratio = float(np.clip(1.0 - entropy_val / 4.0, 0.0, 1.0))
        return self._min_int + ratio * (self._max_int - self._min_int)

    def tick(
        self,
        curr_frame: np.ndarray,
        prev_frame,   # np.ndarray | None
    ) -> tuple[bool, float]:
        """
        Called once per captured frame.

        Parameters
        ----------
        curr_frame : np.ndarray   BGR frame
        prev_frame : np.ndarray | None

        Returns
        -------
        (should_infer, interval_ms)
            should_infer -- True if YOLO+gaze inference should run.
            interval_ms  -- current EMA interval in milliseconds (for logging).
        """
        now = time.monotonic()

        if not self._enabled:
            self._last_sample_ts = now
            return True, self._min_int * 1000

        if prev_frame is None:
            self._last_sample_ts = now
            return True, self._ema_interval * 1000

        diff            = self._frame_diff(prev_frame, curr_frame)
        entropy_val     = self._motion_entropy(diff)
        target_interval = self._entropy_to_interval(entropy_val)

        self._ema_interval = (
            self._alpha * target_interval +
            (1 - self._alpha) * self._ema_interval
        )

        elapsed = now - self._last_sample_ts
        if elapsed >= self._ema_interval:
            self._last_sample_ts = now
            return True, self._ema_interval * 1000

        return False, self._ema_interval * 1000

    @property
    def current_interval_sec(self) -> float:
        """Current EMA-smoothed sampling interval in seconds."""
        return self._ema_interval
