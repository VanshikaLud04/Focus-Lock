"""
camera.py -- threaded OpenCV camera capture

Runs capture in a background thread so the main loop never blocks
waiting for a new frame. Drop-in replacement for a blocking read loop.
"""

import threading
import cv2
import numpy as np
from typing import Generator, Optional


class CameraCapture:
    """
    Thread-safe webcam wrapper.

    Usage
    -----
    cam = CameraCapture(cfg["camera"])
    for frame in cam.stream():
        # frame is a BGR numpy array
        ...
    cam.release()
    """

    def __init__(self, cfg: dict) -> None:
        self._index  = cfg.get("index", 0)
        self._width  = cfg.get("width", 1280)
        self._height = cfg.get("height", 720)
        self._fps    = cfg.get("fps", 30)

        self._cap    = self._open()
        self._frame: Optional[np.ndarray] = None
        self._lock   = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None

        self._start_thread()

    def _open(self) -> cv2.VideoCapture:
        cap = cv2.VideoCapture(self._index)
        if not cap.isOpened():
            raise RuntimeError(
                f"Cannot open camera index {self._index}. "
                "Check config.yaml -> camera.index."
            )
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self._width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        cap.set(cv2.CAP_PROP_FPS,          self._fps)
        return cap

    def _start_thread(self) -> None:
        self._running = True
        self._thread  = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()

    def _reader(self) -> None:
        """Continuously grab frames into self._frame."""
        while self._running:
            ret, frame = self._cap.read()
            if ret:
                with self._lock:
                    self._frame = frame

    def read(self) -> Optional[np.ndarray]:
        """Return the latest frame, or None if not yet available."""
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    def stream(self) -> Generator[np.ndarray, None, None]:
        """Yield frames indefinitely, skipping if none are ready yet."""
        while self._running:
            frame = self.read()
            if frame is not None:
                yield frame

    def release(self) -> None:
        """Stop the reader thread and release the device."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
        self._cap.release()

    @property
    def resolution(self) -> tuple[int, int]:
        return (self._width, self._height)
