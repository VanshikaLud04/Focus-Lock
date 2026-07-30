"""
resources.py -- Thread-safe resource monitor
"""

import threading
import time
import psutil
import os
from typing import Dict, Any

class ResourceMonitor:
    """
    Background thread that samples CPU, RAM, and maintains rolling FPS.
    """
    def __init__(self, interval_sec: float = 2.0):
        self.interval_sec = interval_sec
        self._running = False
        self._thread = None
        self._lock = threading.Lock()
        
        self.process = psutil.Process(os.getpid())
        
        self._metrics = {
            "cpu_percent": 0.0,
            "ram_mb": 0.0,
            "fps": 0.0
        }
        
        # FPS Tracking
        self._frame_count = 0
        self._last_fps_time = time.monotonic()

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)

    def tick_frame(self) -> None:
        """Called by the inference thread every time a frame is processed."""
        with self._lock:
            self._frame_count += 1

    def _run_loop(self) -> None:
        while self._running:
            try:
                # CPU is measured as a percentage over the interval (non-blocking if interval=None but since we sleep it's fine)
                cpu = self.process.cpu_percent(interval=None) / psutil.cpu_count()
                ram_info = self.process.memory_info()
                ram_mb = ram_info.rss / (1024 * 1024)
                
                now = time.monotonic()
                with self._lock:
                    elapsed = now - self._last_fps_time
                    if elapsed > 0:
                        fps = self._frame_count / elapsed
                    else:
                        fps = 0.0
                    
                    self._frame_count = 0
                    self._last_fps_time = now
                    
                    self._metrics["cpu_percent"] = round(cpu, 1)
                    self._metrics["ram_mb"] = round(ram_mb, 1)
                    self._metrics["fps"] = round(fps, 1)
            except Exception as e:
                print(f"[ResourceMonitor] Error collecting metrics: {e}")
            
            time.sleep(self.interval_sec)

    def get_metrics(self) -> Dict[str, Any]:
        with self._lock:
            return self._metrics.copy()
