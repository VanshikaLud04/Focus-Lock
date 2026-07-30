"""
attention.py -- Attention Engine
"""

from __future__ import annotations
import time
from .events import AttentionEvent, EventBus
from .resources import ResourceMonitor

class AttentionEngine:
    """
    Normalizes FSM state and CV detections into a unified AttentionEvent.
    """
    def __init__(self, event_bus: EventBus, resource_monitor: ResourceMonitor):
        self.bus = event_bus
        self.rm = resource_monitor

    def process_frame(
        self,
        session_id: int,
        state: str,
        confidence: float,
        trigger: str,
        interval_ms: float,
        yaw: float,
        pitch: float,
        active_app: str,
        face_visible: bool,
        phone_detected: bool,
    ) -> None:
        """
        Constructs and emits an AttentionEvent based on raw state.
        """
        metrics = self.rm.get_metrics()
        
        # We can calculate instantaneous FPS if we want, or use the rolling one.
        # interval_ms is the AdaptiveSampler's requested interval.
        fps_instant = 1000.0 / interval_ms if interval_ms > 0 else metrics["fps"]

        event = AttentionEvent(
            timestamp=time.time(),
            session_id=session_id,
            attention_state=state,
            trigger=trigger,
            confidence=confidence,
            fps=fps_instant,
            cpu=metrics["cpu_percent"],
            ram=metrics["ram_mb"],
            face_visible=face_visible,
            phone_detected=phone_detected,
            gaze_yaw=yaw,
            gaze_pitch=pitch,
            active_app=active_app
        )
        
        self.emit(event)

    def emit(self, event: AttentionEvent) -> None:
        self.bus.publish("attention_events", event)

    def flush(self) -> None:
        pass
