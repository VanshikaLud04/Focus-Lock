"""
events.py -- Event Bus for Domain Events
"""

from __future__ import annotations
import threading
from typing import Callable, Any, Dict, List
from dataclasses import dataclass

@dataclass
class DomainEvent:
    schema_version: int = 1

@dataclass
class AttentionEvent(DomainEvent):
    timestamp: float
    session_id: int
    attention_state: str
    trigger: str
    confidence: float
    fps: float
    cpu: float
    ram: float
    face_visible: bool
    phone_detected: bool
    gaze_yaw: float
    gaze_pitch: float
    active_app: str

class EventBus:
    """
    In-process Pub/Sub Event Bus for routing Domain Events
    to various subscribers asynchronously (if they implement async)
    or synchronously.
    """
    def __init__(self):
        self._subscribers: Dict[str, List[Callable]] = {}
        self._lock = threading.Lock()

    def subscribe(self, topic: str, handler: Callable) -> None:
        with self._lock:
            if topic not in self._subscribers:
                self._subscribers[topic] = []
            if handler not in self._subscribers[topic]:
                self._subscribers[topic].append(handler)

    def unsubscribe(self, topic: str, handler: Callable) -> None:
        with self._lock:
            if topic in self._subscribers and handler in self._subscribers[topic]:
                self._subscribers[topic].remove(handler)

    def publish(self, topic: str, event: Any) -> None:
        with self._lock:
            handlers = self._subscribers.get(topic, []).copy()
        
        # Publish to all handlers.
        for handler in handlers:
            try:
                handler(event)
            except Exception as e:
                print(f"[EventBus] Error in handler {handler} for topic {topic}: {e}")

    def shutdown(self) -> None:
        with self._lock:
            self._subscribers.clear()
