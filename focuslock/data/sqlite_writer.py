"""
sqlite_writer.py -- SQLite async writer for Event Bus
"""
import sqlite3
import threading
import queue
from focuslock.data.database import SessionDB
from focuslock.analytics.events import AttentionEvent

class SQLiteWriter:
    """
    Subscribes to AttentionEvents and batches writes to SQLite.
    """
    def __init__(self, db: SessionDB):
        self.db = db
        self._queue = queue.Queue()
        self._running = False
        self._thread = None
    
    def start(self):
        if self._running: return
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
            
    def handle_attention_event(self, event: AttentionEvent):
        try:
            self._queue.put_nowait(event)
        except queue.Full:
            pass # In-memory ring buffer full, drop if needed (though infinite by default)
            
    def _run_loop(self):
        with self.db._connect() as conn:
            while self._running or not self._queue.empty():
                try:
                    event = self._queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                try:
                    conn.execute(
                        """INSERT INTO attention_events
                           (session_id, ts, state, trigger, confidence, active_app, fps, cpu, ram, face_visible, phone_detected, gaze_yaw, gaze_pitch)
                           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                        (event.session_id, event.timestamp, event.attention_state, event.trigger,
                         event.confidence, event.active_app, event.fps, event.cpu, event.ram,
                         1 if event.face_visible else 0, 1 if event.phone_detected else 0,
                         event.gaze_yaw, event.gaze_pitch)
                    )
                    conn.commit()
                except Exception as e:
                    print(f"[SQLiteWriter] Write error: {e}")
