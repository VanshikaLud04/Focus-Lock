"""
database.py -- SQLite session store

Every detection event -- timestamp, state, confidence, false positive flag --
gets written to a local SQLite database.

Schema:
  sessions(id, start_ts, end_ts, focused_sec, distracted_sec,
           break_sec, idle_sec, focus_score, note)

  events(id, session_id, ts, state, confidence, yaw_deg, pitch_deg,
         app_bundle, sample_interval_ms, false_positive)
"""

from __future__ import annotations
import sqlite3
import os
import time
from pathlib import Path
from typing import Optional


_CREATE_SESSIONS = """
CREATE TABLE IF NOT EXISTS sessions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    start_ts        REAL    NOT NULL,
    end_ts          REAL,
    focused_sec     REAL    DEFAULT 0,
    distracted_sec  REAL    DEFAULT 0,
    break_sec       REAL    DEFAULT 0,
    idle_sec        REAL    DEFAULT 0,
    focus_score     REAL,
    note            TEXT
);
"""

_CREATE_EVENTS = """
CREATE TABLE IF NOT EXISTS events (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id       INTEGER NOT NULL REFERENCES sessions(id),
    ts               REAL    NOT NULL,
    state            TEXT    NOT NULL,
    confidence       REAL,
    yaw_deg          REAL,
    pitch_deg        REAL,
    app_bundle       TEXT,
    sample_interval_ms REAL,
    false_positive   INTEGER DEFAULT 0
);
"""

_CREATE_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_events_session ON events(session_id);",
    "CREATE INDEX IF NOT EXISTS idx_events_ts      ON events(ts);",
]


class SessionDB:
    """
    SQLite-backed session and event store.

    Usage
    -----
    db         = SessionDB(cfg["database"])
    session_id = db.start_session()
    db.log_event(session_id, state="FOCUSED", ...)
    db.flag_last_event_as_fp(session_id)
    db.end_session(session_id, stats)
    """

    def __init__(self, cfg: dict) -> None:
        raw_path   = cfg.get("path", "~/.focuslock/sessions.db")
        db_path    = Path(os.path.expanduser(raw_path))
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._path = str(db_path)
        self._init_schema()
        print(f"Database: {self._path}")

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._path, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL;")
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(_CREATE_SESSIONS)
            conn.execute(_CREATE_EVENTS)
            for idx in _CREATE_INDEXES:
                conn.execute(idx)

    def start_session(self, note: str = "") -> int:
        """Insert a new session row and return its ID."""
        with self._connect() as conn:
            cur = conn.execute(
                "INSERT INTO sessions(start_ts, note) VALUES (?, ?)",
                (time.time(), note)
            )
            return cur.lastrowid

    def end_session(self, session_id: int, stats) -> None:
        """Close the session row with final accumulated stats."""
        with self._connect() as conn:
            conn.execute(
                """UPDATE sessions SET
                    end_ts        = ?,
                    focused_sec   = ?,
                    distracted_sec= ?,
                    break_sec     = ?,
                    idle_sec      = ?
                WHERE id = ?""",
                (
                    time.time(),
                    stats.focused_sec,
                    stats.distracted_sec,
                    stats.break_sec,
                    stats.idle_sec,
                    session_id,
                )
            )

    def update_focus_score(self, session_id: int, score: float) -> None:
        """Write the computed Focus Score (0-100) to the session row."""
        with self._connect() as conn:
            conn.execute(
                "UPDATE sessions SET focus_score = ? WHERE id = ?",
                (score, session_id)
            )

    def log_event(
        self,
        session_id:  int,
        state:       str,
        confidence:  float  = 0.0,
        yaw_deg:     float  = 0.0,
        pitch_deg:   float  = 0.0,
        app_bundle:  str    = "unknown",
        interval_ms: float  = 0.0,
    ) -> None:
        """Insert one detection event row."""
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO events
                   (session_id, ts, state, confidence, yaw_deg, pitch_deg,
                    app_bundle, sample_interval_ms)
                   VALUES (?,?,?,?,?,?,?,?)""",
                (session_id, time.time(), state, confidence,
                 yaw_deg, pitch_deg, app_bundle, interval_ms)
            )

    def flag_last_event_as_fp(self, session_id: int) -> None:
        """Mark the most recent event in this session as a false positive."""
        with self._connect() as conn:
            conn.execute(
                """UPDATE events SET false_positive = 1
                   WHERE id = (
                       SELECT id FROM events
                       WHERE session_id = ?
                       ORDER BY ts DESC LIMIT 1
                   )""",
                (session_id,)
            )

    def get_sessions(self, since_ts: float) -> list[dict]:
        """Return all sessions since a Unix timestamp."""
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM sessions WHERE start_ts >= ? ORDER BY start_ts",
                (since_ts,)
            ).fetchall()
        return [dict(r) for r in rows]

    def get_events(self, session_id: int) -> list[dict]:
        """Return all events for a given session."""
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM events WHERE session_id = ? ORDER BY ts",
                (session_id,)
            ).fetchall()
        return [dict(r) for r in rows]
