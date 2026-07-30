"""
analytics_engine.py -- Edge Analytics for Focus Lock
"""

import sqlite3
import time
from typing import List, Dict, Any
from focuslock.data.database import SessionDB

class AnalyticsEngine:
    """
    Computes derived productivity insights from the event log.
    """
    def __init__(self, db: SessionDB):
        self.db = db

    def build_timeline(self, session_id: int) -> List[Dict[str, Any]]:
        """
        Collapses contiguous events into optimal render blocks for the dashboard.
        """
        events = []
        with self.db._connect() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT ts, state FROM attention_events WHERE session_id = ? ORDER BY ts ASC",
                (session_id,)
            ).fetchall()

        if not rows:
            return []

        timeline = []
        current_block = {
            "state": rows[0]["state"],
            "start_ts": rows[0]["ts"],
            "end_ts": rows[0]["ts"]
        }

        for row in rows[1:]:
            # If state changes, push the block and start a new one
            if row["state"] != current_block["state"]:
                current_block["end_ts"] = row["ts"]
                # Only append if duration is > 1s (noise filter)
                if current_block["end_ts"] - current_block["start_ts"] >= 1.0:
                    timeline.append(current_block)
                current_block = {
                    "state": row["state"],
                    "start_ts": row["ts"],
                    "end_ts": row["ts"]
                }
            else:
                current_block["end_ts"] = row["ts"]

        # Append final block
        if current_block["end_ts"] - current_block["start_ts"] >= 1.0 or not timeline:
            timeline.append(current_block)

        return timeline

    def build_heatmap(self, days: int = 30) -> Dict[int, float]:
        """
        Event -> Hour Bucket -> Duration Sum -> Normalize
        Returns a dict mapping hour (0-23) to focus ratio (0.0-1.0)
        """
        cutoff = time.time() - (days * 24 * 3600)
        
        with self.db._connect() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT ts, state FROM attention_events WHERE ts >= ? ORDER BY ts ASC",
                (cutoff,)
            ).fetchall()

        # Simple algorithm: iterate over events, assign to hour bucket based on local time
        from datetime import datetime
        hour_totals = {h: 0.0 for h in range(24)}
        hour_focus = {h: 0.0 for h in range(24)}

        # To calculate durations, we need to look at consecutive events
        for i in range(len(rows) - 1):
            curr_row = rows[i]
            next_row = rows[i+1]
            duration = next_row["ts"] - curr_row["ts"]
            
            # If gap > 5 minutes, assume session ended
            if duration > 300:
                continue

            dt = datetime.fromtimestamp(curr_row["ts"])
            hour = dt.hour
            hour_totals[hour] += duration
            if curr_row["state"] == "FOCUSED":
                hour_focus[hour] += duration

        heatmap = {}
        for h in range(24):
            if hour_totals[h] > 0:
                heatmap[h] = round(hour_focus[h] / hour_totals[h], 2)
            else:
                heatmap[h] = 0.0
                
        return heatmap

    def get_longest_streak(self, session_id: int = None) -> float:
        """
        Calculate the longest continuous run of FOCUSED (uninterrupted by DISTRACTED).
        BREAK states don't add to the duration but don't break the streak.
        If session_id is given, scopes to that session. Otherwise today.
        """
        query = "SELECT ts, state FROM attention_events "
        params = ()
        if session_id:
            query += "WHERE session_id = ? ORDER BY ts ASC"
            params = (session_id,)
        else:
            # Today's events
            from datetime import datetime
            midnight = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
            query += "WHERE ts >= ? ORDER BY ts ASC"
            params = (midnight,)

        with self.db._connect() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(query, params).fetchall()

        max_streak = 0.0
        current_streak = 0.0
        last_ts = None
        
        for row in rows:
            if last_ts is None:
                last_ts = row["ts"]
                if row["state"] == "FOCUSED":
                    current_streak += 1.0 # seed it
                continue
                
            duration = row["ts"] - last_ts
            last_ts = row["ts"]
            
            if duration > 300: # large gap resets
                current_streak = 0.0
                continue
                
            if row["state"] == "FOCUSED":
                current_streak += duration
                max_streak = max(max_streak, current_streak)
            elif row["state"] == "DISTRACTED":
                current_streak = 0.0
            # If BREAK or IDLE, do nothing (doesn't add to streak, doesn't reset it)
            
        return round(max_streak, 1)

    def generate_daily_summary(self) -> None:
        """
        Materializes today's events into the daily_summary table.
        """
        from datetime import datetime
        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        midnight = now.replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
        
        with self.db._connect() as conn:
            conn.row_factory = sqlite3.Row
            # Sum focus, distracted, phone pickups
            events = conn.execute(
                "SELECT state, trigger, fps, cpu, ram, ts FROM attention_events WHERE ts >= ? ORDER BY ts ASC",
                (midnight,)
            ).fetchall()
            
            if not events:
                return

            focus_sec = 0.0
            distracted_sec = 0.0
            phone_pickups = 0
            
            sum_fps = 0.0
            sum_cpu = 0.0
            sum_ram = 0.0
            count = len(events)
            
            last_ts = None
            
            for row in events:
                sum_fps += row["fps"] or 0
                sum_cpu += row["cpu"] or 0
                sum_ram += row["ram"] or 0
                
                if row["trigger"] == "phone":
                    phone_pickups += 1
                    
                if last_ts:
                    duration = row["ts"] - last_ts
                    if duration < 300:
                        if row["state"] == "FOCUSED":
                            focus_sec += duration
                        elif row["state"] == "DISTRACTED":
                            distracted_sec += duration
                            
                last_ts = row["ts"]
                
            longest_streak = self.get_longest_streak()
            
            conn.execute(
                """INSERT OR REPLACE INTO daily_summary 
                   (date, total_focus_seconds, total_distracted_seconds, phone_pickups, longest_streak_seconds, avg_fps, avg_cpu, avg_ram)
                   VALUES (?,?,?,?,?,?,?,?)""",
                (date_str, focus_sec, distracted_sec, phone_pickups, longest_streak,
                 sum_fps/count, sum_cpu/count, sum_ram/count)
            )
            conn.commit()

    def get_trends(self) -> Dict[str, Any]:
        """
        Least-squares linear regression over the last 14 days of focus_pct.
        """
        with self.db._connect() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT date, total_focus_seconds, total_distracted_seconds FROM daily_summary ORDER BY date DESC LIMIT 14"
            ).fetchall()
            
        if len(rows) < 2:
            return {"slope": 0.0, "trend": "flat"}
            
        # Reverse to chronological
        rows = list(reversed(rows))
        
        x = list(range(len(rows)))
        y = []
        for r in rows:
            total = r["total_focus_seconds"] + r["total_distracted_seconds"]
            pct = (r["total_focus_seconds"] / total * 100) if total > 0 else 0
            y.append(pct)
            
        # Linear regression slope:
        n = len(x)
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(x[i]*y[i] for i in range(n))
        sum_xx = sum(x[i]*x[i] for i in range(n))
        
        denominator = (n * sum_xx - sum_x * sum_x)
        slope = 0.0
        if denominator != 0:
            slope = (n * sum_xy - sum_x * sum_y) / denominator
            
        # Dead-band of ±2%/day
        if slope > 2.0:
            trend = "improving"
        elif slope < -2.0:
            trend = "declining"
        else:
            trend = "flat"
            
        return {"slope": round(slope, 2), "trend": trend}
