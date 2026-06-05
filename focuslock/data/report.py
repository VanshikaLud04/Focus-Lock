"""
report.py -- CLI session reporter

Reads the SQLite database and prints a formatted session summary.

Commands:
    python main.py report           # today
    python main.py report --week    # last 7 days
"""

from __future__ import annotations
import os
import time
from datetime import datetime, timedelta

from rich.console import Console
from rich.table import Table
from rich import box


console = Console()


def _load_db(db_cfg: dict):
    from focuslock.data.database import SessionDB
    return SessionDB(db_cfg)


def print_report(db_cfg: dict, period: str = "today") -> None:
    """
    Print a formatted session table to the terminal.

    Parameters
    ----------
    db_cfg : dict   -- cfg["database"]
    period : str    -- "today" | "week"
    """
    db = _load_db(db_cfg)

    if period == "week":
        since = time.time() - 7 * 24 * 3600
        title = "Focus Lock -- Last 7 Days"
    else:
        midnight = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        since    = midnight.timestamp()
        title    = f"Focus Lock -- {datetime.now().strftime('%A, %d %b %Y')}"

    sessions = db.get_sessions(since_ts=since)

    if not sessions:
        console.print(f"\n[dim]{title}[/dim]")
        console.print("[yellow]No sessions found for this period.[/yellow]\n")
        return

    table = Table(
        title       = title,
        box         = box.ROUNDED,
        show_footer = True,
        header_style= "bold cyan",
    )

    table.add_column("Session",        style="white",    no_wrap=True)
    table.add_column("Duration",       style="cyan",     justify="right")
    table.add_column("Focused",        style="green",    justify="right")
    table.add_column("Distracted",     style="red",      justify="right")
    table.add_column("Focus %",        style="bold",     justify="right")
    table.add_column("Score",          style="yellow",   justify="right")
    table.add_column("Top App",        style="dim")

    total_focused = 0.0
    total_dur     = 0.0

    for s in sessions:
        start     = datetime.fromtimestamp(s["start_ts"])
        end_ts    = s["end_ts"] or time.time()
        end       = datetime.fromtimestamp(end_ts)
        duration  = end_ts - s["start_ts"]
        focus_pct = (s["focused_sec"] / duration * 100) if duration > 0 else 0.0
        score_str = f"{s['focus_score']:.0f}/100" if s["focus_score"] is not None else "--"
        events    = db.get_events(s["id"])
        top_app   = _top_app(events)

        table.add_row(
            f"{start.strftime('%H:%M')}-{end.strftime('%H:%M')}",
            _fmt_sec(duration),
            _fmt_sec(s["focused_sec"]),
            _fmt_sec(s["distracted_sec"]),
            f"{focus_pct:.0f}%",
            score_str,
            top_app,
        )

        total_focused += s["focused_sec"]
        total_dur     += duration

    overall_pct = (total_focused / total_dur * 100) if total_dur > 0 else 0.0
    console.print()
    console.print(table)
    console.print(
        f"\n[bold]Overall focus rate:[/bold] [green]{overall_pct:.1f}%[/green] "
        f"across {_fmt_sec(total_dur)} of tracked work.\n"
    )


def _fmt_sec(seconds: float) -> str:
    """Format seconds as Xh Ym or Mm Xs."""
    seconds = int(seconds)
    h, rem  = divmod(seconds, 3600)
    m, s    = divmod(rem, 60)
    if h > 0:
        return f"{h}h {m}m"
    if m > 0:
        return f"{m}m {s}s"
    return f"{s}s"


def _top_app(events: list[dict]) -> str:
    """Return the most frequent app bundle in the event list."""
    if not events:
        return "--"
    from collections import Counter
    counts = Counter(e["app_bundle"] for e in events if e["app_bundle"])
    if not counts:
        return "--"
    bundle = counts.most_common(1)[0][0]
    _NAMES = {
        "com.microsoft.VSCode":  "VSCode",
        "com.google.Chrome":     "Chrome",
        "com.apple.Safari":      "Safari",
        "com.apple.Xcode":       "Xcode",
        "unknown":               "--",
    }
    return _NAMES.get(bundle, bundle.split(".")[-1])
