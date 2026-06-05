"""
focus_score.py -- focus score algorithm (0-100)

Computes a score at the end of each session based on:
  - focused fraction of total session time
  - recovery speed after distractions
  - false positive rate

Formula:
  base     = focused_sec / total_sec
  recovery = mean(1 / recovery_times_sec)  (capped at 1)
  fp_ratio = false_positives / total_events

  FocusScore = 100 * base
             * (1 + recovery_weight * recovery)
             * (1 - fp_penalty_weight * fp_ratio)
             clamped to [0, 100]
"""

from __future__ import annotations
import math
from dataclasses import dataclass


@dataclass
class ScoreBreakdown:
    """Breakdown of how the Focus Score was computed."""
    raw_focus_pct:    float   # focused_sec / total_sec * 100
    recovery_factor:  float   # boost from quick distraction recovery
    fp_penalty:       float   # penalty from false positives
    final_score:      float   # the 0-100 score
    grade:            str     # A / B / C / D / F


class FocusScorer:
    """
    Computes the Focus Score for a completed session.

    Usage
    -----
    scorer = FocusScorer(cfg["scoring"])
    events = db.get_events(session_id)
    score  = scorer.compute(stats, events)
    """

    _MAX_RECOVERY_SEC = 60.0

    def __init__(self, cfg: dict) -> None:
        self._recovery_w = cfg.get("recovery_weight",    0.3)
        self._fp_w       = cfg.get("fp_penalty_weight",  0.1)

    def compute(self, stats, events: list[dict]) -> ScoreBreakdown:
        """
        Compute the Focus Score.

        Parameters
        ----------
        stats  : SessionStats  -- from FocusFSM.session_stats()
        events : list[dict]    -- from SessionDB.get_events()

        Returns
        -------
        ScoreBreakdown
        """
        total = stats.total_sec
        if total <= 0:
            return ScoreBreakdown(0, 0, 0, 0, "--")

        base     = stats.focused_sec / total
        recovery = self._compute_recovery(events)
        fp_ratio = self._compute_fp_ratio(events)

        score = 100 * base
        score *= (1 + self._recovery_w * recovery)
        score *= (1 - self._fp_w * fp_ratio)
        score  = max(0.0, min(100.0, score))

        return ScoreBreakdown(
            raw_focus_pct   = round(base * 100, 1),
            recovery_factor = round(recovery, 3),
            fp_penalty      = round(fp_ratio, 3),
            final_score     = round(score, 1),
            grade           = self._grade(score),
        )

    def _compute_recovery(self, events: list[dict]) -> float:
        """
        How fast the user returned to FOCUSED after each DISTRACTED event.
        Returns [0, 1]: 1 = always recovered instantly, 0 = never recovered.
        """
        if len(events) < 2:
            return 0.5

        recoveries = []
        for i in range(1, len(events)):
            if events[i - 1]["state"] == "DISTRACTED" and events[i]["state"] == "FOCUSED":
                dt = events[i]["ts"] - events[i - 1]["ts"]
                normalised = 1.0 - min(dt / self._MAX_RECOVERY_SEC, 1.0)
                recoveries.append(normalised)

        return sum(recoveries) / len(recoveries) if recoveries else 0.5

    @staticmethod
    def _compute_fp_ratio(events: list[dict]) -> float:
        """Fraction of events flagged as false positives."""
        if not events:
            return 0.0
        fp_count = sum(1 for e in events if e.get("false_positive"))
        return fp_count / len(events)

    @staticmethod
    def _grade(score: float) -> str:
        if score >= 90: return "A"
        if score >= 80: return "B"
        if score >= 65: return "C"
        if score >= 50: return "D"
        return "F"
