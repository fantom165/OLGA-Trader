"""Condense the footprint indicators into one A-through-E rating.

A means heavy accumulation, C means churn, E means heavy distribution. The
grade is a weighted blend of four independent readings so that no single noisy
input decides it; components that cannot be computed (too little history) drop
out and their weight is shared among the rest.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import indicators
from .indicators import ACCUMULATION, DISTRIBUTION, NEUTRAL, SignalSummary

# Grade thresholds on the 0-100 composite. A neutral 50 lands on a flat C.
GRADE_BANDS: list[tuple[float, str]] = [
    (82.0, "A"),
    (75.0, "A-"),
    (68.0, "B+"),
    (61.0, "B"),
    (56.0, "B-"),
    (52.0, "C+"),
    (47.0, "C"),
    (42.0, "C-"),
    (35.0, "D+"),
    (28.0, "D"),
    (21.0, "D-"),
]
WORST_GRADE = "E"

COMPONENT_WEIGHTS = {
    "ud_ratio": 0.35,
    "weekly_balance": 0.30,
    "ad_trend": 0.20,
    "close_strength": 0.15,
}

COMPONENT_LABELS = {
    "ud_ratio": "Up/down volume (50d)",
    "weekly_balance": "Weekly accumulation balance",
    "ad_trend": "A/D line trend",
    "close_strength": "Volume-weighted close strength",
}


@dataclass
class Rating:
    grade: str
    score: float
    stance: str
    components: dict[str, float] = field(default_factory=dict)
    inputs: dict[str, float] = field(default_factory=dict)

    @property
    def is_accumulation(self) -> bool:
        return self.stance == ACCUMULATION

    @property
    def is_distribution(self) -> bool:
        return self.stance == DISTRIBUTION


def _score_ud_ratio(ratio: float) -> float:
    """1.0 (churn) scores 50; 2.0 scores 90; 0.5 scores 10."""
    if not math.isfinite(ratio) or ratio <= 0:
        return float("nan")
    return float(np.clip(50.0 + 40.0 * math.log2(ratio), 0.0, 100.0))


def _score_weekly_balance(summary: SignalSummary) -> float:
    """Each net accumulation week over the lookback moves the score 10 points."""
    if summary.lookback_weeks == 0:
        return float("nan")
    return float(np.clip(50.0 + 10.0 * summary.net_weeks, 0.0, 100.0))


def _score_ad_trend(daily: pd.DataFrame, window: int = 50) -> float:
    """A/D line slope expressed against one day's average volume.

    A slope of +0.35 average-daily-volumes per day is treated as a fully
    bullish footprint; -0.35 as fully bearish.
    """
    if len(daily) < 20:
        return float("nan")
    slope = indicators.trend_slope(indicators.ad_line(daily), window=window)
    avg_volume = float(daily["Volume"].tail(window).mean())
    if not math.isfinite(slope) or not math.isfinite(avg_volume) or avg_volume <= 0:
        return float("nan")
    normalized = float(np.clip((slope / avg_volume) / 0.35, -1.0, 1.0))
    return float(50.0 + 50.0 * normalized)


def _score_close_strength(weekly: pd.DataFrame, window: int = 10) -> float:
    """Average weekly range position over recent weeks, weighted by volume.

    Weighting by volume is what keeps the reading institutional: a strong close
    on a quiet week counts for less than a strong close on a heavy one.
    """
    if weekly.empty or "RangePos" not in weekly.columns:
        return float("nan")
    tail = weekly.tail(window)
    positions = tail["RangePos"].astype("float64")
    weights = tail["Volume"].astype("float64")
    mask = positions.notna() & weights.notna() & (weights > 0)
    if not mask.any():
        return float("nan")
    weighted = float(
        np.average(positions[mask].to_numpy(), weights=weights[mask].to_numpy())
    )
    return float(np.clip(weighted * 100.0, 0.0, 100.0))


def _to_grade(score: float) -> str:
    for threshold, grade in GRADE_BANDS:
        if score >= threshold:
            return grade
    return WORST_GRADE


def _stance(score: float, summary: SignalSummary) -> str:
    """Reduce the grade to a one-word stance, with recent weeks breaking ties."""
    if score >= 56.0:
        return ACCUMULATION
    if score <= 42.0:
        return DISTRIBUTION
    if summary.streak_weeks >= 2:
        return summary.streak_signal
    return NEUTRAL


def rate(
    daily: pd.DataFrame,
    weekly: pd.DataFrame,
    summary: SignalSummary,
    *,
    ud_window: int = 50,
) -> Rating:
    """Blend the indicators into a grade for one stock."""
    if daily.empty:
        return Rating(grade=WORST_GRADE, score=float("nan"), stance=NEUTRAL)

    ud_series = indicators.up_down_volume_ratio(daily, window=ud_window)
    ud_latest = float(ud_series.iloc[-1]) if not ud_series.empty else float("nan")

    components = {
        "ud_ratio": _score_ud_ratio(ud_latest),
        "weekly_balance": _score_weekly_balance(summary),
        "ad_trend": _score_ad_trend(daily, window=ud_window),
        "close_strength": _score_close_strength(weekly),
    }

    usable = {k: v for k, v in components.items() if math.isfinite(v)}
    if not usable:
        return Rating(grade=WORST_GRADE, score=float("nan"), stance=NEUTRAL)

    total_weight = sum(COMPONENT_WEIGHTS[k] for k in usable)
    score = sum(COMPONENT_WEIGHTS[k] * v for k, v in usable.items()) / total_weight

    return Rating(
        grade=_to_grade(score),
        score=round(score, 1),
        stance=_stance(score, summary),
        components=components,
        inputs={"ud_ratio": ud_latest},
    )
