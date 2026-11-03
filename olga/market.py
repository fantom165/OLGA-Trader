"""Market-level distribution days for the major indexes.

A distribution day is a session where the index closes at least 0.2% lower on
heavier volume than the day before — big money leaving while the tape still
looks orderly. One means nothing. Five or more inside a rolling 25 sessions is
the warning the article describes: institutions bailing out of the market.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from . import config, indicators

PRESSURE_BANDS: list[tuple[int, str, str]] = [
    (6, "critical", "Heavy institutional selling — the market is under pressure."),
    (5, "serious", "Distribution days are stacking up. Treat rallies with suspicion."),
    (3, "warning", "Selling is picking up. Worth watching closely."),
    (0, "good", "Distribution is light. Institutions are not rushing for the exits."),
]


@dataclass
class MarketHealth:
    """One index's distribution-day picture."""

    ticker: str
    name: str
    frame: pd.DataFrame
    count: int
    up_volume_days: int
    window: int
    level: str
    verdict: str
    last_close: float
    last_change_pct: float

    @property
    def distribution_dates(self) -> list[pd.Timestamp]:
        return self.frame.index[self.frame["IsDistributionDay"]].tolist()


def _pressure(count: int) -> tuple[str, str]:
    for threshold, level, verdict in PRESSURE_BANDS:
        if count >= threshold:
            return level, verdict
    return "good", PRESSURE_BANDS[-1][2]


def analyze_index(
    ticker: str,
    daily: pd.DataFrame,
    *,
    pct_threshold: float = -0.2,
    window: int = 25,
) -> MarketHealth:
    """Flag distribution days and count them over a rolling window."""
    frame = daily.copy()
    frame["PctChange"] = frame["Close"].pct_change() * 100.0
    frame["PrevVolume"] = frame["Volume"].shift(1)
    heavier = frame["Volume"] > frame["PrevVolume"]

    frame["IsDistributionDay"] = (frame["PctChange"] <= pct_threshold) & heavier
    # The mirror image: a gain on rising volume, institutions stepping in.
    frame["IsAccumulationDay"] = (frame["PctChange"] >= abs(pct_threshold)) & heavier

    frame["DistributionCount"] = (
        frame["IsDistributionDay"].rolling(window, min_periods=1).sum()
    )
    frame["AccumulationCount"] = (
        frame["IsAccumulationDay"].rolling(window, min_periods=1).sum()
    )
    frame["RelVolume"] = indicators.relative_volume(daily, window=50)

    recent = frame.tail(window)
    count = int(recent["IsDistributionDay"].sum())
    level, verdict = _pressure(count)

    last = frame.iloc[-1] if not frame.empty else None
    return MarketHealth(
        ticker=ticker,
        name=config.display_name(ticker),
        frame=frame,
        count=count,
        up_volume_days=int(recent["IsAccumulationDay"].sum()),
        window=window,
        level=level,
        verdict=verdict,
        last_close=float(last["Close"]) if last is not None else float("nan"),
        last_change_pct=float(last["PctChange"]) if last is not None else float("nan"),
    )


def analyze_indexes(
    bars: dict[str, pd.DataFrame],
    tickers: list[str],
    *,
    pct_threshold: float = -0.2,
    window: int = 25,
) -> list[MarketHealth]:
    return [
        analyze_index(t, bars[t], pct_threshold=pct_threshold, window=window)
        for t in tickers
        if t in bars and not bars[t].empty
    ]
