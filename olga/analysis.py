"""Assemble the per-stock analysis the dashboard renders.

This is the seam between the indicator maths and the UI: everything the app
needs about one ticker is computed once, here, and handed over as a single
object.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from . import indicators, rating
from .config import Settings
from .indicators import SignalSummary
from .rating import Rating


@dataclass
class StockAnalysis:
    """Everything the dashboard knows about one stock's institutional footprint."""

    ticker: str
    daily: pd.DataFrame
    weekly: pd.DataFrame
    ud_ratio: pd.Series
    ad_line: pd.Series
    obv: pd.Series
    levels: pd.DataFrame
    summary: SignalSummary
    rating: Rating

    @property
    def last_close(self) -> float:
        return float(self.daily["Close"].iloc[-1])

    @property
    def last_date(self) -> pd.Timestamp:
        return self.daily.index[-1]

    @property
    def ud_latest(self) -> float:
        series = self.ud_ratio.dropna()
        return float(series.iloc[-1]) if not series.empty else float("nan")

    @property
    def daily_rel_volume(self) -> float:
        series = indicators.relative_volume(self.daily, window=50).dropna()
        return float(series.iloc[-1]) if not series.empty else float("nan")

    @property
    def change_pct(self) -> float:
        closes = self.daily["Close"]
        if len(closes) < 2:
            return float("nan")
        return float(closes.iloc[-1] / closes.iloc[-2] - 1.0) * 100.0

    @property
    def latest_week(self) -> pd.Series | None:
        return None if self.weekly.empty else self.weekly.iloc[-1]

    @property
    def week_in_progress(self) -> bool:
        """True when the newest weekly bar is still being formed.

        Weekly bars are labelled by their Friday, so a bar whose Friday has not
        traded yet is partial and its signal is provisional.
        """
        if self.weekly.empty:
            return False
        sessions = self.weekly["Sessions"].iloc[-1]
        return bool(pd.notna(sessions) and sessions < 5)


def analyze(ticker: str, daily: pd.DataFrame, settings: Settings) -> StockAnalysis:
    weekly = indicators.weekly_signals(
        indicators.to_weekly(daily),
        vol_window=settings.weekly_vol_window,
        vol_multiple=settings.weekly_vol_multiple,
        accumulation_position=settings.accumulation_close_position,
        distribution_position=settings.distribution_close_position,
    )
    summary = indicators.summarize_signals(
        weekly["Signal"] if "Signal" in weekly.columns else pd.Series(dtype="object"),
        lookback_weeks=settings.signal_lookback_weeks,
    )
    return StockAnalysis(
        ticker=ticker,
        daily=daily,
        weekly=weekly,
        ud_ratio=indicators.up_down_volume_ratio(daily, window=settings.ud_window),
        ad_line=indicators.ad_line(daily),
        obv=indicators.obv(daily),
        levels=indicators.swing_levels(daily),
        summary=summary,
        rating=rating.rate(daily, weekly, summary, ud_window=settings.ud_window),
    )


def analyze_all(
    bars: dict[str, pd.DataFrame], settings: Settings
) -> dict[str, StockAnalysis]:
    return {
        ticker: analyze(ticker, frame, settings)
        for ticker, frame in bars.items()
        if not frame.empty
    }


def watchlist_table(analyses: dict[str, StockAnalysis]) -> pd.DataFrame:
    """One row per stock, ranked strongest accumulation first."""
    rows = []
    for item in analyses.values():
        week = item.latest_week
        rows.append(
            {
                "Ticker": item.ticker,
                "Grade": item.rating.grade,
                "Score": item.rating.score,
                "Stance": item.rating.stance,
                "Close": item.last_close,
                "Day %": item.change_pct,
                "U/D Ratio": item.ud_latest,
                "Week Close in Range": (
                    float(week["RangePos"]) if week is not None else float("nan")
                ),
                "Week Vol x Avg": (
                    float(week["VolRatio"]) if week is not None else float("nan")
                ),
                "Acc Weeks": item.summary.accumulation_weeks,
                "Dist Weeks": item.summary.distribution_weeks,
                "Streak": (
                    f"{item.summary.streak_weeks}x {item.summary.streak_signal}"
                    if item.summary.streak_weeks
                    else "—"
                ),
                "Read": item.summary.headline,
            }
        )
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    return table.sort_values("Score", ascending=False).reset_index(drop=True)
