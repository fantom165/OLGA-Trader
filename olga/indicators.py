"""The footprint indicators: price-range position, volume weight, and the
up/down volume ratio that together separate accumulation from distribution.

Every function takes a daily or weekly OHLCV frame indexed by date with float
columns ``Open, High, Low, Close, Volume`` and returns aligned pandas objects.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

ACCUMULATION = "accumulation"
DISTRIBUTION = "distribution"
NEUTRAL = "neutral"


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Element-wise division where a zero or missing denominator yields NaN."""
    denom = denominator.astype("float64").where(lambda s: s.ne(0.0))
    return numerator.astype("float64") / denom


def to_weekly(daily: pd.DataFrame) -> pd.DataFrame:
    """Roll daily bars up into weekly bars labelled by their Friday.

    Weekly bars are the timeframe the article's accumulation and distribution
    patterns are read on: a week's high/low is the "trading range" and its close
    is where the week finished inside that range.
    """
    if daily.empty:
        return daily.copy()
    weekly = daily.resample("W-FRI").agg(
        {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
    )
    weekly = weekly.dropna(subset=["Close"])
    weekly["Sessions"] = (
        daily["Close"].resample("W-FRI").count().reindex(weekly.index).fillna(0)
    )
    return weekly


def close_range_position(bars: pd.DataFrame) -> pd.Series:
    """Where the close sits inside the bar's range: 0.0 = the low, 1.0 = the high.

    This is the single most direct reading of the article's core clue. A bar
    whose high equals its low carries no range information, so it reads as the
    midpoint rather than as an extreme.
    """
    span = bars["High"] - bars["Low"]
    position = _safe_divide(bars["Close"] - bars["Low"], span)
    return position.fillna(0.5).clip(0.0, 1.0)


def up_down_volume_ratio(daily: pd.DataFrame, window: int = 50) -> pd.Series:
    """Volume traded on up days divided by volume on down days over ``window``.

    Above 1.0 means more shares changed hands while the stock rose; below 1.0
    means the heavier trade happened on declines. 1.0 is churn.
    """
    if daily.empty:
        return pd.Series(dtype="float64", index=daily.index)
    change = daily["Close"].diff()
    volume = daily["Volume"].astype("float64")
    min_periods = max(5, window // 2)
    up = volume.where(change > 0, 0.0).rolling(window, min_periods=min_periods).sum()
    down = volume.where(change < 0, 0.0).rolling(window, min_periods=min_periods).sum()
    return _safe_divide(up, down)


def relative_volume(bars: pd.DataFrame, window: int = 50) -> pd.Series:
    """Volume as a multiple of its trailing average, excluding the current bar.

    Excluding the current bar matters: a genuine institutional volume spike
    should be measured against what normal looked like *before* it, otherwise a
    big bar partly sets its own baseline.
    """
    if bars.empty:
        return pd.Series(dtype="float64", index=bars.index)
    volume = bars["Volume"].astype("float64")
    baseline = volume.rolling(window, min_periods=max(3, window // 3)).mean().shift(1)
    return _safe_divide(volume, baseline)


def ad_line(bars: pd.DataFrame) -> pd.Series:
    """Chaikin Accumulation/Distribution line.

    Each bar contributes its volume weighted by where the close sat in the
    range, so a rising line is the cumulative footprint of buyers taking
    control of the close and a falling line is the footprint of sellers.
    """
    if bars.empty:
        return pd.Series(dtype="float64", index=bars.index)
    span = bars["High"] - bars["Low"]
    money_flow_multiplier = _safe_divide(
        (bars["Close"] - bars["Low"]) - (bars["High"] - bars["Close"]), span
    ).fillna(0.0)
    return (money_flow_multiplier * bars["Volume"].astype("float64")).cumsum()


def obv(bars: pd.DataFrame) -> pd.Series:
    """On-balance volume: full volume added on up closes, subtracted on down."""
    if bars.empty:
        return pd.Series(dtype="float64", index=bars.index)
    direction = np.sign(bars["Close"].diff().fillna(0.0))
    return (direction * bars["Volume"].astype("float64")).cumsum()


def weekly_signals(
    weekly: pd.DataFrame,
    *,
    vol_window: int = 10,
    vol_multiple: float = 1.25,
    accumulation_position: float = 0.70,
    distribution_position: float = 0.30,
) -> pd.DataFrame:
    """Label each week accumulation, distribution, or neutral.

    A week is accumulation when it closes in the top of its range on volume
    well above normal, and distribution when it closes in the bottom of its
    range on that same heavy volume. Heavy volume without a decisive close, or
    a decisive close on quiet volume, is neutral — the article's point that a
    March pullback on *light* volume means institutions paused rather than sold.
    """
    out = weekly.copy()
    if out.empty:
        for column in ("RangePos", "AvgVolume", "VolRatio", "PctChange", "Signal"):
            out[column] = pd.Series(dtype="float64" if column != "Signal" else "object")
        return out

    out["RangePos"] = close_range_position(weekly)
    volume = weekly["Volume"].astype("float64")
    out["AvgVolume"] = (
        volume.rolling(vol_window, min_periods=max(3, vol_window // 3)).mean().shift(1)
    )
    out["VolRatio"] = _safe_divide(volume, out["AvgVolume"])
    out["PctChange"] = weekly["Close"].pct_change() * 100.0

    heavy = out["VolRatio"] >= vol_multiple
    out["Signal"] = NEUTRAL
    out.loc[heavy & (out["RangePos"] >= accumulation_position), "Signal"] = ACCUMULATION
    out.loc[heavy & (out["RangePos"] <= distribution_position), "Signal"] = DISTRIBUTION
    return out


@dataclass
class SignalSummary:
    """The read on a stock's recent weeks, in the terms the article uses."""

    streak_signal: str
    streak_weeks: int
    accumulation_weeks: int
    distribution_weeks: int
    lookback_weeks: int

    @property
    def net_weeks(self) -> int:
        return self.accumulation_weeks - self.distribution_weeks

    @property
    def headline(self) -> str:
        if self.streak_weeks >= 2:
            return f"{self.streak_weeks} consecutive {self.streak_signal} weeks"
        if self.streak_weeks == 1:
            return f"latest week is {self.streak_signal}"
        if self.net_weeks > 0:
            return f"net accumulation over {self.lookback_weeks} weeks"
        if self.net_weeks < 0:
            return f"net distribution over {self.lookback_weeks} weeks"
        return "churning — neither side in control"


def summarize_signals(signals: pd.Series, lookback_weeks: int = 13) -> SignalSummary:
    """Count the streak of consecutive like weeks and the balance over a window."""
    labels = [str(value) for value in signals.dropna().tolist()]
    window = labels[-lookback_weeks:] if lookback_weeks > 0 else labels

    streak_signal, streak_weeks = NEUTRAL, 0
    if labels and labels[-1] != NEUTRAL:
        streak_signal = labels[-1]
        for label in reversed(labels):
            if label != streak_signal:
                break
            streak_weeks += 1

    return SignalSummary(
        streak_signal=streak_signal,
        streak_weeks=streak_weeks,
        accumulation_weeks=window.count(ACCUMULATION),
        distribution_weeks=window.count(DISTRIBUTION),
        lookback_weeks=len(window),
    )


def trend_slope(series: pd.Series, window: int = 50) -> float:
    """Least-squares slope of the last ``window`` points, per point.

    Used to turn the A/D line into the "buying momentum or selling pressure"
    trend line the article describes.
    """
    tail = series.dropna().tail(window)
    if len(tail) < 3:
        return float("nan")
    x = np.arange(len(tail), dtype="float64")
    y = tail.to_numpy(dtype="float64")
    slope, _ = np.polyfit(x, y, 1)
    return float(slope)


def swing_levels(
    daily: pd.DataFrame,
    *,
    span: int = 5,
    lookback: int = 260,
    tolerance: float = 0.025,
    min_touches: int = 2,
    max_levels: int = 5,
) -> pd.DataFrame:
    """Find price levels the stock has repeatedly turned at.

    These are the floors of support institutions defend and the ceilings of
    resistance they sell into. A pivot is a bar whose high (or low) is the
    extreme of a ``2*span+1`` window; nearby pivots are clustered into one
    level, and a level needs ``min_touches`` pivots to count.
    """
    empty = pd.DataFrame(columns=["Level", "Touches", "Role"])
    if daily.empty or len(daily) < (2 * span + 2):
        return empty

    bars = daily.tail(lookback)
    width = 2 * span + 1
    rolling_max = bars["High"].rolling(width, center=True).max()
    rolling_min = bars["Low"].rolling(width, center=True).min()
    pivots = pd.concat(
        [
            bars.loc[bars["High"].eq(rolling_max), "High"],
            bars.loc[bars["Low"].eq(rolling_min), "Low"],
        ]
    ).dropna()
    if pivots.empty:
        return empty

    clusters: list[list[float]] = []
    for price in sorted(pivots.tolist()):
        if clusters:
            centre = float(np.mean(clusters[-1]))
            if centre > 0 and abs(price - centre) / centre <= tolerance:
                clusters[-1].append(price)
                continue
        clusters.append([price])

    last_close = float(bars["Close"].iloc[-1])
    rows = [
        {
            "Level": float(np.mean(group)),
            "Touches": len(group),
            "Role": "resistance" if float(np.mean(group)) > last_close else "support",
        }
        for group in clusters
        if len(group) >= min_touches
    ]
    if not rows:
        return empty

    levels = pd.DataFrame(rows)
    levels["Distance"] = (levels["Level"] - last_close).abs()
    levels = levels.sort_values(
        ["Touches", "Distance"], ascending=[False, True]
    ).head(max_levels)
    return levels.sort_values("Level").drop(columns="Distance").reset_index(drop=True)
