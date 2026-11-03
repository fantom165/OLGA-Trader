"""Plotly figures for the dashboard.

Design rules held throughout: one measure per axis (stacked panels rather than
twin y-scales), thin marks on hairline chrome, a legend whenever two or more
series share a panel, selective direct labels instead of a number on every
point, and a hover layer on every chart. Every figure has a table-view twin in
the app so no value is reachable only by hovering.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from . import theme
from .analysis import StockAnalysis
from .indicators import ACCUMULATION, DISTRIBUTION, NEUTRAL
from .market import MarketHealth

SIGNAL_ORDER = [NEUTRAL, ACCUMULATION, DISTRIBUTION]


def _hover_dates(index: pd.DatetimeIndex) -> list[str]:
    return [d.strftime("%b %d, %Y") for d in index]


def weekly_footprint(
    analysis: StockAnalysis, pal: dict[str, str], weeks: int = 78
) -> go.Figure:
    """Weekly candles and volume, both coloured by the week's signal.

    Hollow candles closed down for the week, filled candles closed up, so
    direction survives even though hue is spent on the accumulation reading.
    """
    weekly = analysis.weekly.tail(weeks)
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.68, 0.32],
        vertical_spacing=0.05,
        subplot_titles=("", ""),
    )
    if weekly.empty:
        return _empty(fig, pal, "No weekly history available")

    dates = _hover_dates(weekly.index)

    for signal in SIGNAL_ORDER:
        mask = weekly["Signal"].eq(signal)
        if not mask.any():
            continue
        style = theme.SIGNAL_STYLE[signal]
        color = theme.signal_color(signal, pal)
        # Blank the other weeks rather than dropping them, so all three traces
        # span the same dates and Plotly gives every candle the same width.
        subset = weekly.where(mask, axis=0)
        fig.add_trace(
            go.Candlestick(
                x=weekly.index,
                open=subset["Open"],
                high=subset["High"],
                low=subset["Low"],
                close=subset["Close"],
                name=f"{style['glyph']} {style['label']}",
                legendgroup=signal,
                increasing={
                    "line": {"color": color, "width": 1.4},
                    "fillcolor": color,
                },
                decreasing={
                    "line": {"color": color, "width": 1.4},
                    "fillcolor": pal["surface"],
                },
                whiskerwidth=0.2,
                customdata=np.column_stack(
                    [
                        subset["RangePos"].to_numpy(),
                        subset["VolRatio"].to_numpy(),
                        subset["PctChange"].to_numpy(),
                    ]
                ),
                hovertext=dates,
                hovertemplate=(
                    "<b>%{hovertext}</b><br>"
                    "Range %{low:,.2f} – %{high:,.2f}<br>"
                    "Close %{close:,.2f} "
                    "(%{customdata[0]:.0%} up the range)<br>"
                    "Volume %{customdata[1]:.2f}x average<br>"
                    "Week %{customdata[2]:+.2f}%<extra></extra>"
                ),
            ),
            row=1,
            col=1,
        )

    # Levels the stock has repeatedly turned at — the floors institutions
    # defend and the ceilings they sell into. Every line is drawn, but only the
    # best-attested few are labelled, and only where a label will not land on
    # top of its neighbour. The full list lives in the levels table.
    price_span = float(weekly["High"].max() - weekly["Low"].min())
    min_label_gap = price_span * 0.07
    labelled: list[float] = []
    by_conviction = analysis.levels.sort_values("Touches", ascending=False)
    for _, level in by_conviction.iterrows():
        y = float(level["Level"])
        show_label = len(labelled) < 3 and all(
            abs(y - other) >= min_label_gap for other in labelled
        )
        extra: dict = {}
        if show_label:
            labelled.append(y)
            extra["annotation"] = {
                "text": f"{level['Role']} {y:,.2f}",
                "font": {"color": pal["muted"], "size": 10},
                "xanchor": "left",
                "yanchor": "bottom",
                "x": 0,
            }
        fig.add_hline(
            y=y,
            line={"color": pal["muted"], "width": 1, "dash": "dot"},
            row=1,
            col=1,
            **extra,
        )

    # Volume is plotted in millions of shares: SI-prefixed ticks would read
    # "2.0G" for a heavily traded name and "200M" for a quiet one, which makes
    # two stocks impossible to compare at a glance.
    volume_m = weekly["Volume"] / 1e6
    volume_colors = [theme.signal_color(s, pal) for s in weekly["Signal"]]
    fig.add_trace(
        go.Bar(
            x=weekly.index,
            y=volume_m,
            name="Weekly volume",
            marker={
                "color": volume_colors,
                "line": {"width": 2, "color": pal["surface"]},
            },
            showlegend=False,
            hovertext=dates,
            customdata=np.column_stack(
                [weekly["Volume"].to_numpy(), weekly["VolRatio"].to_numpy()]
            ),
            hovertemplate=(
                "<b>%{hovertext}</b><br>Volume %{customdata[0]:,.0f}"
                "<br>%{customdata[1]:.2f}x average<extra></extra>"
            ),
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=weekly.index,
            y=weekly["AvgVolume"] / 1e6,
            name="Average volume",
            mode="lines",
            line={"color": pal["muted"], "width": 2},
            customdata=weekly["AvgVolume"].to_numpy(),
            hovertemplate="Average %{customdata:,.0f}<extra></extra>",
        ),
        row=2,
        col=1,
    )

    fig.update_layout(
        **theme.layout(
            pal,
            title={
                "text": (
                    f"{analysis.ticker} — weekly range, close position and volume"
                ),
                "font": {"size": 15, "color": pal["text_primary"]},
                "x": 0.005,
                "xanchor": "left",
            },
            height=560,
            bargap=0.35,
            margin={"l": 8, "r": 8, "t": 70, "b": 8},
        ),
        xaxis_rangeslider_visible=False,
    )
    fig.update_xaxes(**theme.axis(pal, grid=False), row=1, col=1)
    fig.update_xaxes(**theme.axis(pal, grid=False), row=2, col=1)
    fig.update_yaxes(
        **theme.axis(pal, title={"text": "Price", "font": {"color": pal["muted"]}}),
        row=1,
        col=1,
    )
    fig.update_yaxes(
        **theme.axis(
            pal,
            title={"text": "Volume (millions)", "font": {"color": pal["muted"]}},
        ),
        row=2,
        col=1,
    )
    return fig


def ud_ratio_chart(
    analysis: StockAnalysis, pal: dict[str, str], days: int = 250
) -> go.Figure:
    """The up/down volume ratio through time, against the 1.0 churn line."""
    series = analysis.ud_ratio.dropna().tail(days)
    fig = go.Figure()
    if series.empty:
        return _empty(fig, pal, "Not enough history for a 50-day ratio")

    fig.add_trace(
        go.Scatter(
            x=series.index,
            y=series,
            mode="lines",
            name="Up/down volume ratio",
            line={"color": pal["series_1"], "width": 2},
            hovertext=_hover_dates(series.index),
            hovertemplate="<b>%{hovertext}</b><br>Ratio %{y:.2f}<extra></extra>",
        )
    )
    # The churn rule carries no inline label: this series oscillates around
    # 1.0 by construction, so any label placed on the line lands on the data.
    # The axis tick and the chart title already name it.
    fig.add_hline(y=1.0, line={"color": pal["axis"], "width": 1})

    # One direct label on the endpoint rather than a number on every point.
    latest = float(series.iloc[-1])
    fig.add_annotation(
        x=series.index[-1],
        y=latest,
        text=f"<b>{latest:.2f}</b>",
        showarrow=False,
        xanchor="left",
        xshift=8,
        font={"color": pal["text_primary"], "size": 13},
    )

    fig.update_layout(
        **theme.layout(
            pal,
            title={
                "text": "Up/down volume ratio — above 1.0 means heavier buying",
                "font": {"size": 14, "color": pal["text_primary"]},
                "x": 0.005,
                "xanchor": "left",
            },
            height=300,
            margin={"l": 8, "r": 8, "t": 68, "b": 8},
            showlegend=False,
        )
    )
    fig.update_xaxes(**theme.axis(pal, grid=False))
    fig.update_yaxes(**theme.axis(pal))
    return fig


def ad_trend_chart(
    analysis: StockAnalysis, pal: dict[str, str], days: int = 250
) -> go.Figure:
    """The A/D line with its least-squares trend — buying momentum or selling
    pressure, in the same units on one axis."""
    series = analysis.ad_line.dropna().tail(days)
    fig = go.Figure()
    if len(series) < 3:
        return _empty(fig, pal, "Not enough history for an A/D line")

    fig.add_trace(
        go.Scatter(
            x=series.index,
            y=series,
            mode="lines",
            name="A/D line",
            line={"color": pal["series_1"], "width": 2},
            hovertext=_hover_dates(series.index),
            hovertemplate="<b>%{hovertext}</b><br>A/D %{y:,.3s}<extra></extra>",
        )
    )

    x = np.arange(len(series), dtype="float64")
    slope, intercept = np.polyfit(x, series.to_numpy(dtype="float64"), 1)
    fig.add_trace(
        go.Scatter(
            x=series.index,
            y=slope * x + intercept,
            mode="lines",
            name=f"Trend ({'rising' if slope > 0 else 'falling'})",
            line={"color": pal["series_2"], "width": 2, "dash": "dot"},
            hovertemplate="Trend %{y:,.3s}<extra></extra>",
        )
    )

    fig.update_layout(
        **theme.layout(
            pal,
            title={
                "text": "Accumulation/distribution line — the cumulative footprint",
                "font": {"size": 14, "color": pal["text_primary"]},
                "x": 0.005,
                "xanchor": "left",
            },
            height=300,
            margin={"l": 8, "r": 8, "t": 68, "b": 8},
        )
    )
    fig.update_xaxes(**theme.axis(pal, grid=False))
    fig.update_yaxes(**theme.axis(pal, tickformat=".2s"))
    return fig


def market_chart(
    health: MarketHealth, pal: dict[str, str], days: int = 120
) -> go.Figure:
    """An index with its distribution days marked, over the rolling count."""
    frame = health.frame.tail(days)
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.66, 0.34],
        vertical_spacing=0.06,
    )
    if frame.empty:
        return _empty(fig, pal, "No index history available")

    dates = _hover_dates(frame.index)
    fig.add_trace(
        go.Scatter(
            x=frame.index,
            y=frame["Close"],
            mode="lines",
            name="Close",
            line={"color": pal["series_1"], "width": 2},
            hovertext=dates,
            customdata=frame["PctChange"].to_numpy(),
            hovertemplate=(
                "<b>%{hovertext}</b><br>Close %{y:,.2f}"
                "<br>%{customdata:+.2f}%<extra></extra>"
            ),
        ),
        row=1,
        col=1,
    )

    for column, glyph, symbol, label, level, offset in (
        (
            "IsDistributionDay",
            "▼",
            "triangle-down",
            "Distribution day",
            "critical",
            1.004,
        ),
        (
            "IsAccumulationDay",
            "▲",
            "triangle-up",
            "Up day on rising volume",
            "good",
            0.996,
        ),
    ):
        marks = frame[frame[column]]
        if marks.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=marks.index,
                y=marks["Close"] * offset,
                mode="markers",
                name=f"{glyph} {label}",
                marker={
                    "symbol": symbol,
                    "size": 11,
                    "color": pal[f"status_{level}"],
                    "line": {"width": 2, "color": pal["surface"]},
                },
                hovertext=_hover_dates(marks.index),
                customdata=np.column_stack(
                    [marks["PctChange"].to_numpy(), marks["RelVolume"].to_numpy()]
                ),
                hovertemplate=(
                    f"<b>{label}</b><br>%{{hovertext}}<br>"
                    "%{customdata[0]:+.2f}% on %{customdata[1]:.2f}x volume"
                    "<extra></extra>"
                ),
            ),
            row=1,
            col=1,
        )

    fig.add_trace(
        go.Scatter(
            x=frame.index,
            y=frame["DistributionCount"],
            mode="lines",
            name=f"Distribution days in {health.window} sessions",
            line={"color": pal["series_2"], "width": 2},
            hovertext=dates,
            hovertemplate="<b>%{hovertext}</b><br>%{y:.0f} in window<extra></extra>",
        ),
        row=2,
        col=1,
    )
    fig.add_hline(
        y=5,
        line={"color": pal["muted"], "width": 1, "dash": "dot"},
        annotation={
            "text": "5 — warning level",
            "font": {"color": pal["muted"], "size": 10},
            "xanchor": "left",
            "x": 0,
        },
        row=2,
        col=1,
    )

    fig.update_layout(
        **theme.layout(
            pal,
            title={
                "text": f"{health.name} — distribution days",
                "font": {"size": 15, "color": pal["text_primary"]},
                "x": 0.005,
                "xanchor": "left",
            },
            height=460,
            margin={"l": 8, "r": 8, "t": 70, "b": 8},
        )
    )
    fig.update_xaxes(**theme.axis(pal, grid=False), row=1, col=1)
    fig.update_xaxes(**theme.axis(pal, grid=False), row=2, col=1)
    fig.update_yaxes(
        **theme.axis(pal, title={"text": "Index", "font": {"color": pal["muted"]}}),
        row=1,
        col=1,
    )
    fig.update_yaxes(
        **theme.axis(
            pal,
            title={"text": "Count", "font": {"color": pal["muted"]}},
            rangemode="tozero",
            dtick=2,
        ),
        row=2,
        col=1,
    )
    return fig


def _empty(fig: go.Figure, pal: dict[str, str], message: str) -> go.Figure:
    fig.update_layout(**theme.layout(pal, height=240, showlegend=False))
    fig.add_annotation(
        text=message,
        showarrow=False,
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        font={"color": pal["muted"], "size": 13},
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return fig
