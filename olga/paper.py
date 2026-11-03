"""Hypothetical paper trades from OLGA's live buy and exit rules.

One long at a time. Enter when the top-ranked buy scenario is "Best — live"
(the two constructive setups only). Fill the next session's open. Exit the
next open after any Exit watch item is triggered, or after a close under the
support level from the entry signal. A position still on at the end of the
loaded history is marked to the last close and left open.

Buy and exit-watch decisions run on the last session of each week, plus the
final loaded bar. Reading every session is the same rules, but a three-year
history makes that walk too slow to sit behind a button. The hard stop still
watches every close. Fills stay on the next session's open. Every decision
sees bars through that session only — never a later close.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd

from olga import analysis, market
from olga.config import Settings
from scenario import _exit_watch, _nearest_support, _rank_scenarios

_BEST_LIVE = "Best — live"
_IGNORED_ENTRY = "Buy despite a C grade"


@dataclass
class PaperTrade:
    """One hypothetical long, filled at the open after the signal."""

    signal_date: pd.Timestamp
    entry_date: pd.Timestamp
    entry_price: float
    entry_reason: str
    shares: int
    exit_signal_date: pd.Timestamp | None
    exit_date: pd.Timestamp | None
    exit_price: float | None
    exit_reason: str
    pnl: float
    return_pct: float
    status: str  # closed | open


@dataclass
class PaperResult:
    """Blotter, mark-to-market equity, and the summary tiles."""

    ticker: str
    starting_cash: float
    ending_equity: float
    cash: float
    closed: list[PaperTrade]
    open_trade: PaperTrade | None
    win_rate: float
    max_drawdown: float
    equity_curve: pd.DataFrame

    @property
    def total_return_pct(self) -> float:
        if self.starting_cash <= 0:
            return float("nan")
        return (self.ending_equity / self.starting_cash - 1.0) * 100.0

    @property
    def trade_count(self) -> int:
        return len(self.closed) + (0 if self.open_trade is None else 1)


def _through(frame: pd.DataFrame, day: pd.Timestamp) -> pd.DataFrame:
    """Rows on or before ``day``. Later sessions are not in the slice."""
    if frame.empty:
        return frame
    pos = int(frame.index.searchsorted(pd.Timestamp(day), side="right"))
    return frame.iloc[:pos]


def as_of(
    daily: pd.DataFrame,
    day: pd.Timestamp,
    settings: Settings,
    *,
    ticker: str = "",
) -> analysis.StockAnalysis:
    """Analyze ``daily`` as it stood at the close of ``day``.

    Bars after ``day`` are dropped before any indicator runs.
    """
    window = _through(daily, day)
    if window.empty:
        raise ValueError(f"no bars on or before {day}")
    return analysis.analyze(ticker, window, settings)


def _indexes_as_of(
    indexes: dict[str, pd.DataFrame],
    day: pd.Timestamp,
    settings: Settings,
) -> list[market.MarketHealth]:
    sliced: dict[str, pd.DataFrame] = {}
    names: list[str] = []
    for name, frame in indexes.items():
        window = _through(frame, day)
        if window.empty:
            continue
        sliced[name] = window
        names.append(name)
    return market.analyze_indexes(
        sliced,
        names,
        pct_threshold=settings.distribution_day_pct,
        window=settings.distribution_day_window,
    )


def _empty_result(ticker: str, cash: float) -> PaperResult:
    return PaperResult(
        ticker=ticker,
        starting_cash=cash,
        ending_equity=cash,
        cash=cash,
        closed=[],
        open_trade=None,
        win_rate=float("nan"),
        max_drawdown=0.0,
        equity_curve=pd.DataFrame(columns=["Date", "Equity"]),
    )


def _trade_from_position(
    *,
    signal_date: pd.Timestamp,
    entry_date: pd.Timestamp,
    entry_price: float,
    entry_reason: str,
    shares: int,
    cost: float,
    exit_signal_date: pd.Timestamp | None,
    exit_date: pd.Timestamp | None,
    exit_price: float,
    exit_reason: str,
    status: str,
) -> PaperTrade:
    pnl = shares * exit_price - cost
    return_pct = (pnl / cost) * 100.0 if cost else float("nan")
    return PaperTrade(
        signal_date=signal_date,
        entry_date=entry_date,
        entry_price=entry_price,
        entry_reason=entry_reason,
        shares=shares,
        exit_signal_date=exit_signal_date,
        exit_date=exit_date,
        exit_price=exit_price,
        exit_reason=exit_reason,
        pnl=pnl,
        return_pct=return_pct,
        status=status,
    )


def _ends_week(dates: pd.DatetimeIndex, i: int) -> bool:
    """True on the last session of an ISO week, and on the final bar."""
    if i >= len(dates) - 1:
        return True
    return dates[i].isocalendar()[:2] != dates[i + 1].isocalendar()[:2]


def run_paper(
    daily: pd.DataFrame,
    indexes: dict[str, pd.DataFrame],
    settings: Settings,
    *,
    cash: float = 10_000.0,
    ticker: str = "",
    hard_stop: bool = True,
) -> PaperResult:
    """Walk completed sessions and build a one-name paper blotter.

    ``indexes`` is the same bar dict the Market tab uses. A decision session's
    exit check sees those series truncated to that session, so index
    distribution can fire without reading the future.
    """
    if daily is None or daily.empty or cash <= 0:
        return _empty_result(ticker, cash if cash > 0 else 0.0)

    if not daily.index.is_monotonic_increasing:
        daily = daily.sort_index()
    indexes = {
        name: (
            frame
            if frame.index.is_monotonic_increasing
            else frame.sort_index()
        )
        for name, frame in indexes.items()
        if frame is not None and not frame.empty
    }

    starting = float(cash)
    idle = float(cash)
    shares = 0
    cost = 0.0
    entry_date: pd.Timestamp | None = None
    entry_price = 0.0
    entry_reason = ""
    entry_signal: pd.Timestamp | None = None
    entry_support: float | None = None
    pending: tuple | None = None
    closed: list[PaperTrade] = []
    equity_rows: list[tuple[pd.Timestamp, float]] = []

    dates = daily.index
    last_i = len(dates) - 1

    for i, day in enumerate(dates):
        row = daily.iloc[i]
        open_px = float(row["Open"])

        if pending is not None and pending[0] == "entry":
            qty = (
                int(idle // open_px)
                if math.isfinite(open_px) and open_px > 0
                else 0
            )
            if qty >= 1:
                cost = qty * open_px
                idle -= cost
                shares = qty
                entry_date = day
                entry_price = open_px
                entry_reason = pending[1]
                entry_signal = pending[2]
                entry_support = pending[3]
            pending = None
        elif pending is not None and pending[0] == "exit" and shares > 0:
            if not math.isfinite(open_px) or open_px <= 0:
                open_px = float(row["Close"])
            closed.append(
                _trade_from_position(
                    signal_date=entry_signal,
                    entry_date=entry_date,
                    entry_price=entry_price,
                    entry_reason=entry_reason,
                    shares=shares,
                    cost=cost,
                    exit_signal_date=pending[2],
                    exit_date=day,
                    exit_price=open_px,
                    exit_reason=pending[1],
                    status="closed",
                )
            )
            idle += shares * open_px
            shares = 0
            cost = 0.0
            entry_date = None
            entry_signal = None
            entry_support = None
            pending = None

        # Full OLGA reads are weekly. The hard stop is a close comparison,
        # so it still runs every session.
        reasons: list[str] = []
        close_px = float(row["Close"])
        if (
            shares > 0
            and hard_stop
            and entry_support is not None
            and math.isfinite(close_px)
            and close_px < entry_support
        ):
            reasons.append(
                f"Close below entry-date support ({entry_support:.2f})"
            )
        if _ends_week(dates, i):
            item = as_of(daily, day, settings, ticker=ticker)
            if shares > 0 and entry_date is not None:
                reasons.extend(
                    check.name
                    for check in _exit_watch(
                        item, settings, _indexes_as_of(indexes, day, settings)
                    )
                    if check.status == "triggered"
                )
            elif pending is None and shares == 0 and i < last_i:
                ranked = _rank_scenarios(item)
                if (
                    ranked
                    and ranked[0][2] == _BEST_LIVE
                    and ranked[0][1] != _IGNORED_ENTRY
                ):
                    pending = (
                        "entry",
                        ranked[0][1],
                        day,
                        _nearest_support(item),
                    )
        if shares > 0 and reasons and pending is None:
            pending = ("exit", "; ".join(reasons), day)

        close_px = float(row["Close"])
        if shares and math.isfinite(close_px):
            equity = idle + shares * close_px
        else:
            equity = idle
        equity_rows.append((pd.Timestamp(day), equity))

    open_trade = None
    if shares > 0 and entry_date is not None and entry_signal is not None:
        mark = float(daily["Close"].iloc[-1])
        exit_signal = None
        reason = "Open — marked to last close"
        if pending is not None and pending[0] == "exit":
            exit_signal = pending[2]
            reason = (
                "Open — marked to last close; exit signaled "
                f"({pending[1]}) but the next open is not in this history"
            )
        open_trade = _trade_from_position(
            signal_date=entry_signal,
            entry_date=entry_date,
            entry_price=entry_price,
            entry_reason=entry_reason,
            shares=shares,
            cost=cost,
            exit_signal_date=exit_signal,
            exit_date=None,
            exit_price=mark,
            exit_reason=reason,
            status="open",
        )

    wins = sum(1 for trade in closed if trade.pnl > 0)
    win_rate = (wins / len(closed)) if closed else float("nan")
    peak = 0.0
    max_dd = 0.0
    for _, equity in equity_rows:
        if equity > peak:
            peak = equity
        if peak > 0:
            max_dd = max(max_dd, (peak - equity) / peak)

    ending = equity_rows[-1][1] if equity_rows else idle
    return PaperResult(
        ticker=ticker,
        starting_cash=starting,
        ending_equity=ending,
        cash=idle,
        closed=closed,
        open_trade=open_trade,
        win_rate=win_rate,
        max_drawdown=max_dd,
        equity_curve=pd.DataFrame(equity_rows, columns=["Date", "Equity"]),
    )
