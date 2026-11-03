"""Yahoo Finance via yfinance — the original provider, kept as the no-key path.

Yahoo's API is unofficial and changes without notice, which is the whole reason
the paid providers exist. It stays because it needs no API key: the app runs
out of the box, and it remains the fallback while keys are being sorted out.

Its one genuine advantage: real composite index volume for ``^GSPC`` and
``^IXIC``, which is exactly what distribution-day counting wants.
"""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import yfinance as yf

from .base import OHLCV, DownloadResult, SymbolScope, empty_frame, normalize_ohlcv

name = "yahoo"
requires_key = False
label = "Yahoo Finance"


def _extract(raw: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Pull one ticker's OHLCV out of a yfinance download frame.

    yfinance returns MultiIndex columns of (field, ticker) for both single and
    multi-ticker requests, but older versions flatten single requests, so both
    layouts are handled.
    """
    if isinstance(raw.columns, pd.MultiIndex):
        tickers = set(raw.columns.get_level_values(-1))
        if ticker not in tickers:
            return empty_frame()
        frame = raw.xs(ticker, axis=1, level=-1)
    else:
        frame = raw

    if any(col not in frame.columns for col in OHLCV):
        return empty_frame()
    return normalize_ohlcv(frame)


def download(
    tickers: list[str],
    *,
    start: date,
    end: date,
    symbol_scope: SymbolScope = "all",
) -> DownloadResult:
    """One batched call for the whole list — yfinance threads it internally.

    ``symbol_scope`` is accepted for interface parity and all but ignored:
    Yahoo carries the real indexes, so it has no proxy candidates to filter.
    A caller asking specifically for a proxy gets nothing rather than a
    silently-native frame.
    """
    result = DownloadResult()
    if not tickers or symbol_scope == "proxy":
        return result

    try:
        raw = yf.download(
            tickers,
            start=start.isoformat(),
            end=(end + timedelta(days=1)).isoformat(),  # yfinance end is exclusive
            interval="1d",
            auto_adjust=True,
            progress=False,
            threads=True,
            group_by="column",
        )
    except Exception as exc:  # network/parse failures vary widely by version
        for ticker in tickers:
            result.errors[ticker] = f"download failed: {exc}"
        return result

    for ticker in tickers:
        frame = _extract(raw, ticker) if raw is not None and not raw.empty else empty_frame()
        if frame.empty:
            result.errors[ticker] = "no price data returned"
        else:
            result.bars[ticker] = frame
    return result
