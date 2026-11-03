"""A fresh last trade, separate from the cached daily bars.

Daily history is cached for hours, and the newest bar's close can still be
today's open. A "current price" question has to read a quote. The last trade,
the session open, and the previous close stay separate fields so a reply
cannot substitute one for another.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import pandas as pd

from . import config
from .providers import polygon, tiingo
from .providers.base import request_json

QUOTE_TTL_SECONDS = 45
_STALE_AFTER = pd.Timedelta(minutes=20)

_cache: dict[str, tuple[float, "Quote"]] = {}


@dataclass(frozen=True)
class Quote:
    """One last trade, with the open and prior close kept beside it."""

    ticker: str
    last: float
    open: float | None
    previous_close: float | None
    as_of: pd.Timestamp | None
    source: str
    detail: str


def is_stale(as_of: pd.Timestamp | None, *, now: pd.Timestamp | None = None) -> bool:
    """True when the print is too old to call current."""
    if as_of is None or pd.isna(as_of):
        return False
    stamp = pd.Timestamp(as_of)
    clock = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize("America/New_York")
    if clock.tzinfo is None:
        clock = clock.tz_localize("America/New_York")
    return clock.tz_convert("UTC") - stamp.tz_convert("UTC") > _STALE_AFTER


def quote_from_ohlcv(
    ticker: str,
    frame: pd.DataFrame | None,
    *,
    source: str,
    detail: str,
) -> Quote | None:
    """Last trade from intraday bars. A daily candle is rejected.

    The last price is the final bar's close. The open is the first bar of
    that same session. Those are different rows on a minute series, which is
    the whole point: a daily bar can report them as the same number.
    """
    if frame is None or frame.empty or "Close" not in getattr(frame, "columns", []):
        return None

    work = frame.sort_index()
    if isinstance(work.columns, pd.MultiIndex):
        work = work.copy()
        work.columns = work.columns.get_level_values(0)

    dates = _exchange_dates(work.index)
    if dates is None or dates.empty:
        return None
    last_day = dates.iloc[-1]
    session = work.loc[dates.eq(last_day).to_numpy()]
    session = session.dropna(subset=["Close"])
    if session.empty or not _is_intraday(session):
        return None

    last = _finite(session["Close"].iloc[-1])
    if last is None or last <= 0:
        return None
    open_px = _finite(session["Open"].iloc[0]) if "Open" in session.columns else None
    prior = work.loc[dates.lt(last_day).to_numpy()]
    previous = _finite(prior["Close"].iloc[-1]) if not prior.empty else None
    as_of = pd.Timestamp(session.index[-1])
    return Quote(
        ticker=ticker,
        last=last,
        open=open_px,
        previous_close=previous,
        as_of=as_of,
        source=source,
        detail=detail,
    )


def quote_from_polygon(ticker: str, payload: object) -> Quote | None:
    """Last trade from a Polygon stock snapshot. Never the day's open."""
    ticker_obj = _polygon_ticker(payload)
    if ticker_obj is None:
        return None

    last: float | None = None
    as_of: pd.Timestamp | None = None
    trade = ticker_obj.get("lastTrade") or {}
    if isinstance(trade, dict):
        last = _finite(trade.get("p"))
        as_of = _epoch(trade.get("t"))
    if last is None:
        minute = ticker_obj.get("min") or {}
        if isinstance(minute, dict):
            last = _finite(minute.get("c"))
            as_of = as_of or _epoch(minute.get("t"))
    if last is None:
        day = ticker_obj.get("day") or {}
        if isinstance(day, dict):
            # Running daily close. Still not the open (`o`).
            last = _finite(day.get("c"))
    if last is None or last <= 0:
        return None

    day = ticker_obj.get("day") if isinstance(ticker_obj.get("day"), dict) else {}
    prev = ticker_obj.get("prevDay") if isinstance(ticker_obj.get("prevDay"), dict) else {}
    return Quote(
        ticker=ticker,
        last=last,
        open=_finite(day.get("o")),
        previous_close=_finite(prev.get("c")),
        as_of=as_of,
        source="polygon",
        detail="Polygon last trade; free-tier prints can be delayed",
    )


def quote_from_tiingo(payload: object) -> Quote | None:
    """Tiingo IEX / reference quote. The open is stored beside the last."""
    row = payload[0] if isinstance(payload, list) and payload else payload
    if not isinstance(row, dict):
        return None
    # tngoLast is Tiingo's cross-venue reference. `last` is the IEX print.
    reference = _finite(row.get("tngoLast"))
    iex_last = _finite(row.get("last"))
    last = reference if reference is not None else iex_last
    if last is None or last <= 0:
        return None
    detail = (
        "Tiingo reference price"
        if reference is not None
        else "Tiingo IEX print — IEX only, not the consolidated tape"
    )
    as_of = pd.to_datetime(row.get("timestamp"), utc=True, errors="coerce")
    if pd.isna(as_of):
        as_of = None
    else:
        as_of = pd.Timestamp(as_of)
    return Quote(
        ticker=str(row.get("ticker") or "").upper() or "?",
        last=last,
        open=_finite(row.get("open")),
        previous_close=_finite(row.get("prevClose")),
        as_of=as_of,
        source="tiingo",
        detail=detail,
    )


def fetch(ticker: str, *, use_cache: bool = True) -> Quote | None:
    """Latest trade for ``ticker``. Misses are not cached."""
    symbol = config.normalize_ticker(ticker)
    if not symbol:
        return None
    now = time.time()
    if use_cache:
        hit = _cache.get(symbol)
        if hit is not None and now - hit[0] < QUOTE_TTL_SECONDS:
            return hit[1]
    quote = _fetch_fresh(symbol)
    if quote is not None and use_cache:
        _cache[symbol] = (now, quote)
    return quote


def _fetch_fresh(symbol: str) -> Quote | None:
    getters = (_from_yahoo,)
    if not symbol.startswith("^"):
        getters = (_from_yahoo, _from_polygon, _from_tiingo)
    for getter in getters:
        try:
            quote = getter(symbol)
        except Exception:
            quote = None
        if quote is not None:
            return quote
    return None


def _from_yahoo(symbol: str) -> Quote | None:
    import yfinance as yf

    hist = yf.Ticker(symbol).history(
        period="5d",
        interval="1m",
        auto_adjust=False,
        prepost=False,
        timeout=15,
    )
    return quote_from_ohlcv(
        symbol,
        hist,
        source="yahoo",
        detail="Yahoo Finance 1-minute bars, may be delayed",
    )


def _from_polygon(symbol: str) -> Quote | None:
    if polygon.api_key() is None:
        return None
    import requests

    session = requests.Session()
    session.headers.update({"Authorization": f"Bearer {polygon.api_key()}"})
    vendor = symbol.replace("-", ".")
    try:
        payload = request_json(
            session,
            f"{polygon.BASE_URL}/v2/snapshot/locale/us/markets/stocks/tickers/{vendor}",
        )
    finally:
        session.close()
    return quote_from_polygon(symbol, payload)


def _from_tiingo(symbol: str) -> Quote | None:
    if tiingo.api_key() is None:
        return None
    import requests

    session = requests.Session()
    session.headers.update(
        {"Authorization": f"Token {tiingo.api_key()}", "Content-Type": "application/json"}
    )
    try:
        payload = request_json(session, f"{tiingo.BASE_URL}/iex/{symbol}")
    finally:
        session.close()
    quote = quote_from_tiingo(payload)
    if quote is None:
        return None
    return Quote(
        ticker=symbol,
        last=quote.last,
        open=quote.open,
        previous_close=quote.previous_close,
        as_of=quote.as_of,
        source=quote.source,
        detail=quote.detail,
    )


def _polygon_ticker(payload: object) -> dict | None:
    if not isinstance(payload, dict):
        return None
    status = str(payload.get("status", "")).upper()
    if status in {"ERROR", "NOT_AUTHORIZED"}:
        return None
    nested = payload.get("ticker")
    if isinstance(nested, dict):
        return nested
    if "day" in payload or "lastTrade" in payload:
        return payload
    return None


def _exchange_dates(index: pd.Index) -> pd.Series | None:
    try:
        stamps = pd.DatetimeIndex(pd.to_datetime(index))
    except (TypeError, ValueError):
        return None
    if stamps.tz is not None:
        stamps = stamps.tz_convert("America/New_York")
    return pd.Series([stamp.date() for stamp in stamps], index=index)


def _is_intraday(session: pd.DataFrame) -> bool:
    """Daily candles are stamped at midnight and are not a quote."""
    if len(session) >= 2:
        deltas = pd.Series(pd.DatetimeIndex(session.index)).diff().dropna()
        if deltas.empty:
            return False
        return deltas.median() <= pd.Timedelta(minutes=30)
    stamp = pd.Timestamp(session.index[-1])
    if stamp.tzinfo is not None:
        stamp = stamp.tz_convert("America/New_York")
    return not (stamp.hour == 0 and stamp.minute == 0 and stamp.second == 0)


def _epoch(value: object) -> pd.Timestamp | None:
    try:
        n = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if n > 10**17:
        unit = "ns"
    elif n > 10**14:
        unit = "us"
    elif n > 10**11:
        unit = "ms"
    else:
        unit = "s"
    stamp = pd.to_datetime(n, unit=unit, utc=True, errors="coerce")
    if pd.isna(stamp):
        return None
    return pd.Timestamp(stamp)


def _finite(value: object) -> float | None:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number
