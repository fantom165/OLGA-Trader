"""A fresh last trade, separate from the cached daily bars.

Daily history is cached for hours, and the newest bar's close can still be
today's open. A "current price" question reads one quote response: Tiingo
IEX when a key is set, otherwise a Yahoo quote. The last trade, the session
open, and the previous close stay separate fields.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from urllib.parse import quote as urlquote

import pandas as pd

from . import config
from .providers import tiingo

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


def quote_from_yahoo(ticker: str, payload: object) -> Quote | None:
    """Last trade from one Yahoo chart quote. Never the session open."""
    if not isinstance(payload, dict):
        return None
    chart = payload.get("chart")
    if not isinstance(chart, dict):
        return None
    rows = chart.get("result") or []
    if not rows or not isinstance(rows[0], dict):
        return None
    block = rows[0]
    meta = block.get("meta") if isinstance(block.get("meta"), dict) else {}
    last = _finite(meta.get("regularMarketPrice"))
    if last is None or last <= 0:
        return None
    previous = _finite(meta.get("chartPreviousClose"))
    if previous is None:
        previous = _finite(meta.get("previousClose"))
    return Quote(
        ticker=ticker,
        last=last,
        open=_yahoo_open(block),
        previous_close=previous,
        as_of=_epoch(meta.get("regularMarketTime")),
        source="yahoo",
        detail="Yahoo Finance quote, may be delayed",
    )


def _fetch_fresh(symbol: str) -> Quote | None:
    # One small response. Tiingo IEX when the key is set; Yahoo quote otherwise.
    # Indexes are not on the IEX endpoint.
    getters = [_from_yahoo]
    if not symbol.startswith("^") and tiingo.api_key() is not None:
        getters = [_from_tiingo, _from_yahoo]
    for getter in getters:
        try:
            found = getter(symbol)
        except Exception:
            found = None
        if found is not None:
            return found
    return None


def _from_yahoo(symbol: str) -> Quote | None:
    payload = _get_json(
        "https://query1.finance.yahoo.com/v8/finance/chart/" + urlquote(symbol, safe=""),
        params={"range": "1d", "interval": "1d", "includePrePost": "false"},
        headers={"User-Agent": "Mozilla/5.0"},
    )
    return quote_from_yahoo(symbol, payload)


def _from_tiingo(symbol: str) -> Quote | None:
    if tiingo.api_key() is None:
        return None
    payload = _get_json(
        f"{tiingo.BASE_URL}/iex/{urlquote(symbol, safe='')}",
        headers={
            "Authorization": f"Token {tiingo.api_key()}",
            "Content-Type": "application/json",
        },
    )
    found = quote_from_tiingo(payload)
    if found is None:
        return None
    return Quote(
        ticker=symbol,
        last=found.last,
        open=found.open,
        previous_close=found.previous_close,
        as_of=found.as_of,
        source=found.source,
        detail=found.detail,
    )


def _get_json(
    url: str,
    *,
    params: dict[str, str] | None = None,
    headers: dict[str, str] | None = None,
) -> object:
    """One GET. A quote must not sit through the daily-bar retry loop."""
    import requests

    response = requests.get(url, params=params, headers=headers, timeout=8)
    response.raise_for_status()
    return response.json()


def _yahoo_open(block: dict) -> float | None:
    indicators = block.get("indicators")
    if not isinstance(indicators, dict):
        return None
    quotes = indicators.get("quote") or []
    if not quotes or not isinstance(quotes[0], dict):
        return None
    opens = quotes[0].get("open")
    if not isinstance(opens, list):
        return None
    for value in reversed(opens):
        number = _finite(value)
        if number is not None and number > 0:
            return number
    return None


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
