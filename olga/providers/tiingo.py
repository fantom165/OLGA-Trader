"""Tiingo end-of-day prices.

    GET https://api.tiingo.com/tiingo/daily/{ticker}/prices
        ?startDate=YYYY-MM-DD&endDate=YYYY-MM-DD&format=json
    Authorization: Token <TIINGO_API_KEY>

Tiingo returns both raw and adjusted series per bar. OLGA takes the ``adj*``
fields throughout — including ``adjVolume`` — so that a split does not put a
step change in either the weekly range or the volume baseline it is measured
against.

Dates come back as ``2026-09-15T00:00:00.000Z``: that midnight-UTC stamp *is*
the trading date, so it is read as a plain date and never shifted into another
timezone.
"""

from __future__ import annotations

import os
from datetime import date

import pandas as pd
import requests

from .base import (
    DownloadResult,
    ProviderError,
    SymbolScope,
    download_candidates,
    empty_frame,
    normalize_ohlcv,
    request_json,
)

name = "tiingo"
requires_key = True
key_env = "TIINGO_API_KEY"
label = "Tiingo (EOD)"

BASE_URL = os.environ.get("TIINGO_BASE_URL", "https://api.tiingo.com").rstrip("/")
DEFAULT_WORKERS = 4

# adjusted field -> raw fallback, for the rare bar that carries only one of them
FIELDS = {
    "Open": ("adjOpen", "open"),
    "High": ("adjHigh", "high"),
    "Low": ("adjLow", "low"),
    "Close": ("adjClose", "close"),
    "Volume": ("adjVolume", "volume"),
}


def api_key() -> str | None:
    key = os.environ.get(key_env, "").strip()
    return key or None


def _max_workers() -> int:
    try:
        return max(1, int(os.environ.get("TIINGO_MAX_WORKERS", DEFAULT_WORKERS)))
    except ValueError:
        return DEFAULT_WORKERS


def _to_frame(rows: list[dict]) -> pd.DataFrame:
    if not rows:
        return empty_frame()

    raw = pd.DataFrame(rows)
    if "date" not in raw.columns:
        return empty_frame()

    frame = pd.DataFrame(index=pd.RangeIndex(len(raw)))
    for column, (adjusted, plain) in FIELDS.items():
        series = None
        if adjusted in raw.columns:
            series = pd.to_numeric(raw[adjusted], errors="coerce")
        if plain in raw.columns:
            fallback = pd.to_numeric(raw[plain], errors="coerce")
            series = fallback if series is None else series.fillna(fallback)
        if series is None:
            return empty_frame()
        frame[column] = series.to_numpy()

    # The stamp names the trading date; strip UTC rather than converting, or a
    # midnight-UTC bar would slide into the previous session.
    stamps = pd.to_datetime(raw["date"], utc=True, errors="coerce")
    frame.index = pd.DatetimeIndex(stamps).tz_localize(None)
    return normalize_ohlcv(frame)


def download(
    tickers: list[str],
    *,
    start: date,
    end: date,
    symbol_scope: SymbolScope = "all",
) -> DownloadResult:
    key = api_key()
    if not key:
        raise ProviderError(
            f"Tiingo is the selected provider but {key_env} is not set. "
            "Put it in .env (see .env.example) or switch data.provider to yahoo."
        )

    session = requests.Session()
    session.headers.update(
        {"Authorization": f"Token {key}", "Content-Type": "application/json"}
    )
    params = {
        "startDate": start.isoformat(),
        "endDate": end.isoformat(),
        "format": "json",
        "resampleFreq": "daily",
    }

    def fetch_one(symbol: str) -> pd.DataFrame:
        payload = request_json(
            session, f"{BASE_URL}/tiingo/daily/{symbol}/prices", params=params
        )
        if isinstance(payload, dict):
            # Tiingo reports "detail" on a bad symbol or an out-of-plan request.
            # Symbol-level, not provider-level: the rest of the list still loads.
            raise RuntimeError(str(payload.get("detail") or payload)[:200])
        if not isinstance(payload, list):
            return empty_frame()
        return _to_frame(payload)

    try:
        return download_candidates(
            name,
            tickers,
            fetch_one,
            max_workers=_max_workers(),
            symbol_scope=symbol_scope,
        )
    finally:
        session.close()
