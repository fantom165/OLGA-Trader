"""Polygon daily aggregate bars.

    GET https://api.polygon.io/v2/aggs/ticker/{sym}/range/1/day/{from}/{to}
        ?adjusted=true&sort=asc&limit=50000
    Authorization: Bearer <POLYGON_API_KEY>

``adjusted=true`` gives split-adjusted bars. Polygon does *not* apply dividend
adjustments to aggregates, so a long history of a high-yield name will show
slightly larger drawdowns here than under Yahoo's ``auto_adjust``. Every OLGA
reading is a ratio inside a rolling window, so this shifts nothing material —
but it is the one real difference between the providers.

Polygon stamps each daily bar with midnight *Eastern* as epoch milliseconds, so
timestamps are converted through ``America/New_York`` to land on the right
trading date.

Free tier is 5 requests/minute, so requests are serialised by default. Raise
``POLYGON_MAX_WORKERS`` if you are on a paid plan.
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

name = "polygon"
requires_key = True
key_env = "POLYGON_API_KEY"
label = "Polygon (daily aggregates)"

# Polygon rebranded to Massive in 2026; api.polygon.io still serves the v2 API.
BASE_URL = os.environ.get("POLYGON_BASE_URL", "https://api.polygon.io").rstrip("/")
DEFAULT_WORKERS = 1
MAX_PAGES = 10  # 50k bars/page — a guard against a runaway next_url chain

FIELDS = {"Open": "o", "High": "h", "Low": "l", "Close": "c", "Volume": "v"}


def api_key() -> str | None:
    key = os.environ.get(key_env, "").strip()
    return key or None


def _max_workers() -> int:
    try:
        return max(1, int(os.environ.get("POLYGON_MAX_WORKERS", DEFAULT_WORKERS)))
    except ValueError:
        return DEFAULT_WORKERS


def _to_frame(results: list[dict]) -> pd.DataFrame:
    if not results:
        return empty_frame()

    raw = pd.DataFrame(results)
    if "t" not in raw.columns:
        return empty_frame()

    frame = pd.DataFrame(index=pd.RangeIndex(len(raw)))
    for column, source in FIELDS.items():
        if source in raw.columns:
            frame[column] = pd.to_numeric(raw[source], errors="coerce").to_numpy()
        elif column == "Volume":
            # Index aggregates omit volume entirely. Carried through as NaN so
            # base.has_usable_volume can reject the frame and force the proxy.
            frame[column] = float("nan")
        else:
            return empty_frame()

    stamps = pd.to_datetime(raw["t"], unit="ms", utc=True, errors="coerce")
    frame.index = pd.DatetimeIndex(stamps).tz_convert("America/New_York")
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
            f"Polygon is the selected provider but {key_env} is not set. "
            "Put it in .env (see .env.example) or switch data.provider to yahoo."
        )

    session = requests.Session()
    session.headers.update({"Authorization": f"Bearer {key}"})

    def fetch_one(symbol: str) -> pd.DataFrame:
        url = (
            f"{BASE_URL}/v2/aggs/ticker/{symbol}/range/1/day"
            f"/{start.isoformat()}/{end.isoformat()}"
        )
        params: dict[str, object] | None = {
            "adjusted": "true",
            "sort": "asc",
            "limit": 50000,
        }
        rows: list[dict] = []
        for _ in range(MAX_PAGES):
            payload = request_json(session, url, params=params)
            status = str(payload.get("status", "")).upper()
            if status in ("ERROR", "NOT_AUTHORIZED"):
                raise RuntimeError(
                    str(payload.get("message") or payload.get("error") or status)[:200]
                )
            rows.extend(payload.get("results") or [])
            next_url = payload.get("next_url")
            if not next_url:
                break
            # next_url already carries the cursor; the key rides in the header.
            url, params = next_url, None
        return _to_frame(rows)

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
