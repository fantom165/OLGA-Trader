"""The contract every market-data provider honours.

A provider's only job is to turn a list of OLGA tickers into split- and
dividend-adjusted daily OHLCV frames. Everything downstream — indicators,
rating, charts — sees identical frames no matter which vendor produced them,
which is what makes the provider swappable at all.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Callable, Literal, Protocol

import pandas as pd

OHLCV = ["Open", "High", "Low", "Close", "Volume"]

# Vendors rate-limit aggressively on free tiers. These are polite, not fast.
RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})
MAX_ATTEMPTS = 4
BACKOFF_SECONDS = 1.5
REQUEST_TIMEOUT = 20.0


class ProviderError(RuntimeError):
    """A provider could not be used at all — usually a missing API key.

    Distinct from a per-ticker failure, which is reported in
    ``DownloadResult.errors`` so the rest of the watchlist still loads.
    """


@dataclass
class DownloadResult:
    """What one provider managed to fetch, keyed by *OLGA* ticker.

    ``proxies`` records substitutions the provider had to make — in practice
    an index the vendor does not sell, served from an ETF standing in for it.
    The caller keys the frame under the requested index symbol regardless, so
    ``market.analyze_indexes`` keeps working; the substitution is surfaced in
    the UI rather than hidden.
    """

    bars: dict[str, pd.DataFrame] = field(default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)
    proxies: dict[str, str] = field(default_factory=dict)


# Which spellings of a ticker a caller is willing to accept. ``native`` means
# "the real instrument or nothing"; ``proxy`` means "only the stand-in". The
# split lets data.fetch walk a index down a chain of fallbacks — real symbol,
# then another vendor, then an ETF — without ever re-requesting a symbol that
# has already failed.
SymbolScope = Literal["all", "native", "proxy"]


class Provider(Protocol):
    """Structural type for a provider module or object."""

    name: str

    def download(
        self,
        tickers: list[str],
        *,
        start: date,
        end: date,
        symbol_scope: SymbolScope = "all",
    ) -> DownloadResult: ...


def empty_frame() -> pd.DataFrame:
    frame = pd.DataFrame(columns=OHLCV, dtype="float64")
    frame.index = pd.DatetimeIndex([], name="Date")
    return frame


def normalize_ohlcv(frame: pd.DataFrame) -> pd.DataFrame:
    """Coerce a vendor frame into OLGA's one true shape.

    Exactly ``Open, High, Low, Close, Volume`` as float64, on a naive
    date-normalized index named ``Date``, deduped and sorted. Rows without a
    close are dropped — a bar with no close tells us nothing about who won the
    session. Anything unusable comes back empty rather than half-formed.
    """
    if frame is None or frame.empty:
        return empty_frame()

    missing = [col for col in OHLCV if col not in frame.columns]
    if missing:
        return empty_frame()

    out = frame.loc[:, OHLCV].copy()
    out = out.apply(pd.to_numeric, errors="coerce")
    out = out.dropna(subset=["Close"])
    if out.empty:
        return empty_frame()

    out = out.astype("float64")

    index = pd.DatetimeIndex(out.index)
    if index.tz is not None:
        # Strip the zone, keep the wall clock: a daily bar's timestamp already
        # names its trading date in the exchange's own time.
        index = index.tz_localize(None)
    out.index = index.normalize()
    out.index.name = "Date"

    out = out[~out.index.duplicated(keep="last")].sort_index()
    return out


def has_usable_volume(frame: pd.DataFrame) -> bool:
    """True when the frame carries real volume.

    Index aggregates from some vendors come back with price only. Distribution
    days are defined by volume rising, so a zero-volume frame would silently
    count zero distribution days forever — worse than an error.
    """
    if frame.empty or "Volume" not in frame.columns:
        return False
    volume = frame["Volume"]
    return bool(volume.notna().any() and (volume.fillna(0.0) > 0).any())


def request_json(
    session: Any,
    url: str,
    *,
    params: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
) -> Any:
    """GET JSON with polite retries on rate limits and transient 5xx.

    Honours ``Retry-After`` when the vendor sends one, otherwise backs off
    linearly. Raises the underlying ``requests`` exception once attempts run
    out, which the provider turns into a per-ticker error message.
    """
    last_exc: Exception | None = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            response = session.get(
                url, params=params, headers=headers, timeout=REQUEST_TIMEOUT
            )
        except Exception as exc:  # connection resets, DNS, read timeouts
            last_exc = exc
            if attempt == MAX_ATTEMPTS:
                raise
            time.sleep(BACKOFF_SECONDS * attempt)
            continue

        if response.status_code in RETRY_STATUSES and attempt < MAX_ATTEMPTS:
            wait = BACKOFF_SECONDS * attempt
            retry_after = response.headers.get("Retry-After")
            if retry_after:
                try:
                    wait = max(wait, min(float(retry_after), 60.0))
                except ValueError:
                    pass
            time.sleep(wait)
            continue

        response.raise_for_status()
        return response.json()

    if last_exc is not None:
        raise last_exc
    raise ProviderError(f"gave up after {MAX_ATTEMPTS} attempts: {url}")


def download_candidates(
    provider_name: str,
    tickers: list[str],
    fetch_one: Callable[[str], pd.DataFrame],
    *,
    max_workers: int = 4,
    symbol_scope: SymbolScope = "all",
) -> DownloadResult:
    """Drive a per-symbol fetcher across a ticker list, honouring the symbol map.

    Shared by every keyed provider, because the interesting logic is identical
    for all of them: translate the ticker, try the candidate symbols in order,
    and reject an index that arrives without volume. ``fetch_one`` fetches
    exactly one vendor symbol and raises on failure; it knows nothing about
    proxies or scopes.

    ``symbol_scope`` narrows the candidate list so ``data.fetch`` can ask for
    the real instrument first and only come back for a proxy once the other
    fallbacks are exhausted.
    """
    from . import symbols as symbol_map

    result = DownloadResult()
    if not tickers:
        return result

    def resolve(ticker: str) -> tuple[str, pd.DataFrame | None, str, bool, str]:
        """-> (ticker, frame|None, symbol used, was proxy, failure reason)"""
        wanted = [
            c
            for c in symbol_map.candidates(provider_name, ticker)
            if symbol_scope == "all"
            or (symbol_scope == "proxy") == c.is_proxy
        ]
        if not wanted:
            return ticker, None, "", False, f"not served by {provider_name}"

        reason = "no price data returned"
        for candidate in wanted:
            try:
                frame = fetch_one(candidate.symbol)
            except ProviderError:
                raise
            except Exception as exc:
                reason = f"{candidate.symbol}: {describe_http_error(exc)}"
                continue
            if frame.empty:
                reason = f"{candidate.symbol}: no price data returned"
                continue
            if symbol_map.is_index(ticker) and not has_usable_volume(frame):
                # Price-only index bars cannot support distribution-day counting.
                reason = f"{candidate.symbol}: index bars carry no volume"
                continue
            return ticker, frame, candidate.symbol, candidate.is_proxy, ""
        return ticker, None, "", False, reason

    workers = max(1, min(int(max_workers), len(tickers)))
    if workers == 1:
        resolved = [resolve(t) for t in tickers]
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            resolved = list(pool.map(resolve, tickers))

    for ticker, frame, symbol, is_proxy, reason in resolved:
        if frame is None:
            result.errors[ticker] = reason
            continue
        result.bars[ticker] = frame
        if is_proxy:
            result.proxies[ticker] = symbol
    return result


def _vendor_detail(response: Any) -> str:
    """The vendor's own explanation, if it sent one.

    Both vendors put something useful in the body — Tiingo a ``detail`` field,
    Polygon a ``message`` — and it is usually more specific than anything we
    could infer from the status code alone. Never contains the API key, which
    travels in a header.
    """
    if response is None:
        return ""
    try:
        payload = response.json()
    except Exception:
        return ""
    if not isinstance(payload, dict):
        return ""
    for field_name in ("detail", "message", "error"):
        value = payload.get(field_name)
        if value:
            return str(value).strip()[:200]
    return ""


def describe_http_error(exc: Exception) -> str:
    """A short, key-free explanation of a failed request."""
    response = getattr(exc, "response", None)
    status = getattr(response, "status_code", None)
    detail = _vendor_detail(response)

    if status in (401, 403):
        base = "not authorized (check the API key and your plan's entitlements)"
    elif status == 404:
        base = "symbol not found on this provider"
    elif status == 429:
        # Vendor quotas are hourly or daily, so no amount of backoff inside a
        # single request will clear them. Say what actually helps.
        base = "rate limited — quota spent; cached bars serve until it resets"
    elif status is not None:
        base = f"HTTP {status}"
    else:
        return str(exc) or exc.__class__.__name__

    return f"{base}: {detail}" if detail else base
