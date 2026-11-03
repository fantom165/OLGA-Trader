"""Daily OHLCV retrieval — the single seam between OLGA and the outside world.

``fetch()`` is the only public entry point, and it returns the same thing no
matter which vendor served it: split-adjusted daily ``Open/High/Low/Close/
Volume`` as float64 on a naive date index. Indicators, ratings and charts have
never needed to know where the bars came from, and still do not.

The vendor is chosen in ``config/watchlist.json`` under ``data.provider``
(``auto`` | ``tiingo`` | ``polygon`` | ``yahoo``) — see ``olga/providers/``.
Cache files are namespaced by provider so switching vendors never re-serves
another vendor's bars.
"""

from __future__ import annotations

import pickle
import time
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from . import config, providers
from .providers import OHLCV, ProviderError  # noqa: F401  — re-exported

CACHE_TTL_SECONDS = 60 * 60 * 4  # intraday staleness is irrelevant for daily bars

# How far back each period string reaches. Providers are date-based, so the
# conversion happens once, here, rather than in every provider.
PERIOD_OFFSETS: dict[str, pd.DateOffset] = {
    "1mo": pd.DateOffset(months=1),
    "3mo": pd.DateOffset(months=3),
    "6mo": pd.DateOffset(months=6),
    "1y": pd.DateOffset(years=1),
    "2y": pd.DateOffset(years=2),
    "3y": pd.DateOffset(years=3),
    "5y": pd.DateOffset(years=5),
    "10y": pd.DateOffset(years=10),
    "max": pd.DateOffset(years=30),
}
DEFAULT_PERIOD = "3y"


@dataclass
class FetchResult:
    """Bars keyed by ticker, plus whatever failed and why."""

    bars: dict[str, pd.DataFrame]
    errors: dict[str, str]
    from_cache: list[str]
    provider: str = "yahoo"
    proxies: dict[str, str] = field(default_factory=dict)
    sources: dict[str, str] = field(default_factory=dict)

    @property
    def ok(self) -> list[str]:
        return list(self.bars)

    def source_of(self, ticker: str) -> str:
        """Which provider actually served this ticker."""
        return self.sources.get(ticker, self.provider)


# --------------------------------------------------------------------------- #
# Periods
# --------------------------------------------------------------------------- #
def period_to_dates(period: str, *, today: date | None = None) -> tuple[date, date]:
    """Turn a caller's period string into an inclusive ``(start, end)`` pair.

    ``ytd`` and bare day counts (``120d``) are accepted alongside the table
    above; anything unrecognised falls back to the default window rather than
    raising, because a typo in a period should not take the dashboard down.
    """
    end = today or date.today()
    key = (period or DEFAULT_PERIOD).strip().lower()
    stamp = pd.Timestamp(end)

    if key == "ytd":
        return date(end.year, 1, 1), end
    if key.endswith("d") and key[:-1].isdigit():
        return end - timedelta(days=int(key[:-1])), end

    offset = PERIOD_OFFSETS.get(key, PERIOD_OFFSETS[DEFAULT_PERIOD])
    return (stamp - offset).date(), end


# --------------------------------------------------------------------------- #
# Cache
# --------------------------------------------------------------------------- #
def _cache_path(provider: str, ticker: str, period: str) -> Path:
    safe = ticker.replace("^", "_idx_").replace("/", "-").replace(".", "_")
    return config.CACHE_DIR / f"{provider}__{safe}__{period}.pkl"


def _read_cache(provider: str, ticker: str, period: str, ttl: int) -> pd.DataFrame | None:
    path = _cache_path(provider, ticker, period)
    if not path.exists():
        return None
    if ttl >= 0 and (time.time() - path.stat().st_mtime) > ttl:
        return None
    try:
        with path.open("rb") as fh:
            frame = pickle.load(fh)
    except (pickle.UnpicklingError, OSError, EOFError, AttributeError):
        return None
    if isinstance(frame, pd.DataFrame) and not frame.empty:
        return frame
    return None


def _write_cache(provider: str, ticker: str, period: str, frame: pd.DataFrame) -> None:
    path = _cache_path(provider, ticker, period)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("wb") as fh:
            pickle.dump(frame, fh, protocol=pickle.HIGHEST_PROTOCOL)
    except OSError:
        pass  # a cache that can't be written is a slow app, not a broken one


# --------------------------------------------------------------------------- #
# Fetch
# --------------------------------------------------------------------------- #
def active_provider(configured: str | None = None) -> str:
    """The provider name this process will actually use."""
    name = configured if configured is not None else config.load().provider
    return providers.resolve_name(name)


SOURCE_ATTR = "olga_source"
PROXY_ATTR = "olga_proxy"


def _stamp_provenance(frame: pd.DataFrame, source: str, proxy: str | None) -> None:
    """Record who served this frame, on the frame itself.

    ``DataFrame.attrs`` survives pickling, so the cache remembers that an index
    came from Yahoo (or from an ETF) even though it is filed under the active
    provider's cache namespace. Without this the UI's proxy warning would
    vanish on the first cache hit and quietly mislead.
    """
    frame.attrs[SOURCE_ATTR] = source
    if proxy:
        frame.attrs[PROXY_ATTR] = proxy


def _recover_provenance(
    frame: pd.DataFrame,
    ticker: str,
    active: str,
    sources: dict[str, str],
    proxies: dict[str, str],
) -> None:
    """Read provenance back off a cached frame. Pre-1.1 caches simply lack it."""
    attrs = getattr(frame, "attrs", None) or {}
    source = attrs.get(SOURCE_ATTR)
    if source and source != active:
        sources[ticker] = str(source)
    proxy = attrs.get(PROXY_ATTR)
    if proxy:
        proxies[ticker] = str(proxy)


def _download(
    name: str,
    tickers: list[str],
    start: date,
    end: date,
    scope: str,
) -> providers.DownloadResult:
    """Call one provider, turning any escape into per-ticker errors."""
    if not tickers:
        return providers.DownloadResult()
    try:
        return providers.get_provider(name).download(
            tickers, start=start, end=end, symbol_scope=scope
        )
    except ProviderError as exc:
        # Provider-level failure — a missing key, or an unknown name. Every
        # ticker fails the same way; say so once, per ticker.
        return providers.DownloadResult(
            errors={ticker: str(exc) for ticker in tickers}
        )
    except Exception as exc:  # defensive: a provider should not leak these
        return providers.DownloadResult(
            errors={ticker: f"download failed: {exc}" for ticker in tickers}
        )


def _resolve_indexes(
    name: str,
    downloaded: providers.DownloadResult,
    misses: list[str],
    start: date,
    end: date,
    sources: dict[str, str],
) -> None:
    """Walk unresolved indexes down the fallback chain, in place.

    The active provider has already been asked for the real index symbol and
    come up short — either it does not sell indexes (Tiingo) or it returned
    price without volume (Polygon). Two fallbacks remain, in this order:

    1. **Yahoo**, which carries genuine composite index volume. This is what
       distribution-day counting is defined against, so it is worth a second
       vendor rather than settling for a proxy.
    2. **An ETF proxy** on the active provider, if Yahoo is unreachable too.
       SPY volume is not composite volume, and the count really does differ,
       so this is a last resort and the UI says when it happens.
    """
    yahoo_name = providers.yahoo.name

    def unresolved() -> list[str]:
        return [
            t
            for t in misses
            if t not in downloaded.bars and providers.symbols.is_index(t)
        ]

    if name == yahoo_name:
        return

    pending = unresolved()
    if pending:
        composite = _download(yahoo_name, pending, start, end, "native")
        for ticker, frame in composite.bars.items():
            if providers.has_usable_volume(frame):
                downloaded.bars[ticker] = frame
                downloaded.errors.pop(ticker, None)
                sources[ticker] = yahoo_name

    pending = unresolved()
    if pending:
        proxied = _download(name, pending, start, end, "proxy")
        downloaded.bars.update(proxied.bars)
        downloaded.proxies.update(proxied.proxies)
        for ticker in proxied.bars:
            downloaded.errors.pop(ticker, None)
        for ticker, why in proxied.errors.items():
            downloaded.errors.setdefault(ticker, why)


def fetch(
    tickers: list[str],
    period: str = DEFAULT_PERIOD,
    *,
    use_cache: bool = True,
    ttl: int = CACHE_TTL_SECONDS,
    provider: str | None = None,
) -> FetchResult:
    """Fetch daily bars for ``tickers`` from the active provider.

    Cache first, then one provider call for everything still missing, then the
    index fallback chain described in ``_resolve_indexes``. A ticker nobody can
    serve falls back to any cached copy, however old, before being reported as
    an error — a stale chart beats a blank one, and ``from_cache`` tells the UI
    how much of the view is not fresh.
    """
    name = active_provider(provider)
    wanted = config.dedupe([config.normalize_ticker(t) for t in tickers])
    bars: dict[str, pd.DataFrame] = {}
    errors: dict[str, str] = {}
    proxies: dict[str, str] = {}
    sources: dict[str, str] = {}
    from_cache: list[str] = []

    misses: list[str] = []
    for ticker in wanted:
        cached = _read_cache(name, ticker, period, ttl) if use_cache else None
        if cached is not None:
            bars[ticker] = cached
            from_cache.append(ticker)
            _recover_provenance(cached, ticker, name, sources, proxies)
        else:
            misses.append(ticker)

    if misses:
        start, end = period_to_dates(period)
        # "native" only: an index must not settle for a proxy until Yahoo has
        # had its turn, and asking for "all" here would spend the proxy call
        # before the better fallback was tried.
        downloaded = _download(name, misses, start, end, "native")
        _resolve_indexes(name, downloaded, misses, start, end, sources)

        proxies.update(downloaded.proxies)
        for ticker in misses:
            frame = downloaded.bars.get(ticker)
            if frame is not None and not frame.empty:
                _stamp_provenance(
                    frame, sources.get(ticker, name), downloaded.proxies.get(ticker)
                )
                bars[ticker] = frame
                if use_cache:
                    _write_cache(name, ticker, period, frame)
                continue
            stale = _read_cache(name, ticker, period, ttl=-1)
            if stale is not None:
                bars[ticker] = stale
                from_cache.append(ticker)
            else:
                errors[ticker] = downloaded.errors.get(ticker, "no price data returned")

    ordered = {t: bars[t] for t in wanted if t in bars}
    return FetchResult(
        bars=ordered,
        errors=errors,
        from_cache=from_cache,
        provider=name,
        proxies={t: s for t, s in proxies.items() if t in ordered},
        sources={t: s for t, s in sources.items() if t in ordered},
    )


def clear_cache() -> int:
    """Delete every cached frame, for every provider. Returns files removed."""
    if not config.CACHE_DIR.exists():
        return 0
    removed = 0
    for path in config.CACHE_DIR.glob("*.pkl"):
        try:
            path.unlink()
            removed += 1
        except OSError:
            pass
    return removed


def last_bar_date(bars: dict[str, pd.DataFrame]) -> pd.Timestamp | None:
    dates = [frame.index[-1] for frame in bars.values() if not frame.empty]
    return max(dates) if dates else None
