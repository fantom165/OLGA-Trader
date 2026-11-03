"""OLGA ticker → provider ticker.

OLGA speaks Yahoo internally (``^GSPC``, ``BRK-B``) because that is what the
watchlist file has always held. Each vendor spells the same instruments
differently, and none of them sell every instrument, so this module answers one
question: what should I actually ask this vendor for?

Index entitlements are the sharp edge. Tiingo's end-of-day service covers
equities and ETFs, not equity indices. Polygon carries indices under ``I:``
tickers but only on an Indices subscription, and even then the aggregates carry
no volume — and distribution days are *defined* by volume rising. So every
index gets a liquid ETF proxy as its last resort, flagged so the UI can say so.
"""

from __future__ import annotations

from dataclasses import dataclass

# Vendor symbols for the indexes OLGA knows about. ``None`` means "this vendor
# does not sell it at all, go straight to the proxy".
INDEX_SYMBOLS: dict[str, dict[str, str | None]] = {
    "tiingo": {
        # Tiingo's daily endpoint is equities/ETFs/mutual funds. It sells no
        # equity indices at *any* plan tier — verified live against a paid
        # plan, not assumed from the docs.
        #
        # Do not be tempted to port Polygon's spellings across. Tiingo resolves
        # several of them to entirely unrelated equities, and a 200 response
        # would sail straight past every error path:
        #   COMP -> Compass Inc Class A (NYSE, ~$10), not the Nasdaq Composite
        #   SPX  -> Spenda Limited (ASX), not the S&P 500
        # Mapping an index to a small-cap stock is the one failure mode here
        # that would look plausible on a chart. Hence None, always.
        "^GSPC": None,
        "^IXIC": None,
        "^DJI": None,
        "^RUT": None,
        "^NDX": None,
    },
    "polygon": {
        "^GSPC": "I:SPX",
        "^IXIC": "I:COMP",
        "^DJI": "I:DJI",
        "^RUT": "I:RUT",
        "^NDX": "I:NDX",
    },
}

# The stand-ins. Chosen for volume that actually reflects institutional flow.
ETF_PROXIES: dict[str, str] = {
    "^GSPC": "SPY",
    "^IXIC": "QQQ",   # Nasdaq-100 ETF standing in for the Composite
    "^DJI": "DIA",
    "^RUT": "IWM",
    "^NDX": "QQQ",
}

PROXY_NOTES: dict[str, str] = {
    "^IXIC": "QQQ tracks the Nasdaq-100, not the full Composite",
}


@dataclass(frozen=True)
class Candidate:
    """One symbol worth trying, and what it costs us if we use it."""

    symbol: str
    is_proxy: bool


def is_index(ticker: str) -> bool:
    return ticker.startswith("^")


def proxy_note(ticker: str, proxy: str) -> str:
    detail = PROXY_NOTES.get(ticker)
    base = f"{proxy} used in place of {ticker}"
    return f"{base} — {detail}" if detail else base


def _equity_symbol(provider: str, ticker: str) -> str:
    """Translate a stock ticker into the vendor's spelling.

    Share classes are the only real divergence: Yahoo and Tiingo write
    ``BRK-B``, Polygon writes ``BRK.B``.
    """
    if provider == "polygon":
        return ticker.replace("-", ".")
    return ticker


def candidates(provider: str, ticker: str) -> list[Candidate]:
    """Symbols to try for ``ticker``, best first.

    Stocks yield exactly one candidate. Indexes yield the vendor's native
    symbol (when it has one) followed by the ETF proxy, so a plan without
    index entitlements degrades instead of going dark.
    """
    if provider == "yahoo":
        return [Candidate(ticker, False)]

    if not is_index(ticker):
        return [Candidate(_equity_symbol(provider, ticker), False)]

    out: list[Candidate] = []
    native = INDEX_SYMBOLS.get(provider, {}).get(ticker)
    if native:
        out.append(Candidate(native, False))
    proxy = ETF_PROXIES.get(ticker)
    if proxy:
        out.append(Candidate(_equity_symbol(provider, proxy), True))
    if not out:
        # An index we have no mapping for. Ask as-is and let the vendor 404.
        out.append(Candidate(ticker, False))
    return out
