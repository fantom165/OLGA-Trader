"""Market-data providers behind one interface.

``get_provider(name)`` hands back a module satisfying ``base.Provider``. The
active provider is chosen in ``config/watchlist.json`` under ``data.provider``;
``resolve_name("auto")`` picks the best keyed provider available so the app
upgrades itself the moment a key lands in ``.env``.
"""

from __future__ import annotations

from types import ModuleType

from . import polygon, symbols, tiingo, yahoo
from .base import (
    OHLCV,
    DownloadResult,
    Provider,
    ProviderError,
    empty_frame,
    has_usable_volume,
    normalize_ohlcv,
)

__all__ = [
    "OHLCV",
    "DownloadResult",
    "Provider",
    "ProviderError",
    "PROVIDER_NAMES",
    "empty_frame",
    "get_provider",
    "has_key",
    "has_usable_volume",
    "key_status",
    "normalize_ohlcv",
    "polygon",
    "resolve_name",
    "symbols",
    "tiingo",
    "yahoo",
]

_REGISTRY: dict[str, ModuleType] = {
    yahoo.name: yahoo,
    tiingo.name: tiingo,
    polygon.name: polygon,
}

PROVIDER_NAMES = tuple(_REGISTRY)

# Preference order for ``auto``: paid and stable first, unofficial last.
AUTO_ORDER = (tiingo.name, polygon.name, yahoo.name)


def get_provider(name: str) -> ModuleType:
    provider = _REGISTRY.get((name or "").strip().lower())
    if provider is None:
        raise ProviderError(
            f"unknown data provider {name!r}; expected one of "
            + ", ".join(PROVIDER_NAMES)
        )
    return provider


def has_key(name: str) -> bool:
    """True when this provider is usable right now."""
    provider = get_provider(name)
    if not getattr(provider, "requires_key", False):
        return True
    return provider.api_key() is not None


def resolve_name(name: str) -> str:
    """Turn a configured value — possibly ``auto`` — into a concrete provider."""
    wanted = (name or "auto").strip().lower()
    if wanted != "auto":
        return get_provider(wanted).name
    for candidate in AUTO_ORDER:
        if has_key(candidate):
            return candidate
    return yahoo.name


def key_status(name: str) -> tuple[bool, str]:
    """(usable, human explanation) for the sidebar. Never reveals the key."""
    provider = get_provider(name)
    if not getattr(provider, "requires_key", False):
        return True, "no key required"
    env = getattr(provider, "key_env", "")
    if provider.api_key() is not None:
        return True, f"{env} set"
    return False, f"{env} missing"
