"""Watchlist, thresholds, and data-provider configuration.

``config/watchlist.json`` is the **seed**: default tickers, index list,
signal-threshold defaults, and a default ``data.provider`` for a first
visit. Personal copies — tickers, thresholds, and the chosen vendor —
live under ``config/users/`` so one viewer's Save cannot overwrite
another's. The CLI still reads the seed file via ``load()``.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict, replace
from pathlib import Path
from typing import Any, Iterable

from .identity import slug

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config" / "watchlist.json"
USERS_DIR = ROOT / "config" / "users"
CACHE_DIR = ROOT / ".cache"

DEFAULT_TICKERS = ["NVDA", "MSFT", "AAPL", "AMZN", "GOOGL", "META"]
DEFAULT_INDEXES = ["^GSPC", "^IXIC"]

# "auto" resolves to the best keyed provider at runtime — Tiingo, then Polygon,
# then Yahoo — so dropping a key into .env switches vendor with no edit here.
DEFAULT_PROVIDER = "auto"

INDEX_NAMES = {
    "^GSPC": "S&P 500",
    "^IXIC": "Nasdaq Composite",
    "^DJI": "Dow Jones Industrial",
    "^RUT": "Russell 2000",
    "^NDX": "Nasdaq 100",
}


@dataclass(frozen=True)
class Settings:
    """Thresholds that define what counts as accumulation or distribution.

    The defaults follow the conventions described in the source article: a
    50-day up/down volume ratio, weekly closes in the top/bottom 30% of their
    range on volume well above normal, and index distribution days at -0.2%
    on rising volume counted over a rolling 25 sessions.
    """

    ud_window: int = 50
    weekly_vol_window: int = 10
    weekly_vol_multiple: float = 1.25
    accumulation_close_position: float = 0.70
    distribution_close_position: float = 0.30
    signal_lookback_weeks: int = 13
    distribution_day_pct: float = -0.2
    distribution_day_window: int = 25

    def with_overrides(self, **kwargs: Any) -> "Settings":
        known = {k: v for k, v in kwargs.items() if k in self.__dataclass_fields__}
        return replace(self, **known)


@dataclass
class Config:
    tickers: list[str]
    indexes: list[str]
    settings: Settings
    provider: str = DEFAULT_PROVIDER

    def to_dict(self) -> dict[str, Any]:
        return {
            "tickers": self.tickers,
            "indexes": self.indexes,
            "data": {"provider": self.provider},
            "settings": asdict(self.settings),
        }

    def to_user_dict(self) -> dict[str, Any]:
        """Personal overlay — tickers, thresholds, and chosen vendor.

        Indexes stay on the seed file; they are market context, not a
        personal pick.
        """
        return {
            "tickers": self.tickers,
            "data": {"provider": self.provider},
            "settings": asdict(self.settings),
        }


def normalize_ticker(raw: str) -> str:
    return raw.strip().upper().replace(" ", "")


def load() -> Config:
    """Read the shared seed file, tolerating a missing or partial file.

    This is the CLI path and the default for a viewer who has not saved
    a personal watchlist yet. The Streamlit app should call ``load_for``.
    """
    data = _read_json(CONFIG_PATH)
    return _config_from(data)


def load_for(user_id: str) -> Config:
    """Seed indexes plus this viewer's tickers, thresholds, and vendor."""
    shared = load()
    overlay = _read_json(_user_path(user_id))
    if not overlay:
        return shared

    if "tickers" in overlay and overlay["tickers"] is not None:
        shared.tickers = dedupe(normalize_ticker(t) for t in overlay["tickers"])
    raw_settings = overlay.get("settings") or {}
    if raw_settings:
        shared.settings = shared.settings.with_overrides(**raw_settings)
    # OLGA_PROVIDER still pins the process; otherwise the viewer's pick wins.
    if provider_override() is None:
        chosen = _configured_provider(overlay)
        if chosen:
            shared.provider = chosen
    return shared


def save(config: Config) -> None:
    """Write the shared seed file (default vendor, indexes, default watchlist)."""
    _write_json(CONFIG_PATH, config.to_dict())


def save_user(user_id: str, config: Config) -> None:
    """Persist this viewer's tickers, thresholds, and vendor. Not the seed file."""
    _write_json(_user_path(user_id), config.to_user_dict())


def provider_override() -> str | None:
    """``OLGA_PROVIDER``, if the environment sets one.

    Exists so a single run can be pointed at another vendor —
    ``$env:OLGA_PROVIDER = "yahoo"`` — without editing the watchlist file. The
    UI has to know about it: an override wins over the config file, so the
    sidebar's provider picker is disabled while one is in force rather than
    silently losing every change back to the environment.
    """
    value = os.environ.get("OLGA_PROVIDER", "").strip().lower()
    return value or None


def _config_from(data: dict[str, Any]) -> Config:
    tickers = [normalize_ticker(t) for t in data.get("tickers") or DEFAULT_TICKERS]
    indexes = [normalize_ticker(t) for t in data.get("indexes") or DEFAULT_INDEXES]
    settings = Settings().with_overrides(**(data.get("settings") or {}))
    return Config(
        tickers=dedupe(tickers),
        indexes=dedupe(indexes),
        settings=settings,
        provider=_provider_from(data),
    )


def _configured_provider(data: dict[str, Any]) -> str | None:
    raw = (data.get("data") or {}).get("provider")
    if raw is None:
        raw = data.get("provider")
    if raw is None:
        return None
    value = str(raw).strip().lower()
    return value or None


def _provider_from(data: dict[str, Any]) -> str:
    """Env override first, then the config file, then ``auto``."""
    return provider_override() or _configured_provider(data) or DEFAULT_PROVIDER


def _user_path(user_id: str) -> Path:
    return USERS_DIR / f"{slug(user_id)}.json"


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2) + "\n"
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def dedupe(items: Iterable[str]) -> list[str]:
    """Order-preserving dedupe, dropping blanks."""
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return out


def display_name(ticker: str) -> str:
    return INDEX_NAMES.get(ticker, ticker)
