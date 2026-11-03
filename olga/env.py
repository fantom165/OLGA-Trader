"""Load API keys from .env before anything reads the environment.

Imported for its side effect by ``olga/__init__.py``, so every entry point —
``app.py``, ``scenario.py``, ``smoke_test.py`` — gets the same keys without
each having to remember to load them.

Real environment variables win over ``.env`` (``override=False``), so a
one-off ``$env:TIINGO_API_KEY = "..."`` in the shell still beats the file.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = ROOT / ".env"

loaded = False

try:
    from dotenv import load_dotenv
except ImportError:  # python-dotenv is in requirements, but stay runnable
    def load_dotenv(*_args, **_kwargs) -> bool:  # type: ignore[misc]
        return False

if ENV_PATH.exists():
    loaded = bool(load_dotenv(ENV_PATH, override=False))
