"""Who is looking at the app — used only to isolate persisted user inputs.

Market data and the chosen vendor stay shared. Watchlists, signal thresholds
and anything else a person types are keyed by this identity so one viewer
cannot overwrite another.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

LOCAL_USER = "local"

_CLAIM_KEYS = (
    "email",
    "preferred_username",
    "oid",
    "sub",
    "user_name",
    "nickname",
)
_UNSAFE = re.compile(r"[^a-z0-9._-]+")


@dataclass(frozen=True)
class Viewer:
    """Stable id plus a short label for the sidebar."""

    key: str
    label: str


def current() -> Viewer:
    """Identity of the current Streamlit viewer.

    Prefers a signed-in claim (email on Streamlit Cloud / OIDC). A logged-in
    viewer with no email still gets a browser-session key so two accounts
    cannot collide. Local runs and the CLI — no login, one operator — share
    the durable ``local`` profile.
    """
    info = _user_info()
    if info:
        for key in _CLAIM_KEYS:
            value = info.get(key)
            if isinstance(value, str) and value.strip():
                claim = value.strip().lower()
                return Viewer(key=claim, label=claim)
        if info.get("is_logged_in") is True:
            session = _session_id()
            if session:
                return Viewer(key=f"session:{session}", label="this browser session")

    return Viewer(key=LOCAL_USER, label="local")


def slug(user_id: str) -> str:
    """Filesystem-safe name. Readable for emails, unique even after sanitising."""
    raw = (user_id or LOCAL_USER).strip().lower()
    readable = _UNSAFE.sub("_", raw.replace("@", "_at_").replace(":", "_"))
    readable = readable.strip("._-")[:60] or "user"
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]
    return f"{readable}__{digest}"


def _user_info() -> dict[str, object] | None:
    """Claims for this script run, or None when Streamlit is not driving."""
    ctx = _script_ctx()
    if ctx is None:
        return None
    raw = getattr(ctx, "user_info", None)
    return dict(raw) if raw else {}


def _session_id() -> str | None:
    ctx = _script_ctx()
    if ctx is None:
        return None
    sid = getattr(ctx, "session_id", None)
    return str(sid) if sid else None


def _script_ctx() -> object | None:
    try:
        from streamlit.runtime.scriptrunner_utils.script_run_context import (
            get_script_run_ctx,
        )
    except ImportError:
        return None
    try:
        return get_script_run_ctx(suppress_warning=True)
    except TypeError:
        return get_script_run_ctx()
    except Exception:
        return None
