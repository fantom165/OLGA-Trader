"""Vertex AI Gemini client for Ask OLGA.

Numbers still come from OLGA's own maths. This module only turns a packed
dashboard snapshot + the user's question into prose. If Gemini is missing,
unauthenticated, or errors, the chat layer falls back to the rule-based
replies in ``olga.chat``.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any

DEFAULT_PROJECT = "olga-trader"
DEFAULT_LOCATION = "global"
DEFAULT_MODEL = "gemini-3.7-flash"


def project() -> str:
    return os.getenv("GOOGLE_CLOUD_PROJECT", DEFAULT_PROJECT).strip() or DEFAULT_PROJECT


def location() -> str:
    return os.getenv("GOOGLE_CLOUD_LOCATION", DEFAULT_LOCATION).strip() or DEFAULT_LOCATION


def model_name() -> str:
    return os.getenv("GEMINI_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL


def enabled() -> bool:
    flag = os.getenv("OLGA_LLM", "on").strip().lower()
    return flag not in {"0", "false", "off", "no"}


def status() -> tuple[bool, str]:
    """Whether Ask OLGA should try Gemini, and a short UI caption."""
    if not enabled():
        return False, "LLM off — using the built-in rule replies"
    return True, f"{model_name()} · Vertex project {project()}"


def _client():
    from google import genai

    kwargs: dict[str, Any] = {
        "vertexai": True,
        "project": project(),
        "location": location(),
    }
    # Optional API key still works on Vertex if ADC is not set up.
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if api_key:
        kwargs["api_key"] = api_key
    return genai.Client(**kwargs)


@lru_cache(maxsize=1)
def _cached_client():
    return _client()


def generate(
    *,
    system: str,
    user: str,
    history: list[tuple[str, str]] | None = None,
) -> str:
    """Return Gemini text, or raise if the call cannot be completed."""
    from google.genai import types

    client = _cached_client()
    contents: list[types.Content] = []
    for role, text in history or []:
        if not text or not text.strip():
            continue
        api_role = "user" if role == "user" else "model"
        contents.append(
            types.Content(role=api_role, parts=[types.Part.from_text(text=text)])
        )
    contents.append(types.Content(role="user", parts=[types.Part.from_text(text=user)]))

    thinking = os.getenv("GEMINI_THINKING_LEVEL", "LOW").strip().upper() or "LOW"
    config = types.GenerateContentConfig(
        system_instruction=system,
        max_output_tokens=2048,
        thinking_config=types.ThinkingConfig(thinking_level=thinking),
    )
    response = client.models.generate_content(
        model=model_name(),
        contents=contents,
        config=config,
    )
    text = (getattr(response, "text", None) or "").strip()
    if not text:
        raise RuntimeError("Gemini returned an empty reply")
    return text
