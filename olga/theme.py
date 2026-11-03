"""Palette and Plotly chrome.

Colours come from a palette validated for colour-vision deficiency in both
light and dark modes (adjacent-pair CVD deltaE 24.7 light / 26.8 dark against
an >= 8 target, all six checks passing). Accumulation and distribution are
*states*, so they wear the reserved status colours and are always paired with a
label or glyph — never carried by hue alone.
"""

from __future__ import annotations

from typing import Any

# Status roles are fixed and never themed.
STATUS = {
    "good": "#0ca30c",
    "warning": "#fab219",
    "serious": "#ec835a",
    "critical": "#d03b3b",
}

LIGHT: dict[str, str] = {
    "surface": "#fcfcfb",
    "page": "#f9f9f7",
    "text_primary": "#0b0b0b",
    "text_secondary": "#52514e",
    "muted": "#898781",
    "grid": "#e1e0d9",
    "axis": "#c3c2b7",
    "series_1": "#2a78d6",
    "series_2": "#eb6834",
    "neutral_fill": "#c3c2b7",
    "delta_up": "#006300",
}

DARK: dict[str, str] = {
    "surface": "#1a1a19",
    "page": "#0d0d0d",
    "text_primary": "#ffffff",
    "text_secondary": "#c3c2b7",
    "muted": "#898781",
    "grid": "#2c2c2a",
    "axis": "#383835",
    "series_1": "#3987e5",
    "series_2": "#d95926",
    "neutral_fill": "#52514e",
    "delta_up": "#0ca30c",
}

FONT_FAMILY = 'system-ui, -apple-system, "Segoe UI", sans-serif'

# The three states a bar can carry, with the glyph that encodes them
# independently of colour.
SIGNAL_STYLE = {
    "accumulation": {"color": STATUS["good"], "glyph": "▲", "label": "Accumulation"},
    "distribution": {"color": STATUS["critical"], "glyph": "▼", "label": "Distribution"},
    "neutral": {"color": None, "glyph": "•", "label": "Neutral"},
}

GRADE_LEVEL = {
    "A": "good",
    "A-": "good",
    "B+": "good",
    "B": "good",
    "B-": "good",
    "C+": "warning",
    "C": "warning",
    "C-": "warning",
    "D+": "serious",
    "D": "serious",
    "D-": "critical",
    "E": "critical",
}


def palette(mode: str = "light") -> dict[str, str]:
    """Colour roles for the requested mode, with status roles folded in."""
    base = dict(DARK if mode == "dark" else LIGHT)
    base.update({f"status_{k}": v for k, v in STATUS.items()})
    return base


def signal_color(signal: str, pal: dict[str, str]) -> str:
    """Colour for a weekly signal; neutral recedes toward the surface."""
    style = SIGNAL_STYLE.get(signal, SIGNAL_STYLE["neutral"])
    return style["color"] or pal["neutral_fill"]


def grade_color(grade: str, pal: dict[str, str]) -> str:
    return pal[f"status_{GRADE_LEVEL.get(grade, 'warning')}"]


def layout(pal: dict[str, str], **overrides: Any) -> dict[str, Any]:
    """Base Plotly layout: recessive hairline chrome, generous padding."""
    base: dict[str, Any] = {
        "paper_bgcolor": pal["surface"],
        "plot_bgcolor": pal["surface"],
        "font": {"family": FONT_FAMILY, "size": 12, "color": pal["text_secondary"]},
        "margin": {"l": 8, "r": 8, "t": 48, "b": 8},
        "hovermode": "x unified",
        "hoverlabel": {
            "bgcolor": pal["surface"],
            "bordercolor": pal["axis"],
            "font": {"family": FONT_FAMILY, "size": 12, "color": pal["text_primary"]},
        },
        "legend": {
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "x": 0,
            "bgcolor": "rgba(0,0,0,0)",
            "font": {"color": pal["text_secondary"]},
        },
        "title": {
            "font": {"size": 15, "color": pal["text_primary"]},
            "x": 0,
            "xanchor": "left",
            "pad": {"b": 12},
        },
    }
    base.update(overrides)
    return base


def axis(pal: dict[str, str], *, grid: bool = True, **overrides: Any) -> dict[str, Any]:
    """A solid-hairline axis one shade off the surface. Never dashed."""
    base: dict[str, Any] = {
        "showgrid": grid,
        "gridcolor": pal["grid"],
        "gridwidth": 1,
        "griddash": "solid",
        "zeroline": False,
        "linecolor": pal["axis"],
        "linewidth": 1,
        "showline": True,
        "ticks": "outside",
        "ticklen": 4,
        "tickcolor": pal["axis"],
        "tickfont": {"color": pal["muted"], "size": 11},
        "title": {"font": {"color": pal["muted"], "size": 11}},
    }
    base.update(overrides)
    return base
