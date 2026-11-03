"""OLGA Trader — reading institutional footprints in price range and volume.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import math

import pandas as pd
import streamlit as st

from olga import analysis as analysis_mod
from olga import charts, chat, config, data, identity, llm as olga_llm, market, providers, rating, theme
from olga.indicators import ACCUMULATION, DISTRIBUTION, NEUTRAL

PERIOD_CHOICES = {
    "1 year": "1y",
    "2 years": "2y",
    "3 years": "3y",
    "5 years": "5y",
}
WEEKS_CHOICES = {"6 months": 26, "1 year": 52, "18 months": 78, "3 years": 156}

st.set_page_config(
    page_title="OLGA Trader",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --------------------------------------------------------------------------- #
# Theme
# --------------------------------------------------------------------------- #
def current_mode() -> str:
    """Follow Streamlit's active theme so chart surfaces match the page."""
    try:
        kind = st.context.theme.type
    except Exception:
        kind = None
    if kind in ("light", "dark"):
        return kind
    try:
        base = st.get_option("theme.base")
    except Exception:
        base = None
    return "dark" if base == "dark" else "light"


MODE = current_mode()
PAL = theme.palette(MODE)

st.markdown(
    f"""
    <style>
      .olga-tile {{
        background: {PAL['surface']};
        border: 1px solid {PAL['grid']};
        border-radius: 10px;
        padding: 14px 16px;
        height: 100%;
      }}
      .olga-tile-label {{
        color: {PAL['muted']};
        font-size: 11px;
        letter-spacing: .04em;
        text-transform: uppercase;
        margin-bottom: 6px;
      }}
      .olga-tile-value {{
        color: {PAL['text_primary']};
        font-size: 28px;
        line-height: 1.1;
        font-weight: 600;
      }}
      .olga-tile-note {{ color: {PAL['text_secondary']}; font-size: 12px; margin-top: 4px; }}
      .olga-chip {{
        display: inline-flex; align-items: center; gap: 6px;
        border-radius: 999px; padding: 3px 11px;
        font-size: 13px; font-weight: 600;
      }}
      .olga-callout {{
        border-left: 3px solid {PAL['axis']};
        padding: 10px 14px; margin: 4px 0 14px 0;
        color: {PAL['text_secondary']}; font-size: 14px;
      }}
    </style>
    """,
    unsafe_allow_html=True,
)


def tile(label: str, value: str, note: str = "") -> str:
    return (
        f"<div class='olga-tile'><div class='olga-tile-label'>{label}</div>"
        f"<div class='olga-tile-value'>{value}</div>"
        f"<div class='olga-tile-note'>{note}</div></div>"
    )


def chip(text: str, level: str) -> str:
    """A status chip. Always carries a glyph and a word, never colour alone."""
    color = PAL[f"status_{level}"]
    return (
        f"<span class='olga-chip' style='background:{color}22;color:{color};"
        f"border:1px solid {color}66'>{text}</span>"
    )


STANCE_GLYPH = {ACCUMULATION: "▲", DISTRIBUTION: "▼", NEUTRAL: "•"}
STANCE_LEVEL = {ACCUMULATION: "good", DISTRIBUTION: "critical", NEUTRAL: "warning"}


def fmt(value: float, spec: str = ",.2f", dash: str = "—") -> str:
    if value is None or not isinstance(value, (int, float)) or not math.isfinite(value):
        return dash
    return format(value, spec)


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
@st.cache_data(ttl=900, show_spinner=False)
def load_bars(
    tickers: tuple[str, ...],
    period: str,
    provider: str,
    nonce: int,
    skip_disk: bool,
) -> data.FetchResult:
    """``nonce`` is bumped by the refresh button to bypass this session's cache.

    ``skip_disk`` forces a vendor round-trip for this call only. It does not
    wipe the shared market-data cache other viewers still read.
    """
    return data.fetch(
        list(tickers),
        period=period,
        provider=provider,
        use_cache=not skip_disk,
    )


viewer = identity.current()
cfg = config.load_for(viewer.key)
ACTIVE_PROVIDER = data.active_provider(cfg.provider)
PROVIDER_OK, PROVIDER_KEY_NOTE = providers.key_status(ACTIVE_PROVIDER)
if "nonce" not in st.session_state:
    st.session_state.nonce = 0

# --------------------------------------------------------------------------- #
# Sidebar: the single control surface for everything below
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.markdown("### Watchlist")
    st.caption(f"Saved only for **{viewer.label}** — other users keep their own list.")
    edited = st.text_area(
        "One ticker per line",
        value="\n".join(cfg.tickers),
        height=210,
        label_visibility="collapsed",
        key=f"watchlist_text_{identity.slug(viewer.key)}",
        help="Yahoo-style symbols, e.g. NVDA, TTD, BRK-B — translated to the "
        "active provider's spelling automatically. Saved to your account only.",
    )
    if st.button("Save watchlist", width="stretch"):
        cfg.tickers = config.dedupe(
            [config.normalize_ticker(line) for line in edited.splitlines()]
        )
        config.save_user(viewer.key, cfg)
        st.session_state.nonce += 1
        st.rerun()

    st.markdown("### View")
    period_label = st.selectbox("History to load", list(PERIOD_CHOICES), index=2)
    weeks_label = st.selectbox("Weeks charted", list(WEEKS_CHOICES), index=2)

    with st.expander("Signal thresholds"):
        s = cfg.settings
        vol_multiple = st.slider(
            "Heavy volume is at least … × average",
            1.0,
            3.0,
            float(s.weekly_vol_multiple),
            0.05,
            help="How much above its 10-week average a week's volume must be to "
            "count as an institutional footprint.",
        )
        acc_pos = st.slider(
            "Accumulation: close at least this far up the week's range",
            0.5,
            0.95,
            float(s.accumulation_close_position),
            0.05,
        )
        dist_pos = st.slider(
            "Distribution: close at most this far up the week's range",
            0.05,
            0.5,
            float(s.distribution_close_position),
            0.05,
        )
        lookback = st.slider(
            "Weeks in the balance window", 4, 52, int(s.signal_lookback_weeks)
        )
        ud_window = st.slider("Up/down volume window (days)", 20, 100, int(s.ud_window))
        dd_window = st.slider(
            "Distribution-day window (sessions)", 10, 50, int(s.distribution_day_window)
        )

    settings = cfg.settings.with_overrides(
        weekly_vol_multiple=vol_multiple,
        accumulation_close_position=acc_pos,
        distribution_close_position=dist_pos,
        signal_lookback_weeks=lookback,
        ud_window=ud_window,
        distribution_day_window=dd_window,
    )
    if settings != cfg.settings:
        cfg.settings = settings
        config.save_user(viewer.key, cfg)

    st.markdown("### Data")
    provider_choices = ["auto", *providers.PROVIDER_NAMES]
    override = config.provider_override()
    chosen_provider = st.selectbox(
        "Provider",
        provider_choices,
        index=provider_choices.index(cfg.provider)
        if cfg.provider in provider_choices
        else 0,
        disabled=override is not None,
        help="Saved to your account only. auto picks the first provider with "
        "an API key: Tiingo, then Polygon, then Yahoo (which needs none)."
        if override is None
        else "Locked by the OLGA_PROVIDER environment variable.",
    )
    # Only persist a real choice. While OLGA_PROVIDER is set it would win on
    # the next load anyway, and saving would rerun forever.
    if override is None and chosen_provider != cfg.provider:
        cfg.provider = chosen_provider
        config.save_user(viewer.key, cfg)
        st.session_state.nonce += 1
        st.rerun()

    if override is not None:
        st.caption(f"Pinned to **{ACTIVE_PROVIDER}** by OLGA_PROVIDER.")
    if PROVIDER_OK:
        via = " (via auto)" if cfg.provider == "auto" else ""
        st.caption(f"Serving **{ACTIVE_PROVIDER}**{via} — {PROVIDER_KEY_NOTE}.")
    else:
        st.error(
            f"**{ACTIVE_PROVIDER}** is selected but {PROVIDER_KEY_NOTE}. "
            "Add it to `.env`, or switch the provider above."
        )

    col_a, col_b = st.columns(2)
    if col_a.button("Refresh", width="stretch"):
        st.session_state.nonce += 1
        st.session_state.skip_disk_nonce = st.session_state.nonce
        st.rerun()
    if col_b.button("Clear cache", width="stretch"):
        st.session_state.nonce += 1
        st.session_state.skip_disk_nonce = st.session_state.nonce
        st.toast("This session will refetch. Shared market-data cache is unchanged.")
        st.rerun()
    st.caption("Refresh applies to this session only.")

period = PERIOD_CHOICES[period_label]
weeks_shown = WEEKS_CHOICES[weeks_label]
all_tickers = tuple(config.dedupe(cfg.tickers + cfg.indexes))

skip_disk = st.session_state.get("skip_disk_nonce") == st.session_state.nonce
with st.spinner("Loading price and volume history…"):
    result = load_bars(
        all_tickers,
        period,
        cfg.provider,
        st.session_state.nonce,
        skip_disk,
    )

stock_bars = {t: f for t, f in result.bars.items() if t in cfg.tickers}
analyses = analysis_mod.analyze_all(stock_bars, settings)
table = analysis_mod.watchlist_table(analyses)
health = market.analyze_indexes(
    result.bars,
    cfg.indexes,
    pct_threshold=settings.distribution_day_pct,
    window=settings.distribution_day_window,
)

# --------------------------------------------------------------------------- #
# Header
# --------------------------------------------------------------------------- #
as_of = data.last_bar_date(result.bars)
st.markdown(
    f"""
    <div style="margin:0 0 6px 0">
      <div style="font-size:11px;letter-spacing:.12em;text-transform:uppercase;
                  color:{PAL['muted']};font-weight:600">Institutional footprints</div>
      <div style="font-size:34px;line-height:1.15;font-weight:700;
                  color:#2a4db8;letter-spacing:-0.02em">OLGA Trader</div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    f"<div class='olga-callout'>Accumulation and distribution read from price "
    f"range and volume — where a stock closes inside its weekly range, and "
    f"whether the volume behind that close is bigger than normal."
    f"{'  Data through <b>' + as_of.strftime('%B %d, %Y') + '</b>.' if as_of is not None else ''}"
    f"</div>",
    unsafe_allow_html=True,
)

if result.errors:
    st.warning(
        "Could not load: "
        + ", ".join(f"**{t}** ({why})" for t, why in result.errors.items())
    )

borrowed = {t: s for t, s in result.sources.items() if t not in result.proxies}
if borrowed:
    st.caption(
        f"**{ACTIVE_PROVIDER}** does not carry "
        + ", ".join(f"`{t}`" for t in borrowed)
        + ", so real composite index volume is read from "
        + ", ".join(sorted(set(borrowed.values())))
        + " instead. Distribution-day counts are unaffected."
    )

if result.proxies:
    st.warning(
        "Neither **"
        + ACTIVE_PROVIDER
        + "** nor Yahoo could supply "
        + ", ".join(f"`{t}`" for t in result.proxies)
        + " with volume, so an ETF stands in: "
        + ", ".join(
            providers.symbols.proxy_note(t, s) for t, s in result.proxies.items()
        )
        + ". Distribution-day counts on the Market pressure tab are therefore "
        "ETF volume, not composite index volume, and will read differently."
    )

if not analyses:
    st.error("No usable price history. Check the ticker symbols in the sidebar.")
    st.stop()

tab_watchlist, tab_detail, tab_market, tab_ask, tab_method = st.tabs(
    [
        "Watchlist",
        "Stock detail",
        "Market pressure",
        "Ask OLGA",
        "How to read this",
    ]
)

# --------------------------------------------------------------------------- #
# Watchlist
# --------------------------------------------------------------------------- #
with tab_watchlist:
    counts = table["Stance"].value_counts() if not table.empty else pd.Series(dtype=int)
    n_acc = int(counts.get(ACCUMULATION, 0))
    n_dist = int(counts.get(DISTRIBUTION, 0))
    n_flat = int(counts.get(NEUTRAL, 0))

    top = table.iloc[0] if not table.empty else None
    bottom = table.iloc[-1] if not table.empty else None

    cols = st.columns(4)
    cols[0].markdown(
        tile("Under accumulation", f"▲ {n_acc}", f"of {len(table)} on the watchlist"),
        unsafe_allow_html=True,
    )
    cols[1].markdown(
        tile("Under distribution", f"▼ {n_dist}", "closing weak on heavy volume"),
        unsafe_allow_html=True,
    )
    cols[2].markdown(
        tile("Churning", f"• {n_flat}", "neither side in control"),
        unsafe_allow_html=True,
    )
    cols[3].markdown(
        tile(
            "Strongest footprint",
            f"{top['Ticker']}" if top is not None else "—",
            f"grade {top['Grade']} · {top['Read']}" if top is not None else "",
        ),
        unsafe_allow_html=True,
    )

    st.markdown("")
    if bottom is not None and bottom["Stance"] == DISTRIBUTION:
        st.markdown(
            "  ".join(
                [
                    chip(f"▼ Heaviest selling: {bottom['Ticker']}", "critical"),
                    chip(f"grade {bottom['Grade']}", "critical"),
                ]
            )
            + f" <span style='color:{PAL['text_secondary']};font-size:13px'>"
            f"{bottom['Read']}</span>",
            unsafe_allow_html=True,
        )

    display = table.copy()
    display["Stance"] = display["Stance"].map(
        lambda v: f"{STANCE_GLYPH.get(v, '•')} {v.title()}"
    )

    styler = display.style.map(
        lambda g: f"color: {theme.grade_color(str(g), PAL)}; font-weight: 700",
        subset=["Grade"],
    ).format(
        {
            # Score is left to its ProgressColumn config below.
            "Close": "{:,.2f}",
            "Day %": "{:+.2f}%",
            "U/D Ratio": "{:.2f}",
            "Week Close in Range": "{:.0%}",
            "Week Vol x Avg": "{:.2f}x",
        },
        na_rep="—",
    )

    st.dataframe(
        styler,
        width="stretch",
        hide_index=True,
        column_config={
            "Grade": st.column_config.TextColumn(
                "Grade", help="A = heavy accumulation, C = churn, E = heavy distribution"
            ),
            "Score": st.column_config.ProgressColumn(
                "Score", min_value=0, max_value=100, format="%.1f"
            ),
            "U/D Ratio": st.column_config.NumberColumn(
                "U/D Ratio",
                help="Volume on up days ÷ volume on down days over the "
                "up/down window. Above 1.0 favours buyers.",
            ),
            "Week Close in Range": st.column_config.NumberColumn(
                "Week Close in Range",
                help="Where the latest weekly close sits between the week's "
                "low (0%) and high (100%).",
            ),
            "Read": st.column_config.TextColumn("Read", width="medium"),
        },
    )

    st.caption(
        "Ranked by composite score. The table is the full value view — every "
        "number on the charts is here too."
    )

# --------------------------------------------------------------------------- #
# Stock detail
# --------------------------------------------------------------------------- #
with tab_detail:
    order = table["Ticker"].tolist() if not table.empty else sorted(analyses)
    ticker = st.selectbox("Stock", order, index=0)
    item = analyses[ticker]
    week = item.latest_week

    cols = st.columns(5)
    cols[0].markdown(
        tile(
            "Grade",
            f"<span style='color:{theme.grade_color(item.rating.grade, PAL)}'>"
            f"{item.rating.grade}</span>",
            f"score {fmt(item.rating.score, '.1f')} / 100",
        ),
        unsafe_allow_html=True,
    )
    cols[1].markdown(
        tile(
            "Stance",
            f"{STANCE_GLYPH[item.rating.stance]} {item.rating.stance.title()}",
            item.summary.headline,
        ),
        unsafe_allow_html=True,
    )
    cols[2].markdown(
        tile(
            f"U/D ratio ({settings.ud_window}d)",
            fmt(item.ud_latest, ".2f"),
            "above 1.0 means heavier volume on up days",
        ),
        unsafe_allow_html=True,
    )
    cols[3].markdown(
        tile(
            "Latest week close",
            fmt(float(week["RangePos"]) if week is not None else float("nan"), ".0%"),
            "up the week's trading range",
        ),
        unsafe_allow_html=True,
    )
    cols[4].markdown(
        tile(
            "Latest week volume",
            fmt(float(week["VolRatio"]) if week is not None else float("nan"), ".2f")
            + "x",
            "versus its 10-week average",
        ),
        unsafe_allow_html=True,
    )

    if item.week_in_progress:
        st.caption(
            "The newest weekly bar is still forming — its signal is provisional "
            "until the week closes."
        )

    st.plotly_chart(
        charts.weekly_footprint(item, PAL, weeks=weeks_shown),
        width="stretch",
        theme=None,
        config={"displaylogo": False},
    )

    left, right = st.columns(2)
    with left:
        st.plotly_chart(
            charts.ud_ratio_chart(item, PAL),
            width="stretch",
            theme=None,
            config={"displaylogo": False},
        )
    with right:
        st.plotly_chart(
            charts.ad_trend_chart(item, PAL),
            width="stretch",
            theme=None,
            config={"displaylogo": False},
        )

    breakdown, levels_col = st.columns([3, 2])
    with breakdown:
        st.markdown("**What drives the grade**")
        rows = [
            {
                "Component": rating.COMPONENT_LABELS[key],
                "Weight": f"{rating.COMPONENT_WEIGHTS[key]:.0%}",
                "Score": item.rating.components.get(key, float("nan")),
            }
            for key in rating.COMPONENT_WEIGHTS
        ]
        st.dataframe(
            pd.DataFrame(rows),
            hide_index=True,
            width="stretch",
            column_config={
                "Score": st.column_config.ProgressColumn(
                    "Score (0-100)", min_value=0, max_value=100, format="%.0f"
                )
            },
        )
    with levels_col:
        st.markdown("**Levels the stock keeps turning at**")
        if item.levels.empty:
            st.caption("No repeated pivot levels found in the loaded history.")
        else:
            shown = item.levels.copy()
            shown["Role"] = shown["Role"].str.title()
            st.dataframe(
                shown,
                hide_index=True,
                width="stretch",
                column_config={
                    "Level": st.column_config.NumberColumn("Price", format="%.2f"),
                    "Touches": st.column_config.NumberColumn("Pivots", format="%d"),
                },
            )

    with st.expander(f"Weekly table — {ticker}"):
        weekly_view = item.weekly.tail(weeks_shown).iloc[::-1].copy()
        weekly_view.index = weekly_view.index.strftime("%Y-%m-%d")
        weekly_view["RangePos"] = weekly_view["RangePos"] * 100.0
        weekly_view["Signal"] = weekly_view["Signal"].map(
            lambda v: f"{STANCE_GLYPH.get(v, '•')} {v.title()}"
        )
        st.dataframe(
            weekly_view[
                [
                    "Open",
                    "High",
                    "Low",
                    "Close",
                    "PctChange",
                    "RangePos",
                    "Volume",
                    "VolRatio",
                    "Signal",
                ]
            ],
            width="stretch",
            column_config={
                "PctChange": st.column_config.NumberColumn("Week %", format="%+.2f%%"),
                "RangePos": st.column_config.NumberColumn(
                    "Close in range", format="%.0f%%"
                ),
                "VolRatio": st.column_config.NumberColumn("Vol x avg", format="%.2fx"),
                "Volume": st.column_config.NumberColumn("Volume", format="%d"),
            },
        )

# --------------------------------------------------------------------------- #
# Market pressure
# --------------------------------------------------------------------------- #
with tab_market:
    if not health:
        st.info("Add an index such as ^GSPC or ^IXIC to the watchlist file.")
    else:
        cols = st.columns(len(health))
        for col, index_health in zip(cols, health):
            col.markdown(
                tile(
                    index_health.name,
                    f"<span style='color:{PAL[f'status_{index_health.level}']}'>"
                    f"{index_health.count}</span>",
                    f"distribution days in {index_health.window} sessions",
                ),
                unsafe_allow_html=True,
            )

        for index_health in health:
            st.markdown("---")
            if index_health.ticker in result.proxies:
                st.caption(
                    "⚠ "
                    + providers.symbols.proxy_note(
                        index_health.ticker, result.proxies[index_health.ticker]
                    )
                    + " — counts below are ETF volume, not composite."
                )
            elif index_health.ticker in result.sources:
                st.caption(
                    f"Composite volume via {result.sources[index_health.ticker]} — "
                    f"{ACTIVE_PROVIDER} does not carry {index_health.ticker}."
                )
            glyph = "▼" if index_health.level in ("serious", "critical") else "▲"
            st.markdown(
                chip(f"{glyph} {index_health.name}", index_health.level)
                + f" <span style='color:{PAL['text_secondary']};font-size:14px'>"
                f"{index_health.verdict}</span>",
                unsafe_allow_html=True,
            )
            st.plotly_chart(
                charts.market_chart(index_health, PAL),
                width="stretch",
                config={"displaylogo": False},
            )
            with st.expander(f"Distribution days — {index_health.name}"):
                frame = index_health.frame
                flagged = frame[frame["IsDistributionDay"]].tail(30).iloc[::-1]
                if flagged.empty:
                    st.caption("No distribution days in the loaded history.")
                else:
                    view = flagged[["Close", "PctChange", "Volume", "RelVolume"]].copy()
                    view.index = view.index.strftime("%Y-%m-%d")
                    st.dataframe(
                        view,
                        width="stretch",
                        column_config={
                            "Close": st.column_config.NumberColumn(
                                "Close", format="%.2f"
                            ),
                            "PctChange": st.column_config.NumberColumn(
                                "Change", format="%+.2f%%"
                            ),
                            "Volume": st.column_config.NumberColumn(
                                "Volume", format="%d"
                            ),
                            "RelVolume": st.column_config.NumberColumn(
                                "Vol x 50d avg", format="%.2fx"
                            ),
                        },
                    )

# --------------------------------------------------------------------------- #
# Ask OLGA — grounded chat over the live readings
# --------------------------------------------------------------------------- #
with tab_ask:
    llm_ok, llm_note = olga_llm.status()
    st.markdown(
        f"<div class='olga-callout'>Ask about the numbers on this dashboard. "
        f"Gemini answers from the live watchlist, stock detail, and market "
        f"pressure only — analyst breakdown plus a short "
        f"<b>Day-trader take</b>. Not a forecast engine."
        f"<br><span style='color:{PAL['muted']}'>{llm_note}</span></div>",
        unsafe_allow_html=True,
    )
    if not llm_ok:
        st.caption(
            "Gemini is off. Replies use the built-in rule packer until "
            "OLGA_LLM is enabled."
        )

    order = table["Ticker"].tolist() if not table.empty else sorted(analyses)
    if "chat_focus" not in st.session_state or st.session_state.chat_focus not in order:
        st.session_state.chat_focus = order[0] if order else None

    focus_col, clear_col = st.columns([3, 1])
    with focus_col:
        focus = st.selectbox(
            "Focus stock (used when you do not name a ticker)",
            order,
            index=order.index(st.session_state.chat_focus)
            if st.session_state.chat_focus in order
            else 0,
            key="ask_olga_focus",
        )
        st.session_state.chat_focus = focus
    with clear_col:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Clear chat", width="stretch"):
            st.session_state.olga_chat = [
                {
                    "role": "assistant",
                    "content": chat.starter_message(focus, order),
                }
            ]
            st.rerun()

    if "olga_chat" not in st.session_state:
        st.session_state.olga_chat = [
            {
                "role": "assistant",
                "content": chat.starter_message(focus, order),
            }
        ]

    suggest_cols = st.columns(min(3, len(chat.SUGGESTIONS)))
    for i, prompt in enumerate(chat.SUGGESTIONS):
        if suggest_cols[i % len(suggest_cols)].button(
            prompt, key=f"ask_suggest_{i}", width="stretch"
        ):
            st.session_state.olga_pending = prompt

    for message in st.session_state.olga_chat:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    pending = st.session_state.pop("olga_pending", None)
    user_text = st.chat_input("Ask about a grade, buy fit, exit watch, market…")
    question = pending or user_text
    if question:
        st.session_state.olga_chat.append({"role": "user", "content": question})
        reply = chat.answer(
            question,
            analyses=analyses,
            table=table,
            health=health,
            settings=settings,
            focus=st.session_state.chat_focus,
            history=st.session_state.olga_chat,
        )
        st.session_state.olga_chat.append({"role": "assistant", "content": reply})
        st.rerun()

# --------------------------------------------------------------------------- #
# Method
# --------------------------------------------------------------------------- #
with tab_method:
    st.markdown(
        f"""
### The clues, and how each one is measured

**Where the week closes in its range.** A week's high and low are its trading
range; the close tells you who won the week. Institutions buying keep pushing
the close toward the top of the range. Selling pins it near the bottom.
Measured as `(Close − Low) ÷ (High − Low)`, so 100% is a close on the high.

**Volume versus normal.** Volume is the first clue because it points to money
only institutions can wield, but it is imperfect on its own — index funds and
high-frequency traders can be heavy without moving anything. Measured as the
week's volume divided by its trailing {settings.weekly_vol_window}-week average,
excluding the current week so a big bar does not set its own baseline.

**Accumulation** is a week closing at or above
{settings.accumulation_close_position:.0%} up its range on volume at least
{settings.weekly_vol_multiple:.2f}x average. **Distribution** is a week closing at
or below {settings.distribution_close_position:.0%} up its range on that same
heavy volume. Repeated over consecutive weeks, the pattern is the footprint.

A pullback on *light* volume is neither — that is institutions taking a break
from buying, not selling.

**Up/down volume ratio.** Volume on up days divided by volume on down days over
{settings.ud_window} days. 2.0 means twice as much volume traded on up days as
down days; below 1.0 means the heavier trade happened on declines; 1.0 is churn,
where neither side is influencing the stock.

**The A/D line.** Each bar's volume weighted by where it closed in its range,
accumulated. Its slope is the buying-momentum or selling-pressure trend.

**Support and resistance.** Prices where the stock has repeatedly turned,
found by clustering swing pivots. Institutions create support by buying as
others sell, and resistance by selling into strength before the wider market
catches on.

**Distribution days.** A session where the index closes
{settings.distribution_day_pct:.1f}% or worse on higher volume than the day
before. One means nothing. Five or more inside
{settings.distribution_day_window} sessions is an early warning that
institutions are bailing out of the market.

### The grade

Components are blended into a 0-100 score and mapped to a letter — **A** heavy
accumulation, **C** churn, **E** heavy distribution:

| Component | Weight |
|---|---|
| Up/down volume ratio | {rating.COMPONENT_WEIGHTS['ud_ratio']:.0%} |
| Accumulation weeks minus distribution weeks | {rating.COMPONENT_WEIGHTS['weekly_balance']:.0%} |
| A/D line trend | {rating.COMPONENT_WEIGHTS['ad_trend']:.0%} |
| Volume-weighted close strength | {rating.COMPONENT_WEIGHTS['close_strength']:.0%} |

A component that cannot be computed from the loaded history drops out and its
weight is shared among the rest.

### Limits worth knowing

These are inferences from public price and volume, not filings — they suggest
institutional behaviour, they do not prove it. Bars are split-adjusted, and
dividend-adjusted on every provider except Polygon. Bars here come from
**{ACTIVE_PROVIDER}**. Composite index volume is what distribution-day counting
expects; when the provider cannot serve an index with volume, an ETF proxy
stands in and the app says so above. Index volume is revised intraday, so the
newest session can shift.

### Disclaimer

OLGA Trader is an informational tool. It infers possible institutional
activity from public price and volume. It does not prove who is buying or
selling, it does not predict future prices, and it is not investment,
trading, tax, or legal advice. You are solely responsible for your own
decisions and for any money you put at risk.

Andrew Randolph, We Work For You, LLC, and the OLGA Trader app are not
liable for any losses, damages, costs, or claims arising from use of the
app, reliance on its grades, charts, chat answers, or other output, or from
errors, delays, outages, or incomplete data. Past readings are not a
guarantee of future results. If you are not willing to accept that, do not
use the app.
        """
    )

st.caption(
    f"Palette validated for colour-vision deficiency in {MODE} mode · "
    f"provider {ACTIVE_PROVIDER} · "
    f"{len(analyses)} stocks, {len(health)} indexes · "
    f"{len(result.from_cache)} of {len(result.bars)} series served from cache"
)
