"""Grounded Q&A over the live OLGA dashboard readings.

Answers come from the same analyses the UI shows — grades, weeks, A/D, market
pressure, buy-scenario fit, exit watch — not from an external model. That keeps
replies tied to what is on screen and avoids inventing fills or forecasts.

Each substantive reply keeps an analyst-style breakdown and ends with a short
**Day-trader take** in plain trading language.
"""

from __future__ import annotations

import re

import pandas as pd

from .analysis import StockAnalysis
from .config import Settings
from .indicators import ACCUMULATION, DISTRIBUTION, NEUTRAL
from .market import MarketHealth
from .rating import COMPONENT_LABELS, COMPONENT_WEIGHTS

SUGGESTIONS = [
    "What's strongest on the watchlist?",
    "Why is the strongest still churning?",
    "Summarize the focused stock",
    "Buy scenarios for the focused stock",
    "Exit watch for the focused stock",
    "How is market pressure?",
    "What does the grade mean?",
]


def _fmt(value: float, spec: str = ".2f") -> str:
    if value is None or not isinstance(value, (int, float)) or value != value:
        return "—"
    return format(value, spec)


def _with_trader_take(body: str, take: str) -> str:
    """Keep the analyst write-up, then add a short day-trader summary."""
    take = (take or "").strip()
    if not take:
        return body
    return f"{body.rstrip()}\n\n---\n**Day-trader take:** {take}"


def _stock_trader_take(item: StockAnalysis) -> str:
    """One short line a day trader can act on psychologically — not a fill."""
    stance = item.rating.stance
    ud = item.ud_latest
    ad = item.rating.components.get("ad_trend", float("nan"))
    week = item.latest_week
    support = _nearest_support(item)
    resist = _nearest_resistance(item)

    if stance == DISTRIBUTION or (ud == ud and ud < 1.0 and (ad != ad or ad < 50)):
        base = (
            f"**{item.ticker}** — sellers have the edge. "
            "Treat strength as a possible bounce to fade, not a green light to chase."
        )
    elif stance == ACCUMULATION:
        if item.summary.net_weeks == 0:
            base = (
                f"**{item.ticker}** leads on relative strength, but the tape is "
                "quiet — no heavy accumulation weeks yet. Fine to watch; don't "
                "force a breakout entry just because it's #1 on the list."
            )
        else:
            base = (
                f"**{item.ticker}** — buyers still own the footprint. "
                "Look for your entry on a quiet dip into support, not a FOMO chase "
                "into resistance."
            )
    else:
        base = (
            f"**{item.ticker}** is chop / no clear side. "
            "Sit on hands or trade smaller until volume picks a direction."
        )

    extras: list[str] = []
    if week is not None:
        vol = float(week["VolRatio"])
        signal = str(week["Signal"])
        if signal == DISTRIBUTION:
            extras.append("This week already smells like distribution — respect it.")
        elif vol == vol and vol < 1.0 and signal != ACCUMULATION:
            extras.append("Latest week is light volume — pause, not panic.")
    if support is not None and resist is not None:
        extras.append(
            f"Map: support ~{_fmt(support)}, resistance ~{_fmt(resist)}."
        )
    if extras:
        return base + " " + " ".join(extras)
    return base


def _pick_ticker(
    question: str,
    analyses: dict[str, StockAnalysis],
    focus: str | None,
) -> str | None:
    upper = question.upper()
    # Longer tickers first so "GOOGL" wins over a stray "GOOG" substring habit.
    for ticker in sorted(analyses, key=len, reverse=True):
        if re.search(rf"\b{re.escape(ticker)}\b", upper):
            return ticker
    if focus and focus in analyses:
        return focus
    return None


def _nearest_support(item: StockAnalysis) -> float | None:
    if item.levels.empty:
        return None
    supports = item.levels.loc[item.levels["Role"] == "support", "Level"]
    if supports.empty:
        return None
    close = item.last_close
    return float(min(supports, key=lambda lvl: abs(float(lvl) - close)))


def _nearest_resistance(item: StockAnalysis) -> float | None:
    if item.levels.empty:
        return None
    resists = item.levels.loc[item.levels["Role"] == "resistance", "Level"]
    if resists.empty:
        return None
    close = item.last_close
    return float(min(resists, key=lambda lvl: abs(float(lvl) - close)))


def _stock_blurb(item: StockAnalysis, settings: Settings) -> str:
    week = item.latest_week
    ad = item.rating.components.get("ad_trend", float("nan"))
    support = _nearest_support(item)
    resist = _nearest_resistance(item)
    lines = [
        f"**{item.ticker}** — grade **{item.rating.grade}** "
        f"(score {_fmt(item.rating.score, '.1f')}), stance "
        f"**{item.rating.stance}**.",
        f"{item.summary.headline.capitalize()}. "
        f"Close {_fmt(item.last_close)} ({_fmt(item.change_pct, '+.2f')}% today).",
        f"U/D ratio ({settings.ud_window}d): **{_fmt(item.ud_latest)}** "
        f"(above 1.0 favours buyers). "
        f"A/D trend component: **{_fmt(ad, '.1f')}** "
        f"(above 50 is a rising footprint).",
    ]
    if week is not None:
        provisional = " — week still forming" if item.week_in_progress else ""
        lines.append(
            f"Latest week: signal **{week['Signal']}**, "
            f"close {_fmt(float(week['RangePos']) * 100, '.0f')}% up its range, "
            f"volume {_fmt(float(week['VolRatio']))}× average{provisional}."
        )
    lines.append(
        f"Lookback ({item.summary.lookback_weeks}w): "
        f"{item.summary.accumulation_weeks} accumulation / "
        f"{item.summary.distribution_weeks} distribution weeks "
        f"(net {item.summary.net_weeks:+d})."
    )
    level_bits = []
    if support is not None:
        level_bits.append(f"nearest support ~{_fmt(support)}")
    if resist is not None:
        level_bits.append(f"nearest resistance ~{_fmt(resist)}")
    if level_bits:
        lines.append("Levels: " + "; ".join(level_bits) + ".")
    return "\n\n".join(lines)


def _grade_breakdown(item: StockAnalysis) -> str:
    rows = []
    for key, weight in COMPONENT_WEIGHTS.items():
        score = item.rating.components.get(key, float("nan"))
        rows.append(
            f"- **{COMPONENT_LABELS[key]}** ({weight:.0%}): {_fmt(score, '.1f')}"
        )
    body = (
        f"**{item.ticker}** grade **{item.rating.grade}** is the blend of:\n"
        + "\n".join(rows)
        + "\n\nA = heavy accumulation, C = churn, E = heavy distribution. "
        "This is an inference from public price and volume, not a forecast."
    )
    letter = item.rating.grade.rstrip("+-")
    if letter in {"A", "B"}:
        take = (
            f"Grade **{item.rating.grade}** ≈ buyers still matter — trade it like "
            "a long-bias name, but only on *your* trigger."
        )
    elif letter == "C":
        take = (
            f"Grade **{item.rating.grade}** ≈ chop. Size down or skip until the "
            "tape picks a side."
        )
    else:
        take = (
            f"Grade **{item.rating.grade}** ≈ sellers / weak footprint. "
            "Longs are fighting the tape."
        )
    return _with_trader_take(body, take)


def _watchlist_summary(table: pd.DataFrame) -> str:
    if table.empty:
        return "The watchlist has no usable readings right now."
    counts = table["Stance"].value_counts()
    n_acc = int(counts.get(ACCUMULATION, 0))
    n_dist = int(counts.get(DISTRIBUTION, 0))
    n_flat = int(counts.get(NEUTRAL, 0))
    top = table.iloc[0]
    bottom = table.iloc[-1]
    lines = [
        f"Watchlist: **{len(table)}** names — "
        f"▲ {n_acc} accumulation, ▼ {n_dist} distribution, • {n_flat} churning.",
        f"Strongest footprint: **{top['Ticker']}** "
        f"(grade {top['Grade']}, score {_fmt(float(top['Score']), '.1f')}) — "
        f"{top['Read']}.",
    ]
    if bottom["Ticker"] != top["Ticker"]:
        lines.append(
            f"Weakest on the list: **{bottom['Ticker']}** "
            f"(grade {bottom['Grade']}) — {bottom['Read']}."
        )
    under_acc = table.loc[table["Stance"] == ACCUMULATION, "Ticker"].tolist()
    if under_acc:
        lines.append("Under accumulation: " + ", ".join(f"**{t}**" for t in under_acc) + ".")
    body = "\n\n".join(lines)
    take = (
        f"Hunt longs near **{top['Ticker']}** first; treat **{bottom['Ticker']}** "
        f"as the avoid / short-bias end of the board. "
        f"{n_acc} names look buyer-led, {n_dist} look seller-led, {n_flat} are chop."
    )
    return _with_trader_take(body, take)


def _explain_rank_vs_read(
    item: StockAnalysis,
    table: pd.DataFrame,
    settings: Settings,
) -> str:
    """Explain 'strongest but churning' — rank/score vs weekly headline vs stance."""
    if table.empty:
        return "The watchlist has no usable readings right now."

    rank_rows = table.reset_index(drop=True)
    matches = rank_rows.index[rank_rows["Ticker"] == item.ticker].tolist()
    rank = int(matches[0]) + 1 if matches else None
    top = rank_rows.iloc[0]
    score = item.rating.score
    stance = item.rating.stance
    headline = item.summary.headline
    ad = item.rating.components.get("ad_trend", float("nan"))
    weekly = item.rating.components.get("weekly_balance", float("nan"))
    ud_comp = item.rating.components.get("ud_ratio", float("nan"))
    close_comp = item.rating.components.get("close_strength", float("nan"))

    lines = [
        f"**{item.ticker}** can be the **strongest footprint** and still show a "
        f"**“{headline}”** read — those labels measure different things.",
        "**1. “Strongest” means highest composite score on this watchlist**, "
        "not “most accumulation weeks.” "
        + (
            f"**{item.ticker}** is rank **#{rank}** of {len(rank_rows)} "
            f"(grade {item.rating.grade}, score {_fmt(score, '.1f')}). "
            if rank is not None
            else f"**{item.ticker}** scores {_fmt(score, '.1f')} "
            f"(grade {item.rating.grade}). "
        )
        + (
            f"Next on the board is **{rank_rows.iloc[1]['Ticker']}** "
            f"at {_fmt(float(rank_rows.iloc[1]['Score']), '.1f')}."
            if rank == 1 and len(rank_rows) > 1
            else f"The list leader is **{top['Ticker']}** "
            f"at {_fmt(float(top['Score']), '.1f')}."
            if rank != 1
            else ""
        ),
        "**2. The “churning” line is only the weekly-signal headline.** "
        f"Over the last {item.summary.lookback_weeks} weeks OLGA counted "
        f"**{item.summary.accumulation_weeks}** labeled accumulation week(s) and "
        f"**{item.summary.distribution_weeks}** distribution week(s) "
        f"(net {item.summary.net_weeks:+d}). "
        "A labeled week needs a decisive close *and* heavy volume. "
        "Quiet weeks, or weeks that close in the middle of the range, stay "
        "neutral — so the Read says churning even when other components look fine.",
        "**3. Stance and grade use the full blend**, not that headline alone. "
        f"Score ≥ 56 → accumulation stance; ≤ 42 → distribution; otherwise "
        f"neutral unless a multi-week streak breaks the tie. "
        f"**{item.ticker}**’s stance is **{stance}** at score {_fmt(score, '.1f')}.",
        "**What is carrying the score right now:**\n"
        f"- Up/down volume: **{_fmt(ud_comp, '.1f')}** "
        f"(live U/D {_fmt(item.ud_latest)} over {settings.ud_window}d)\n"
        f"- Weekly accumulation balance: **{_fmt(weekly, '.1f')}** "
        f"(this is the flat/churning piece)\n"
        f"- A/D line trend: **{_fmt(ad, '.1f')}**\n"
        f"- Volume-weighted close strength: **{_fmt(close_comp, '.1f')}**",
    ]

    if stance == ACCUMULATION and item.summary.net_weeks == 0:
        lines.append(
            f"**Analyst note:** **{item.ticker}** leads the board because buyers "
            "still show up in the U/D / A/D / close-strength mix, not because the "
            "last stretch printed a string of heavy accumulation weeks. The "
            "watchlist tile’s “churning” subtitle is that weekly-balance footnote — "
            "it does **not** cancel the higher composite score."
        )
        trader = (
            f"**{item.ticker}** is the relative-strength leader on *this* list, "
            "not a clean 'load the boat' setup. No heavy green accumulation weeks "
            "yet — watch it, don't chase just because it's #1."
        )
    elif stance == NEUTRAL and rank == 1:
        lines.append(
            f"**Analyst note:** **{item.ticker}** is only “strongest” **relative "
            "to this watchlist** — everyone else scores lower. Stance is still "
            "churn/neutral, so it is the least-weak name on the board, not a "
            "clean accumulation setup."
        )
        trader = (
            f"**{item.ticker}** is 'best of a soft board,' not a high-conviction "
            "long. Prefer wait-and-see over forcing a day trade."
        )
    else:
        lines.append(
            f"**Analyst note:** Rank, Read, and Stance can disagree for a day. "
            "Trust the score for ordering the list; trust the weekly Read for "
            "whether institutions just printed heavy labeled weeks; trust Stance "
            "for the one-word summary of the blend."
        )
        trader = _stock_trader_take(item)

    return _with_trader_take(
        "\n\n".join(line for line in lines if line.strip()),
        trader,
    )


def _market_summary(health: list[MarketHealth]) -> str:
    if not health:
        return (
            "No index readings loaded. Add ^GSPC / ^IXIC (or equivalents) to "
            "the index list to see market pressure."
        )
    lines = [
        "Market pressure counts **distribution days** — index sessions that "
        "close weak on heavier volume than the day before. Five or more inside "
        "the rolling window is the warning level."
    ]
    for h in health:
        lines.append(
            f"- **{h.name}**: {h.count} distribution day(s) in {h.window} "
            f"sessions — **{h.level}**. {h.verdict}"
        )
    max_count = max(h.count for h in health)
    worst = max(health, key=lambda h: h.count)
    if max_count >= 5:
        take = (
            f"Indexes are leaking — **{worst.name}** at {worst.count} distribution "
            "days. Be picky on longs; favor relative strength and tighter risk."
        )
    elif max_count >= 3:
        take = (
            "Pressure is building but not a full alarm. Keep size honest and "
            "don't ignore weak closes on rising volume."
        )
    else:
        take = (
            "Index distribution is light — tape backdrop is OK for trading "
            "long setups that already look constructive."
        )
    return _with_trader_take("\n".join(lines), take)


def _methodology_blurb(settings: Settings) -> str:
    body = (
        "OLGA reads **institutional footprints** from where a stock closes "
        "inside its weekly range and whether volume is heavy.\n\n"
        f"- **Accumulation week**: close ≥ {settings.accumulation_close_position:.0%} "
        f"up the week's range on volume ≥ {settings.weekly_vol_multiple:.2f}× average.\n"
        f"- **Distribution week**: close ≤ {settings.distribution_close_position:.0%} "
        f"up the range on that same heavy volume.\n"
        f"- A quiet pullback (light volume) is **neither** — institutions pausing, "
        "not selling.\n"
        f"- **U/D ratio**: up-day volume ÷ down-day volume over {settings.ud_window} days; "
        "above 1.0 favours buyers.\n"
        "- **A/D line**: cumulative volume weighted by close-in-range; its slope "
        "is the pressure trend.\n"
        "- **Grade A–E**: blend of U/D, weekly accumulation vs distribution, "
        "A/D trend, and volume-weighted close strength.\n\n"
        "Nothing here is a fill, stop, or return forecast — and it is not "
        "investment advice."
    )
    take = (
        "Heavy up-close + big volume = buyers stepping in. Heavy down-close + "
        "big volume = sellers dumping. Quiet dip = pause, not necessarily sell. "
        "Grade is the scorecard; your trigger and stop are still your job."
    )
    return _with_trader_take(body, take)


def _buy_scenarios(item: StockAnalysis, settings: Settings) -> str:
    """Rank the three buy scenarios against the live footprint."""
    # Import lazily so scenario stays a CLI entry point.
    from scenario import _rank_scenarios  # noqa: WPS433

    stance = item.rating.stance
    ad = item.rating.components.get("ad_trend", float("nan"))
    constructive = stance == ACCUMULATION and (ad != ad or ad >= 50.0)
    header = _stock_blurb(item, settings)
    if constructive:
        contrast = (
            "Live read is **constructive**. Ideal buy: soft week on light volume "
            "that is not labeled distribution, holding support, while grade and "
            "A/D stay constructive. Trap: bounce after a heavy distribution week "
            "or buying into resistance while indexes stack distribution days."
        )
    else:
        contrast = (
            "Live read is **not constructive**. Trap risk is elevated — a bounce "
            "can look like a dip-buy while distribution or churn still owns the "
            "footprint."
        )
    ranked = _rank_scenarios(item)
    body = ["**Buy scenarios ranked by OLGA fit now:**"]
    best_name, best_fit = ranked[0][1], ranked[0][2]
    for rank, name, fit, why in ranked:
        body.append(f"{rank}. **{name}** — {fit}\n   {why}")
    body.append(
        "Out of scope: paper fills, stops, sizing, and which scenario made money."
    )
    if "Best" in best_fit or "Good — live" in best_fit:
        take = (
            f"Best fit right now: **{best_name}** ({best_fit}). "
            "That's the playbook lane — still wait for *your* candle/level trigger."
        )
    elif constructive:
        take = (
            f"**{item.ticker}** footprint is OK, but no clean live buy lane yet "
            f"(top fit: {best_name}). Patience > forcing an entry."
        )
    else:
        take = (
            f"Don't invent a long in **{item.ticker}** from a bounce alone — "
            "footprint isn't constructive. Wait or look elsewhere."
        )
    return _with_trader_take(
        "\n\n".join([header, contrast, "\n".join(body)]),
        take,
    )


def _exit_watch_answer(
    item: StockAnalysis,
    settings: Settings,
    health: list[MarketHealth],
) -> str:
    from scenario import _exit_watch  # noqa: WPS433

    checks = _exit_watch(item, settings, health)
    mark = {"clear": "clear", "watch": "watch", "triggered": "triggered"}
    lines = [
        f"**Exit watch for {item.ticker}** — thesis-break cues if you already "
        "bought on accumulation. Not a promise you sell before any loss.",
        _stock_blurb(item, settings),
    ]
    for i, check in enumerate(checks, start=1):
        lines.append(
            f"{i}. **{check.name}** — `{mark.get(check.status, check.status)}`\n"
            f"   Rule: {check.rule}\n"
            f"   Now: {check.now}"
        )
    triggered = sum(1 for c in checks if c.status == "triggered")
    watching = sum(1 for c in checks if c.status == "watch")
    if triggered:
        summary = (
            f"**Summary:** {triggered} triggered, {watching} on watch — "
            "the buy reason is under pressure."
        )
        take = (
            f"**{item.ticker}** — thesis cracks are showing ({triggered} red flag"
            f"{'s' if triggered != 1 else ''}). Tighten risk or scratch; don't "
            "add size into weakness."
        )
    elif watching:
        summary = (
            f"**Summary:** nothing fully triggered, {watching} on watch — "
            "thesis intact but not risk-free."
        )
        take = (
            f"**{item.ticker}** — still okay to hold a plan, but keep the stop "
            "honest. Something is on the radar."
        )
    else:
        summary = "**Summary:** all clear vs these invalidation rules right now."
        take = (
            f"**{item.ticker}** — no exit alarm from OLGA's checklist. "
            "Manage with your own stop under support anyway."
        )
    lines.append(summary)
    lines.append(
        "OLGA does not set a hard price stop. Pair this with a max loss you "
        "decide in advance (often under nearest support)."
    )
    return _with_trader_take("\n\n".join(lines), take)


def _help_text(tickers: list[str]) -> str:
    names = ", ".join(tickers[:12]) + ("…" if len(tickers) > 12 else "")
    return (
        "I answer from the **live OLGA readings** on your watchlist — grades, "
        "stances, weeks, A/D, levels, market pressure, buy scenarios, and exit "
        "watch. I do not place trades or invent P&L.\n\n"
        "Try asking about a ticker by name, or:\n"
        "- What's strongest / weakest on the watchlist?\n"
        "- Summarize MSFT (or another name)\n"
        "- Buy scenarios / should I buy the pullback?\n"
        "- Exit watch / when would the thesis break?\n"
        "- How is market pressure?\n"
        "- What does the grade mean?\n\n"
        f"On the board now: {names}."
    )


ASK_OLGA_SYSTEM_PROMPT = """You are Ask OLGA, the in-app analyst for OLGA Trader.

You explain the live dashboard. You do not invent prices, grades, stances,
volume, P&L, fills, or forecasts. If a figure is not in the DASHBOARD CONTEXT
block, say you do not have it.

Voice:
- Write as a professional analyst first (short, precise, named metrics).
- End every substantive answer with a section headed exactly:
  **Day-trader take:**
  One or two sentences in plain trading language (chase / wait / chop /
  sellers-in-control). Not a trade ticket.

Hard rules:
- Rank ("strongest") is highest composite score on this watchlist.
- The "Read" / "churning" line is only the weekly-signal headline
  (labeled accumulation vs distribution weeks). Stance and grade use the
  full blend (U/D, weekly balance, A/D trend, close strength).
- Score ≥ 56 → accumulation stance; ≤ 42 → distribution; otherwise
  neutral unless a multi-week streak breaks the tie.
- A quiet (light-volume) pullback is a pause, not distribution.
- Never claim OLGA placed an order or computed a stop. Not investment advice.
- If the user asks something outside OLGA (news, earnings gossip, options
  greeks), refuse and steer back to footprint readings.
- Prefer the focused ticker when the user does not name one.
"""


def dashboard_context(
    *,
    analyses: dict[str, StockAnalysis],
    table: pd.DataFrame,
    health: list[MarketHealth],
    settings: Settings,
    focus: str | None = None,
) -> str:
    """Pack the live UI readings into the only facts Gemini may use."""
    lines: list[str] = [
        f"Focus ticker: {focus or 'none'}",
        f"Thresholds: accumulation close ≥ {settings.accumulation_close_position:.0%} "
        f"of weekly range; distribution close ≤ {settings.distribution_close_position:.0%}; "
        f"heavy volume ≥ {settings.weekly_vol_multiple:.2f}×; "
        f"U/D window {settings.ud_window}d; weekly lookback "
        f"{settings.signal_lookback_weeks}w.",
        "",
        "WATCHLIST (ranked by composite score, strongest first):",
    ]
    if table.empty:
        lines.append("(empty)")
    else:
        for i, row in table.reset_index(drop=True).iterrows():
            lines.append(
                f"{i + 1}. {row['Ticker']}: grade={row['Grade']} "
                f"score={_fmt(float(row['Score']), '.1f')} stance={row['Stance']} "
                f"close={_fmt(float(row['Close']))} day%={_fmt(float(row['Day %']), '+.2f')} "
                f"U/D={_fmt(float(row['U/D Ratio']))} "
                f"week_close_in_range={_fmt(float(row['Week Close in Range']) * 100, '.0f')}% "
                f"week_vol_x={_fmt(float(row['Week Vol x Avg']))} "
                f"acc_weeks={int(row['Acc Weeks'])} dist_weeks={int(row['Dist Weeks'])} "
                f"streak={row['Streak']} read={row['Read']}"
            )

    lines.append("")
    lines.append("MARKET PRESSURE:")
    if not health:
        lines.append("(no indexes loaded)")
    else:
        for h in health:
            lines.append(
                f"- {h.name} ({h.ticker}): {h.count} distribution days / "
                f"{h.window} sessions, level={h.level}. {h.verdict}"
            )

    focus_item = analyses.get(focus) if focus else None
    if focus_item is not None:
        lines.append("")
        lines.append(f"FOCUSED STOCK DETAIL ({focus_item.ticker}):")
        lines.append(_stock_blurb(focus_item, settings).replace("**", ""))
        lines.append(_grade_breakdown(focus_item).split("---")[0].strip().replace("**", ""))
        try:
            from scenario import _exit_watch, _rank_scenarios

            lines.append("Buy-scenario ranks (OLGA methodology fit, not P&L):")
            for rank, name, fit, why in _rank_scenarios(focus_item):
                lines.append(f"  {rank}. {name} | {fit} | {why}")
            lines.append("Exit watch:")
            for check in _exit_watch(focus_item, settings, health):
                lines.append(f"  - {check.name}: {check.status}. {check.rule} Now: {check.now}")
        except Exception:
            pass

    lines.append("")
    lines.append("PER-NAME COMPONENTS:")
    for ticker, item in analyses.items():
        comps = ", ".join(
            f"{k}={_fmt(v, '.1f')}" for k, v in item.rating.components.items()
        )
        lines.append(
            f"- {ticker}: grade={item.rating.grade} score={_fmt(item.rating.score, '.1f')} "
            f"stance={item.rating.stance} U/D={_fmt(item.ud_latest)} {comps} "
            f"headline={item.summary.headline}"
        )
    return "\n".join(lines)


def _history_for_llm(
    question: str, history: list[dict[str, str]] | None
) -> list[tuple[str, str]]:
    prior: list[tuple[str, str]] = []
    for message in history or []:
        role = message.get("role")
        text = (message.get("content") or "").strip()
        if role in {"user", "assistant"} and text:
            prior.append((role, text))
    if prior and prior[-1][0] == "user" and prior[-1][1] == question.strip():
        prior = prior[:-1]
    if prior and prior[0][0] == "assistant":
        prior = prior[1:]
    return prior[-6:]


def answer(
    question: str,
    *,
    analyses: dict[str, StockAnalysis],
    table: pd.DataFrame,
    health: list[MarketHealth],
    settings: Settings,
    focus: str | None = None,
    history: list[dict[str, str]] | None = None,
) -> str:
    """Reply from Gemini when available; otherwise the rule-based packer."""
    from . import llm

    q = (question or "").strip()
    if not q:
        return _help_text(list(analyses))

    if llm.enabled():
        try:
            snapshot = dashboard_context(
                analyses=analyses,
                table=table,
                health=health,
                settings=settings,
                focus=focus,
            )
            user = (
                f"DASHBOARD CONTEXT (only source of numbers):\n{snapshot}\n\n"
                f"USER QUESTION:\n{q}"
            )
            return llm.generate(
                system=ASK_OLGA_SYSTEM_PROMPT,
                user=user,
                history=_history_for_llm(q, history),
            )
        except Exception as exc:
            fallback = answer_rules(
                q,
                analyses=analyses,
                table=table,
                health=health,
                settings=settings,
                focus=focus,
            )
            detail = str(exc)
            if "SERVICE_DISABLED" in detail or "has not been used" in detail:
                note = (
                    "Gemini is wired to Vertex project **olga-trader**, but "
                    "the Vertex AI / Agent Platform API is not enabled there yet. "
                    "Showing OLGA's built-in reading."
                )
            else:
                note = "Gemini unavailable. Showing OLGA's built-in reading."
            return fallback + f"\n\n_{note}_"

    return answer_rules(
        q,
        analyses=analyses,
        table=table,
        health=health,
        settings=settings,
        focus=focus,
    )


def answer_rules(
    question: str,
    *,
    analyses: dict[str, StockAnalysis],
    table: pd.DataFrame,
    health: list[MarketHealth],
    settings: Settings,
    focus: str | None = None,
) -> str:
    """Return a markdown reply grounded in the current dashboard state."""
    q = (question or "").strip()
    if not q:
        return _help_text(list(analyses))

    lower = q.lower()
    ticker = _pick_ticker(q, analyses, focus)

    if any(
        k in lower
        for k in (
            "help",
            "what can you",
            "what do you",
            "how do i use",
        )
    ):
        return _help_text(list(analyses))

    if any(
        k in lower
        for k in (
            "what does the grade",
            "what is a grade",
            "what is grade",
            "methodology",
            "how does olga",
            "what is accumulation",
            "what is distribution",
            "how to read",
        )
    ):
        return _methodology_blurb(settings)

    if any(
        k in lower
        for k in (
            "market pressure",
            "distribution day",
            "distribution days",
            "index",
            "s&p",
            "nasdaq",
            "market health",
        )
    ) and ticker is None:
        return _market_summary(health)

    why_rank = any(
        k in lower
        for k in (
            "why",
            "even though",
            "despite",
            "but churn",
            "while churn",
            "although",
            "how can",
            "contradict",
        )
    ) and any(
        k in lower
        for k in (
            "strongest",
            "stronger",
            "rank",
            "churn",
            "neutral",
            "top",
            "highest",
            "leading",
            "lead the",
        )
    )

    if why_rank:
        subject = ticker or (str(table.iloc[0]["Ticker"]) if not table.empty else None)
        if subject and subject in analyses:
            return _explain_rank_vs_read(analyses[subject], table, settings)

    if any(
        k in lower
        for k in (
            "watchlist",
            "strongest",
            "weakest",
            "heaviest",
            "overview",
            "who's accumulating",
            "who is accumulating",
            "under accumulation",
        )
    ) and not any(k in lower for k in ("exit", "buy scenario", "summarize", "why")):
        # "summarize the watchlist" still hits here via watchlist keyword.
        return _watchlist_summary(table)

    if ticker is None:
        return (
            _help_text(list(analyses))
            + "\n\nI could not tell which stock you mean — name a ticker from "
            "the watchlist, or set the focus stock above."
        )

    item = analyses[ticker]

    if any(
        k in lower
        for k in (
            "exit",
            "sell",
            "invalidat",
            "thesis break",
            "when to get out",
            "stop",
        )
    ):
        return _exit_watch_answer(item, settings, health)

    if any(
        k in lower
        for k in (
            "buy scenario",
            "buy scenarios",
            "should i buy",
            "pullback",
            "accumulation week",
            "trap",
            "entry",
        )
    ):
        return _buy_scenarios(item, settings)

    if any(
        k in lower
        for k in (
            "breakdown",
            "what drives",
            "components",
            "why the grade",
            "grade mean for",
        )
    ):
        return _grade_breakdown(item)

    if any(k in lower for k in ("market pressure", "distribution day")):
        return (
            _stock_blurb(item, settings)
            + "\n\n"
            + _market_summary(health)
        )

    # Default for a named (or focused) ticker: full stock summary.
    return _stock_blurb(item, settings) + "\n\n" + _grade_breakdown(item)


def starter_message(focus: str | None, tickers: list[str]) -> str:
    from . import llm

    focus_note = f" Focus is **{focus}**." if focus else ""
    _, llm_note = llm.status()
    return (
        "Ask about anything on the dashboard — a ticker's grade, buy-scenario "
        f"fit, exit watch, or market pressure.{focus_note}\n\n"
        "Replies use an **analyst breakdown**, then a short **Day-trader take** "
        f"in plain trading language. Engine: {llm_note}.\n\n"
        + _help_text(tickers)
    )
