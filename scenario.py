"""Buy-scenario walkthrough for any ticker, using OLGA's live readings.

Prints Stock-detail style metrics, ranks three buy scenarios by how well they
fit the methodology *right now*, then an Exit watch checklist for when the buy
thesis has broken. Stays inside what OLGA measures — no fills, P&L, or
backtests.

Run with:
  .venv\\Scripts\\python scenario.py MSFT
  .venv\\Scripts\\python scenario.py GOOGL --period 2y
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass

from olga import analysis, config, data, indicators, market, providers
from olga.config import Settings, normalize_ticker
from olga.indicators import ACCUMULATION, DISTRIBUTION
from olga.market import MarketHealth

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def _fmt(value: float, spec: str = ".2f") -> str:
    if value != value:  # NaN
        return "—"
    return format(value, spec)


@dataclass
class ExitCheck:
    """One invalidation rule with a live status."""

    name: str
    status: str  # clear | watch | triggered
    rule: str
    now: str


def _nearest_support(item: analysis.StockAnalysis) -> float | None:
    if item.levels.empty:
        return None
    supports = item.levels.loc[item.levels["Role"] == "support", "Level"]
    if supports.empty:
        return None
    close = item.last_close
    return float(min(supports, key=lambda lvl: abs(float(lvl) - close)))


def _exit_watch(
    item: analysis.StockAnalysis,
    settings: Settings,
    market_health: list[MarketHealth],
) -> list[ExitCheck]:
    """Build the Exit watch checklist from live OLGA readings."""
    week = item.latest_week
    ad = item.rating.components.get("ad_trend", float("nan"))
    ud = item.ud_latest
    support = _nearest_support(item)
    vol_mult = settings.weekly_vol_multiple
    dist_pos = settings.distribution_close_position

    # 1) Quiet dip turns into heavy selling (distribution week)
    if week is None:
        dist_status, dist_now = "watch", "No weekly bar yet."
    else:
        signal = str(week["Signal"])
        range_pos = float(week["RangePos"])
        vol = float(week["VolRatio"])
        provisional = " (week still forming)" if item.week_in_progress else ""
        dist_now = (
            f"signal={signal}, close {_fmt(range_pos * 100, '.0f')}% up range, "
            f"vol {_fmt(vol)}x avg{provisional}."
        )
        if signal == DISTRIBUTION:
            dist_status = "triggered"
        elif (
            vol == vol
            and vol >= vol_mult * 0.9
            and range_pos == range_pos
            and range_pos <= dist_pos + 0.1
        ):
            dist_status = "watch"
        else:
            dist_status = "clear"

    # 2) Support gives way on real volume
    if support is None:
        support_status = "watch"
        support_now = "No support cluster available."
        support_rule = (
            f"Close clearly below nearest support on volume ≥ {vol_mult:.2f}x "
            "normal (heavy break, not a quiet poke)."
        )
    else:
        close = item.last_close
        below = close < support
        daily_rvol = item.daily_rel_volume
        heavy = daily_rvol == daily_rvol and daily_rvol >= vol_mult
        pct_from = (close / support - 1.0) * 100.0
        support_now = (
            f"close {close:.2f} vs support {support:.2f} "
            f"({_fmt(pct_from, '+.1f')}%), daily rvol {_fmt(daily_rvol)}x."
        )
        support_rule = (
            f"Close clearly below support ~{support:.2f} on volume ≥ "
            f"{vol_mult:.2f}x normal (heavy break, not a quiet poke)."
        )
        if below and heavy:
            support_status = "triggered"
        elif below or (
            close < support * 1.01
            and daily_rvol == daily_rvol
            and daily_rvol >= 1.0
        ):
            support_status = "watch"
        else:
            support_status = "clear"

    # 3) Footprint flips (stance / A/D / U/D)
    stance = item.rating.stance
    footprint_bits = [
        f"stance={stance}",
        f"A/D component={_fmt(ad, '.1f')}",
        f"U/D={_fmt(ud)}",
    ]
    footprint_now = ", ".join(footprint_bits) + "."
    footprint_rule = (
        "Stance leaves accumulation, A/D component slides under 50, "
        "or U/D ratio drops to/under 1.0."
    )
    bad_stance = stance == DISTRIBUTION
    soft_stance = stance != ACCUMULATION
    ad_triggered = ad == ad and ad < 45.0
    ad_watch = ad == ad and ad < 50.0
    ud_triggered = ud == ud and ud < 1.0
    ud_watch = ud == ud and ud < 1.15
    if bad_stance or (ad_triggered and ud_triggered) or (
        soft_stance and ad_triggered
    ):
        footprint_status = "triggered"
    elif soft_stance or ad_watch or ud_watch:
        footprint_status = "watch"
    else:
        footprint_status = "clear"

    # 4) Index distribution stacking
    if not market_health:
        mkt_status = "watch"
        mkt_now = "No index history loaded."
    else:
        parts = [
            f"{h.name} {h.count}/{h.window} ({h.level})" for h in market_health
        ]
        mkt_now = "; ".join(parts) + "."
        max_count = max(h.count for h in market_health)
        # Five or more in the window is the article's warning level.
        if max_count >= 5:
            mkt_status = "triggered"
        elif max_count >= 3:
            mkt_status = "watch"
        else:
            mkt_status = "clear"

    return [
        ExitCheck(
            name="Quiet dip turns into heavy selling",
            status=dist_status,
            rule=(
                f"A distribution week: close ≤ {dist_pos:.0%} up the week's range "
                f"on volume ≥ {vol_mult:.2f}x average."
            ),
            now=dist_now,
        ),
        ExitCheck(
            name="Support gives way on real volume",
            status=support_status,
            rule=support_rule,
            now=support_now,
        ),
        ExitCheck(
            name="Footprint flips (grade / A/D / U/D)",
            status=footprint_status,
            rule=footprint_rule,
            now=footprint_now,
        ),
        ExitCheck(
            name="Index distribution keeps stacking",
            status=mkt_status,
            rule=(
                f"Five or more index distribution days inside "
                f"{settings.distribution_day_window} sessions "
                "(market backdrop fighting the long)."
            ),
            now=mkt_now,
        ),
    ]


def _print_exit_watch(checks: list[ExitCheck]) -> None:
    mark = {"clear": "[ ]", "watch": "[~]", "triggered": "[X]"}
    print()
    print("=== EXIT WATCH ===")
    print(
        "If you already bought on accumulation, treat these as thesis-break "
        "cues — not a promise you sell before any loss. "
        "[ ] clear  [~] watch  [X] triggered"
    )
    for i, check in enumerate(checks, start=1):
        print(f"{i}. {mark.get(check.status, '[?]')} {check.name} — {check.status}")
        print(f"   rule: {check.rule}")
        print(f"   now:  {check.now}")
    triggered = sum(1 for c in checks if c.status == "triggered")
    watching = sum(1 for c in checks if c.status == "watch")
    if triggered:
        print(
            f"Summary: {triggered} triggered, {watching} on watch — "
            "the buy reason is under pressure; re-check before holding."
        )
    elif watching:
        print(
            f"Summary: nothing fully triggered, {watching} on watch — "
            "thesis intact but not risk-free."
        )
    else:
        print("Summary: all clear vs these invalidation rules right now.")
    print(
        "Note: OLGA does not set a hard price stop. Pair this checklist with "
        "a max loss you decide in advance (often under nearest support)."
    )


def _rank_scenarios(item: analysis.StockAnalysis) -> list[tuple[int, str, str, str]]:
    """Rank the three buy scenarios by current OLGA fit.

    Returns (rank, name, fit label, why) sorted best-first.
    """
    grade = item.rating.grade
    stance = item.rating.stance
    ad = item.rating.components.get("ad_trend", float("nan"))
    week = item.latest_week
    light_pullback = False
    accumulation_week = False
    if week is not None:
        vol = float(week["VolRatio"])
        signal = str(week["Signal"])
        light_pullback = vol == vol and vol < 1.0 and signal != DISTRIBUTION
        accumulation_week = signal == ACCUMULATION

    near_support = False
    if not item.levels.empty:
        supports = item.levels.loc[item.levels["Role"] == "support", "Level"]
        if not supports.empty:
            close = item.last_close
            nearest = min(supports, key=lambda lvl: abs(float(lvl) - close))
            near_support = abs(float(nearest) - close) / close <= 0.05

    constructive = stance == ACCUMULATION and (ad != ad or ad >= 50.0)
    is_c_grade = grade.upper().startswith("C")

    # (2) light-volume pullback near support
    if constructive and light_pullback and near_support:
        s2 = ("Best — live", "Soft tape on quiet volume near support while the footprint still favours demand.")
    elif constructive and light_pullback:
        s2 = ("Good — live pause", "Light-volume soft week with constructive grade/A/D; confirm nearest support in Stock detail.")
    elif constructive:
        s2 = ("Good — setup pending", "Footprint is constructive, but this week's volume/close is not yet a clear quiet pullback.")
    else:
        s2 = ("Weak / not live", "Need accumulation stance and a quiet (not distribution) pullback before this fits.")

    # (1) buy into strength on an accumulation week
    if accumulation_week and constructive:
        s1 = ("Best — live", "Current week is a labeled accumulation week on a constructive grade.")
    elif item.summary.accumulation_weeks > 0 and constructive:
        s1 = ("Good — not live this week", "Recent accumulation weeks exist, but the current bar is not labeled accumulation.")
    elif accumulation_week:
        s1 = ("Mixed", "An accumulation week printed, but grade/A/D are not clearly constructive.")
    else:
        s1 = ("Weak / not live", "No current accumulation-week signal to buy into.")

    # (3) buy despite a C grade
    if is_c_grade:
        s3 = ("Weak / against method", "C means churn — buying anyway fights the composite grade.")
    else:
        s3 = ("N/A", f"Grade is {grade}, not C. This scenario only applies when the stock is churning.")

    ranked = [
        (2, "Buy a light-volume pullback near support", *s2),
        (1, "Buy into strength on an accumulation week", *s1),
        (3, "Buy despite a C grade", *s3),
    ]

    fit_rank = {"Best — live": 0, "Good — live pause": 1, "Good — not live this week": 2,
                "Good — setup pending": 3, "Mixed": 4, "N/A": 5, "Weak / not live": 6,
                "Weak / against method": 7}
    ranked.sort(key=lambda row: (fit_rank.get(row[2], 9), row[0]))
    return [(i + 1, name, fit, why) for i, (_, name, fit, why) in enumerate(ranked)]


def run(ticker: str, period: str, *, provider: str | None = None) -> int:
    cfg = config.load()
    ticker = normalize_ticker(ticker)
    result = data.fetch(
        [ticker] + cfg.indexes, period=period, provider=provider or cfg.provider
    )
    if ticker not in result.bars or result.bars[ticker].empty:
        print(f"No usable history for {ticker}.")
        if result.errors.get(ticker):
            print(f"  fetch error: {result.errors[ticker]}")
        return 1

    item = analysis.analyze(ticker, result.bars[ticker], cfg.settings)
    r = item.rating
    w = item.latest_week
    s = item.summary
    ad_slope = indicators.trend_slope(indicators.ad_line(item.daily), 50)

    recent = item.weekly.tail(cfg.settings.signal_lookback_weeks)[
        ["Close", "RangePos", "VolRatio", "Signal"]
    ].copy()
    recent["RangePos"] = (recent["RangePos"] * 100).round(1)
    recent["VolRatio"] = recent["VolRatio"].round(2)

    print(f"=== {ticker} SNAPSHOT ===")
    print(f"provider={result.provider}")
    for borrowed, served_by in result.sources.items():
        if borrowed not in result.proxies:
            print(f"  note: {borrowed} served by {served_by} (composite volume)")
    for proxied, proxy_symbol in result.proxies.items():
        print(f"  note: {providers.symbols.proxy_note(proxied, proxy_symbol)}")
    print(
        f"as_of={item.last_date.date()} close={item.last_close:.2f} "
        f"chg={_fmt(item.change_pct, '+.2f')}%"
    )
    print(f"grade={r.grade} score={_fmt(r.score, '.1f')} stance={r.stance}")
    comps = ", ".join(f"{k}: {_fmt(v, '.1f')}" for k, v in r.components.items())
    print(f"components={{ {comps} }}")
    print(
        f"ud_latest={_fmt(item.ud_latest)} "
        f"daily_rvol={_fmt(item.daily_rel_volume)}"
    )
    print(
        f"ad_slope_50={ad_slope:.6g} "
        f"ad_component={_fmt(r.components.get('ad_trend', float('nan')), '.1f')}"
    )
    print(
        f"summary: acc={s.accumulation_weeks} dist={s.distribution_weeks} "
        f"net={s.net_weeks} headline={s.headline}"
    )
    if w is not None:
        print(
            f"latest_week: signal={w['Signal']} "
            f"range_pos={float(w['RangePos']):.1%} "
            f"rvol={_fmt(float(w['VolRatio']))} "
            f"in_progress={item.week_in_progress}"
        )

    print(f"--- last {cfg.settings.signal_lookback_weeks} weeks ---")
    print(recent.to_string())
    print("--- levels ---")
    print(item.levels.to_string() if not item.levels.empty else "none")

    mh = market.analyze_indexes(
        result.bars,
        cfg.indexes,
        pct_threshold=cfg.settings.distribution_day_pct,
        window=cfg.settings.distribution_day_window,
    )
    for health in mh:
        print(
            f"market {health.name}: dist_days={health.count} level={health.level}"
        )

    constructive = r.stance == ACCUMULATION
    print()
    print("=== BUY ON ACCUMULATION vs TRAP ===")
    if constructive:
        print(
            f"Live read: constructive — {r.grade} / {r.stance}, "
            f"U/D {_fmt(item.ud_latest)}, A/D component "
            f"{_fmt(r.components.get('ad_trend', float('nan')), '.1f')}."
        )
        print(
            "Ideal buy: soft week on light volume that is not labeled distribution, "
            "holding a support cluster, while grade and A/D stay constructive."
        )
        print(
            "Trap buy: a bounce after a heavy distribution week, A/D rolling over, "
            "or buying into resistance while indexes stack distribution days."
        )
    else:
        print(
            f"Live read: not constructive — {r.grade} / {r.stance}, "
            f"U/D {_fmt(item.ud_latest)}, A/D component "
            f"{_fmt(r.components.get('ad_trend', float('nan')), '.1f')}."
        )
        print(
            "Trap risk is elevated: buying a bounce while distribution (or churn) "
            "owns the footprint can look like a dip-buy and still be unfinished selling."
        )
        print(
            "Ideal buy would need: accumulation stance, rising A/D, and a light-volume "
            "pause — none of that is the headline here."
        )

    print()
    print("=== THREE BUY SCENARIOS (ranked by OLGA fit now) ===")
    for rank, name, fit, why in _rank_scenarios(item):
        print(f"{rank}. {name}")
        print(f"   fit: {fit}")
        print(f"   why: {why}")

    _print_exit_watch(_exit_watch(item, cfg.settings, mh))

    print()
    print(
        "Out of scope: paper fills, stops, sizing, and which scenario made money "
        "need backtesting OLGA does not have."
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="OLGA buy-scenario walkthrough for one ticker."
    )
    parser.add_argument("ticker", help="Symbol to analyze, e.g. MSFT or GOOGL")
    parser.add_argument(
        "--period",
        default="2y",
        help="History window, e.g. 1y 2y 3y 5y ytd 120d (default: 2y)",
    )
    parser.add_argument(
        "--provider",
        default=None,
        choices=["auto", *providers.PROVIDER_NAMES],
        help="Override the provider in config/watchlist.json for this run.",
    )
    args = parser.parse_args(argv)
    return run(args.ticker, args.period, provider=args.provider)


if __name__ == "__main__":
    raise SystemExit(main())
