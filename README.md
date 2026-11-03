# OLGA Trader

A local dashboard that reads institutional footprints out of price range and
volume — telling accumulation (institutions buying) from distribution
(institutions selling) across a watchlist, and counting distribution days on
the S&P 500 and Nasdaq.

Built on the premise that institutions cannot hide: a fund accumulating shares
pushes weekly closes toward the top of the trading range on volume well above
normal, and a fund unloading them pins closes near the low on that same heavy
volume.

## Running it

```powershell
.venv\Scripts\streamlit run app.py
```

The app opens at <http://localhost:8501>. A dedicated virtual environment is
already set up in `.venv` (Python 3.13) — the Anaconda base install on this
machine has a `yfinance` too old to reach Yahoo Finance and a NumPy 2.0 / NumPy
1.x binary mismatch, so it is deliberately not used.

To rebuild the environment from scratch:

```powershell
& "C:\Program Files\Python313\python.exe" -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```

Verify the indicator pipeline and the provider layer at any time:

```powershell
.venv\Scripts\python smoke_test.py
```

Checks that need an API key report `[SKIP]` rather than failing, so the suite
is meaningful with or without keys.

## Market data

Bars come from one of three providers, chosen in `config/watchlist.json`:

```json
"data": { "provider": "auto" }
```

| Provider | Key | Notes |
|---|---|---|
| `tiingo` | `TIINGO_API_KEY` | End-of-day, split- **and** dividend-adjusted. Equities and ETFs only — no indices at any tier. Publishes ~5:30 PM ET, so it has the current session before Yahoo does. |
| `polygon` | `POLYGON_API_KEY` | Daily aggregates, split-adjusted. Indices need an Indices plan and carry no volume. Free tier is 5 requests/minute. |
| `yahoo` | none | The original, via `yfinance`. Unofficial API, but the only one with real composite index volume. |

`auto` picks the first provider that has a key — Tiingo, then Polygon, then
Yahoo — so the app runs with no keys at all and upgrades itself the moment one
lands in `.env`. Each user picks their own vendor in the sidebar (saved under
`config/users/`). `config/watchlist.json` only holds the default for a first
visit. Pin one for a single process with `$env:OLGA_PROVIDER`.

Keys live in `.env` and nowhere else — never in `watchlist.json`, which is
meant to be readable and shareable:

```powershell
Copy-Item .env.example .env   # then paste your key in
```

Selecting a keyed provider without its key is a visible error in the sidebar
and a per-ticker error in the table, never a silent fall back to Yahoo.

### Rate limits

Every Tiingo tier, paid ones included, has an **hourly request allocation**,
and Polygon's free tier allows 5 requests/minute. Normal use never approaches
either: the 4-hour disk cache means a watchlist refresh costs one request per
symbol per four hours, and 14 symbols over 3 years fetch in about 2 seconds at
the default 4 workers. Raising `TIINGO_MAX_WORKERS` does not help — 8 is no
faster and 16 is slower — so it only makes the quota easier to exhaust.

If you do run over, the error text carries the vendor's own explanation and
cached bars keep serving until the window resets. `smoke_test.py` reports
quota-blocked checks as `[SKIP]`, not `[FAIL]`, since no code change fixes
them.

### Indexes: how `^GSPC` and `^IXIC` are sourced

The watchlist holds Yahoo-style symbols (`^GSPC`, `BRK-B`) and each provider
translates them — Polygon wants `I:SPX` and `BRK.B`. Index data is the one
place the providers genuinely differ, and it matters because **a distribution
day is defined by volume rising**:

- **Tiingo sells no equity indices at any plan tier.** Verified live against a
  paid plan, not inferred from the docs.
- **Polygon's index aggregates are price-only**, even with an Indices
  subscription — no volume at all.

So an index walks down a three-step chain until something can supply real
volume:

1. The active provider's native index symbol (`I:SPX` on Polygon).
2. **Yahoo**, which carries genuine composite index volume. Worth a second
   vendor, because composite volume is what the method is defined against.
3. An **ETF proxy** on the active provider — `SPY` for `^GSPC`, `QQQ` for
   `^IXIC` — only if Yahoo is unreachable too.

Step 2 is the normal path on Tiingo, and the app notes it quietly. Step 3 is a
last resort and gets a visible warning, because it really does change the
answer: on 2026-09-16 the composite `^GSPC` read 6 distribution days in 25
sessions where `SPY` read 10. The frame is always keyed under `^GSPC`, so
nothing downstream changes — only where the volume came from.

> **Never map a Tiingo index to a same-named ticker.** Tiingo resolves `COMP`
> to Compass Inc (a ~$10 NYSE stock) and `SPX` to Spenda Limited on the ASX.
> Both return HTTP 200, so no error path would catch the substitution — a
> small-cap stock would just quietly stand in for the Nasdaq Composite.
> `providers/symbols.py` maps every Tiingo index to `None` for exactly this
> reason, and `smoke_test.py` pins it.

## The four tabs

**Watchlist** — every stock graded A (heavy accumulation) through E (heavy
distribution), ranked, with the up/down volume ratio, where the latest weekly
close landed in its range, and how heavy that week's volume was.

**Stock detail** — the weekly candle and volume chart with each week coloured
by its signal, the levels the stock keeps turning at, the up/down volume ratio
through time, the A/D line with its trend, and a breakdown of what drove the
grade.

**Market pressure** — distribution days per index, marked on the chart and
counted over a rolling window against the 5-day warning level.

**How to read this** — the methodology, written out with your current
thresholds substituted in.

## What each signal measures

| Signal | Definition |
|---|---|
| Close-in-range | `(Close − Low) ÷ (High − Low)` on the weekly bar. 100% is a close on the high. |
| Volume vs normal | Weekly volume ÷ its trailing 10-week average, excluding the current week so a big bar cannot set its own baseline. |
| Accumulation week | Closes ≥ 70% up its range on ≥ 1.25× average volume. |
| Distribution week | Closes ≤ 30% up its range on ≥ 1.25× average volume. |
| Up/down volume ratio | Volume on up days ÷ volume on down days over 50 days. 2.0 means twice as much traded on up days; below 1.0 favours sellers; 1.0 is churn. |
| A/D line | Chaikin accumulation/distribution — each bar's volume weighted by where it closed in its range, accumulated. Its slope is the pressure trend. |
| Support / resistance | Prices the stock has repeatedly turned at, found by clustering swing pivots. |
| Distribution day | An index closing ≤ −0.2% on higher volume than the prior session. Five or more in 25 sessions is the warning. |

A pullback on *light* volume is deliberately neither signal — that is
institutions pausing, not selling.

Every threshold in that table is adjustable in the sidebar, and the "How to
read this" tab re-writes itself to match whatever you set.

## The grade

A 0-100 composite mapped to a letter, A through E, blending four independent
readings so no single noisy input decides it:

| Component | Weight |
|---|---|
| Up/down volume ratio | 35% |
| Accumulation weeks minus distribution weeks | 30% |
| A/D line trend | 20% |
| Volume-weighted close strength | 15% |

A component that cannot be computed from the loaded history drops out and its
weight is shared among the rest, so short-history tickers still grade rather
than erroring.

## Layout

```
app.py              Streamlit UI — tabs, tiles, tables
olga/
  config.py         seed watchlist + indexes; personal copies under config/users/
  identity.py       Streamlit viewer id used to isolate saved inputs
  env.py            loads .env before anything reads an API key
  data.py           fetch facade: 4-hour on-disk cache, period → dates
  providers/
    __init__.py     get_provider() / resolve_name("auto") / key_status()
    base.py         the contract: normalize_ohlcv, retries, proxy fallback
    symbols.py      OLGA ticker → provider ticker, and the ETF proxies
    tiingo.py       Tiingo end-of-day
    polygon.py      Polygon daily aggregates
    yahoo.py        yfinance, the no-key path
  indicators.py     the maths: range position, volume ratios, A/D, OBV, pivots
  rating.py         the A-E composite
  market.py         index distribution days
  analysis.py       per-stock assembly, the seam between maths and UI
  charts.py         Plotly figures
  theme.py          palette and chart chrome
smoke_test.py       headless verification of the whole pipeline
config/watchlist.json   indexes and the default watchlist / vendor seed
config/users/           per-user tickers, thresholds, and vendor (git-ignored)
.env                    API keys (git-ignored) · .env.example is the template
```

`data.fetch()` is the only way anything reaches the network, and it returns the
same OHLCV frames whichever provider served them — which is what makes the
provider swappable without touching indicators, ratings or charts.

Each signed-in viewer (Streamlit Cloud / OIDC email on `st.user`) gets their
own watchlist, signal thresholds, and data vendor under `config/users/`.
Saving a ticker, moving a threshold slider, or switching provider writes that
file only — never another user's. `config/watchlist.json` stays the seed for a
first visit (indexes included). Cached bars live in `.cache/`, namespaced by
provider so switching vendors never re-serves another vendor's bars. Refresh
refetches for this session; it does not wipe the shared market-data cache.
API keys in `.env` remain shared.

## Charts

Colour is validated for colour-vision deficiency in both light and dark modes
(worst adjacent-pair CVD ΔE 24.7 light / 26.8 dark against an ≥ 8 target).
Accumulation and distribution are always paired with a ▲/▼ glyph and a word, so
the reading never rests on hue alone. Candles are filled when the week closed
up and hollow when it closed down, which keeps direction legible even though
colour is spent on the signal. Every chart has a table twin, so no value is
reachable only by hovering.

## Limits worth knowing

These are inferences from public price and volume, not from filings — they
suggest institutional behaviour, they do not prove it.

Bars are split-adjusted everywhere, and dividend-adjusted on Tiingo and Yahoo
but not on Polygon, whose aggregates apply splits only. Every OLGA reading is a
ratio inside a rolling window, so this shifts nothing material, but a long
history of a high-yield name will show slightly deeper drawdowns under Polygon.

Composite index volume is what distribution-day counting expects; when the
provider cannot supply it an ETF stands in and the app says so. Index volume is
revised intraday, so the newest session can shift. The newest weekly bar is
partial until Friday closes and the app labels it as provisional.

Yahoo's API is unofficial and occasionally changes — that is why the paid
providers exist. Whatever the provider, `data.py` falls back to cached bars when
a download fails, however stale, and reports which symbols it could not load
rather than showing a blank chart.

Nothing here is investment advice.
