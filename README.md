# QuantBot

A modular, risk-first **trading-signal bot** in Python. It analyses crypto (and,
by design, forex/equities) markets across multiple timeframes, scores setups
against a weighted confluence model, sizes them against a strict risk budget,
and reports what it found over Telegram.

It **does not place orders.** See [Why there is no execution
module](#why-there-is-no-execution-module).

---

## Read this first

No strategy guarantees profit, and this one is no exception. The honest claims
this project can make are narrow:

- Every rule is explicit, configurable and unit-tested.
- The backtester provably does not read the future — that is asserted by a
  [dedicated test suite](tests/test_no_lookahead.py), not just claimed here.
- Costs (fee + slippage, both sides) are subtracted everywhere a reward figure
  appears, including in the reward-to-risk gate.
- Risk limits are enforced in code and survive a restart.

What it cannot tell you is whether the strategy has an edge in *your* market, in
*this* regime. That is what the backtest and then a paper-trading period are
for. The intended order is:

```
config-check  →  fetch  →  backtest on real history  →  optimize (walk-forward)
              →  backtest again on a different period  →  paper trade for weeks
              →  only then consider real money
```

If the backtest shows negative expectancy, the bot says so in plain words and
you should stop. That is a feature.

---

## Table of contents

- [Architecture](#architecture)
- [How a signal is produced](#how-a-signal-is-produced)
- [The scoring system](#the-scoring-system)
- [Risk management](#risk-management)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Backtesting](#backtesting)
- [Optimisation](#optimisation)
- [Telegram](#telegram)
- [Logging](#logging)
- [Deploying to a Linux VPS](#deploying-to-a-linux-vps)
- [Running 24/7](#running-247)
- [Design decisions and honest limitations](#design-decisions-and-honest-limitations)
- [Testing](#testing)
- [Future work](#future-work)

---

## Architecture

Data flows in one direction. Each layer depends only on the ones above it, which
is what makes the backtester able to drive the *same* strategy object the live
loop drives.

```
                      config/          settings.yaml + typed dataclasses
                         │              (validated once, at start-up)
                         ▼
   ┌──────────── data/ ─────────────┐
   │ exchange.py   ccxt adapter     │   ← the only async, failure-prone layer
   │ repository.py cache + refresh  │
   │ sentiment.py  F&G, dominance   │
   │ collector.py  ──────────────┐  │
   └─────────────────────────────│──┘
                                 ▼
                       indicators/          30+ indicators, one pass
                                 │
                                 ▼
                        analysis/           what the numbers MEAN
              swings → structure → patterns → volume → regime
                                 │
                                 ▼
                     TimeframeContext × N timeframes
                                 │
                                 ▼
                       strategies/          scoring, MTF, triggers, gates
                                 │
                                 ▼
                           Signal | Rejection
                                 │
                    ┌────────────┴────────────┐
                    ▼                         ▼
                 risk/                     notify/
       planner → manager → guards        Telegram + JSONL
       portfolio ← execution
                    │
        ┌───────────┴───────────┐
        ▼                       ▼
     live/                  backtest/
   the real loop          the same code,
                          replayed on history
```

### Module map

| Package | Responsibility | Key entry point |
|---|---|---|
| `config/` | Typed settings from YAML + `.env`, validated at start-up | `load_settings()` |
| `utils/` | Logging channels, timeframe arithmetic, retry helpers | `setup_logging()`, `is_bar_closed()` |
| `data/` | ccxt adapter, candle cache, sentiment feeds, snapshot assembly | `DataCollector.collect()` |
| `indicators/` | EMA/RSI/MACD/ADX/ATR/Supertrend/Ichimoku/… in one pass | `IndicatorEngine.compute()` |
| `analysis/` | Swings, BOS/CHoCH, S&R, zones, candlesticks, volume, regime | `ContextBuilder.build()` |
| `strategies/` | Weighted scoring, MTF agreement, triggers, hard gates | `ConfluenceStrategy.evaluate()` |
| `risk/` | Stop/target maths, sizing, circuit breakers, ledger | `RiskManager.approve()` |
| `backtest/` | Bar-closed replay, metrics, reports, walk-forward search | `Backtester.run()` |
| `notify/` | Telegram transport and message templates | `Notifier` |
| `live/` | The scan loop and durable state | `LiveTrader.run()` |

> **Naming note.** The specification asked for a `telegram/` directory. It is
> called `notify/` here because a top-level package named `telegram` shadows the
> PyPI `telegram` module — any dependency doing `import telegram` would get this
> project instead and fail. `notify/telegram_client.py` is the Telegram
> transport.

---

## How a signal is produced

The evaluation order **is** the design. It implements "if the conditions are not
all there, do not enter" as code rather than as a comment.

| # | Stage | Fails when | Result |
|---|---|---|---|
| 1 | **Data sanity** | indicators not yet seeded | rejection `data` |
| 2 | **Direction** | higher timeframes are undecided | rejection `direction` |
| 3 | **Cooldown / daily cap** | too soon after the last signal | rejection `cooldown` |
| 4 | **MTF agreement** | 1d/4h opposes, or 1h does not agree | rejection `mtf_alignment` |
| 5 | **Trigger** | nothing justifies acting on *this* bar | rejection `trigger_required` |
| — | *score computed here, so every later rejection reports it* | | |
| 6 | **Hard gates** | ADX, volume, ATR range, room, structure | rejection named after the gate |
| 7 | **Trade plan** | net RR below `risk.min_rr` | rejection `min_rr` |
| 8 | **Score threshold** | below `strategy.min_score_to_emit` | rejection `score` |

**A score of 95 cannot buy its way past a failed gate.** Gates are structural
statements about whether a setup is tradable at all; the score only ranks the
setups that already are. This is asserted in
`tests/test_strategy.py::test_a_perfect_score_cannot_pass_a_failed_gate`.

### Timeframes have roles

| Role | Default | Rule |
|---|---|---|
| **context** | `1d`, `4h` | may be neutral, must never *oppose* |
| **confirm** | `1h` | must actively agree |
| **entry** | `15m`, `5m` | where the trigger fires |

Agreement is **weighted** (`strategy.mtf.weights`), because "3 of 5 timeframes
agree" is meaningless when the 2 that disagree are the daily and the 4-hour. The
proposed direction comes from the context and confirm timeframes only — never
from the entry chart.

### Triggers

Without a trigger, a trend-following bot enters on every bar the trend persists,
which is how it ends up long at the top. Four are recognised, in priority order:

1. `breakout` — close beyond the **previous** bar's Donchian boundary, funded by
   a volume spike or a released squeeze.
2. `bos_continuation` — a recent break of structure that price has held.
3. `pullback` — an established trend retraces to the EMA or a demand/supply
   zone, prints a rejection, and momentum turns back. All four conditions.
4. `sweep_reversal` — a stop-run beyond a swing extreme that closed back inside.

---

## The scoring system

Eight components, each itself a small vote of several independent readings, so
one distorted input cannot carry a component to full marks.

| Component | Weight | What it reads |
|---|---:|---|
| Trend | 15 | EMA stack, price vs EMA200, Supertrend, Ichimoku cloud, slope |
| MACD | 15 | histogram sign, recent cross, expansion, zero line |
| RSI | 10 | midline reclaim, room before the extreme, turning |
| Volume | 20 | spike, flow direction, OBV, CMF |
| ADX | 10 | above threshold, strong, DI agreement, rising |
| Price action | 15 | candlestick patterns, sweeps, failed breaks, at a zone |
| Structure | 15 | swing trend, BOS/CHoCH, room to the next level, exhaustion |
| Sentiment | 10 | funding, open interest, Fear & Greed, dominance, news |

**Weights are relative and normalised to 0–100.** They sum to 110 here, matching
the specification's numbers exactly, and that is fine: the scorer divides earned
points by the weight that was *available*.

That normalisation exists for one important reason. When a component has no data
— no funding feed on a spot market, no news API key — its weight leaves the
**denominator** entirely. Scoring it as zero instead would cap every score at
~91 and silently change the meaning of the 60/70/80 thresholds you configured.

| Total | Tier |
|---|---|
| ≥ 80 | Strong |
| ≥ 70 | Normal |
| ≥ 60 | Weak |
| < `min_score_to_emit` | ignored |

`strategy.min_score_to_emit` defaults to 70. Raise it to 80 when going live.

---

## Risk management

**Position size is derived from the stop distance, not from a fixed notional.**
That is what makes `risk_per_trade_pct` mean what it says: a wider stop produces
a smaller position, so the cash lost when the stop is hit is the same either way.

```
quantity = (equity × risk_per_trade_pct / 100) / |entry − stop|
```

| Control | Default | Notes |
|---|---|---|
| Risk per trade | 1% | `allowed_risk_levels: [0.5, 1.0, 2.0]`; >2% is rejected outright |
| Stop | structural, ATR floor & cap | swing ± buffer; falls back to ATR when the swing is too far |
| Targets | 2R and 3R, 50/50 | partial exits |
| Minimum RR | 2.0 | **measured net of costs** |
| Break-even | at 1R | placed just beyond entry so the exit still covers the round trip |
| Trailing | from 1.5R, 2×ATR | never moves against the position |
| Daily loss limit | 3% | halts for the day, lifts at the UTC boundary |
| Max drawdown | 15% | halts **permanently** — needs a human |
| Consecutive losses | 4 | halts |
| Max open positions | 3 | plus max 2 in the same direction (correlation) |
| Spread guard | 0.08% | an *unmeasurable* spread warns, never silently passes as zero |
| Volatility guard | 3× median ATR | stand aside during anomalies |

Two details worth knowing:

- **Circuit breakers mark to market.** Realised PnL alone would let an open,
  deeply underwater position sail past the daily limit. (This was a real bug,
  caught by `test_open_losses_count_towards_the_breakers`.)
- **A max-drawdown halt is not cleared by restarting.** It is persisted in
  `logs/state.json`, so restarting the process cannot be used to bypass a risk
  limit. Clear it deliberately, after understanding why it fired.

---

## Installation

Requires **Python 3.10+** (3.11 recommended).

```bash
git clone https://github.com/pdtrhieu2008/Baitapaimbotkiemthu.git
cd Baitapaimbotkiemthu

python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env               # fill in whatever you have; all optional
python main.py config-check
```

Or with `make`:

```bash
make install-dev    # runtime + pytest/ruff/mypy
make check          # lint + tests
```

### Dependency notes

Required: `pandas`, `numpy`, `scipy`, `PyYAML`, `ccxt`, `aiohttp`.
Optional: `plotly` (HTML report), `matplotlib`, `optuna` (smarter search).

Four libraries from the original specification are **deliberately not used**:

| Not used | Why |
|---|---|
| `ta` / `pandas-ta` | All indicators are implemented in-house in `indicators/`. Every formula stays auditable and unit-tested here, with no silent retuning on an upstream release — and `pandas-ta` currently breaks on `numpy>=2` / recent pandas. |
| `vectorbt` | Vectorised, so it would need a *second*, divergent implementation of the same rules. The event-driven backtester here reuses the exact live strategy object; that is the whole point. |
| `backtesting.py` | Same reason. |
| `python-telegram-bot` | This bot only ever **sends**. The Bot API is a plain HTTPS POST, so `notify/telegram_client.py` is ~120 lines on `aiohttp` with no extra dependency. Add PTB if you later want *inbound* commands. |

`websockets` and `schedule` are likewise unnecessary: ccxt handles transport and
scheduling is `asyncio` in `live/trader.py`.

---

## Usage

```bash
python main.py config-check     # validate the configuration, print it masked
python main.py selftest         # exercise every module offline (synthetic data)
python main.py fetch            # download and cache candles
python main.py scan             # one analysis pass, printed
python main.py backtest         # replay history and print a report
python main.py optimize         # walk-forward parameter search
python main.py telegram-test    # verify notifications
python main.py run              # the live signal loop
```

Every command accepts overrides, so nothing needs editing to try a variation:

```bash
python main.py backtest \
  --symbol ETH/USDT --timeframe 1h --start 2024-01-01 \
  --set risk.risk_per_trade_pct=0.5 \
  --set strategy.min_score_to_emit=80
```

A typical first session:

```bash
python main.py config-check
python main.py fetch                        # a few minutes for 3 symbols × 6 TFs
python main.py backtest --symbol BTC/USDT   # read the report and the warnings
python main.py scan                         # see what it thinks right now
python main.py run                          # start the loop
```

`selftest` runs the whole pipeline on a seeded random walk. It proves the modules
run and agree on their interfaces; it says **nothing** about whether the strategy
works, and the command says so itself.

---

## Configuration

Everything lives in [`config/config.yaml`](config/config.yaml). Nothing is
hard-coded elsewhere. Secrets come from `.env` via `${VAR}` placeholders.

Validation is deliberately strict and happens **before** anything runs:

- Unknown keys are errors, not warnings. A typo like `risk_pct` instead of
  `risk_per_trade_pct` would otherwise silently leave the default in place.
- `tp_split` must sum to 1; `min_rr < 1` is rejected (it needs a >50% win rate
  just to break even); `risk_per_trade_pct > 2` is rejected.
- Cross-section checks catch an MTF timeframe that is never downloaded, a warm-up
  shorter than the longest indicator lookback, funding enabled on a spot market,
  and a furthest target that could never satisfy the RR gate.
- `app.mode: live` and `backtest.execution: close` are rejected with an
  explanation, because both are footguns.

```bash
python main.py config-check --set risk.risk_per_trade_pct=5
# Configuration error: risk.risk_per_trade_pct (5.0) is not in allowed_risk_levels [0.5, 1.0, 2.0]
```

### Adapting to forex or equities

The strategy layer knows nothing about crypto. To add a venue, implement
`data.base.MarketDataProvider` (six methods) and change nothing else. Then:

```yaml
exchange:
  market_type: spot        # no funding / open interest
data:
  funding_rate: false
  open_interest: false
  fear_greed: false        # crypto-specific
  btc_dominance: false
```

The sentiment component then marks itself unavailable and its weight is
redistributed. For annualising Sharpe on a session market, pass the right
`bars_per_year(...)` arguments (`market_hours_per_day=6.5,
trading_days_per_year=252` for US equities).

---

## Backtesting

```bash
python main.py backtest --symbol BTC/USDT --timeframe 15m --start 2024-01-01
```

Reports win rate, profit factor, Sharpe, Sortino, Calmar, SQN, expectancy in
both cash and R, average win/loss, drawdown and duration, streaks, exposure,
exit-reason breakdown, and an equity curve (CSV + optional Plotly HTML).

It also prints a **signal funnel** — how many bars were rejected by which gate.
That is usually more informative than the PnL, because it tells you whether "no
trades" means "no setups" or "one gate is misconfigured".

```
-- Signal funnel ------------------------------------------------------
  Signals emitted / taken        38 / 22
  rejected by gate (why the bot stayed out):
        1348  trigger_required
         742  mtf_alignment
         325  volume_confirmation
```

### What makes it honest

| Guarantee | How |
|---|---|
| No lookahead | signal from bar *i*'s close, filled at bar *i+1*'s **open** |
| MTF causality | higher timeframes sliced to the last bar that had genuinely *closed* |
| Pessimistic intrabar | when a bar contains both stop and target, the **stop** is assumed first |
| Gaps respected | a next-open gap through the stop drops the order rather than booking a broken RR |
| Costs applied | fee + slippage on entry *and* exit, in the direction that hurts |
| Same code as live | drives the real `Strategy`, `RiskManager` and `Portfolio` |

And it tells you when not to believe it: fewer than 30 trades, zero losing
trades (usually a lookahead bug, not perfection), zero drawdown, or negative
expectancy each produce an explicit warning.

**Historical funding and sentiment are not reconstructed.** The sentiment
component is therefore marked unavailable throughout a backtest, and the run
warns that live scores will use a slightly different weight base. Reconstructing
them properly means storing the funding series alongside the candles; until then,
saying so is better than back-filling a guess.

---

## Optimisation

```bash
python main.py optimize --set optimization.objective=expectancy
```

**Walk-forward by construction.** Every candidate is fitted on the first
`train_fraction` of the data and **scored on the remainder, which it never saw**.
The reported best is the best *out-of-sample* result; the in-sample figure is
kept only so the gap between them can be inspected — a large gap is the signature
of curve fitting, and the report says so:

```
  out-of-sample expectancy          +0.1840
  in-sample     expectancy          +0.9120
  overfit gap                       +0.7280

  ! The in-sample score is far above the out-of-sample score. These
    parameters largely describe the training window, not the market.
```

The split is chosen on the primary timeframe and applied as a **timestamp** to
every timeframe; splitting each by row count would put the boundary on a
different date per timeframe and let the 4h see past the cut.

Grid search needs nothing extra. `optimization.method: optuna` uses Bayesian
search when Optuna is installed and falls back to grid when it is not.

> A good out-of-sample score is a reason to paper-trade, not a reason to trust.
> Re-run on a different period before committing.

---

## Telegram

1. Message [@BotFather](https://t.me/BotFather) → `/newbot` → copy the token.
2. Send your new bot any message, then open
   `https://api.telegram.org/bot<TOKEN>/getUpdates` and copy
   `result[0].message.chat.id`.
3. Put both in `.env`, set `telegram.enabled: true`, and verify:

```bash
python main.py telegram-test
```

Telegram ships **disabled** so a fresh clone starts without credentials. When
disabled, every notification still goes to `logs/telegram.log` and the JSONL
streams — nothing is lost.

Signal message:

```
==========================
STRONG BUY SIGNAL
==========================
Coin: BTC/USDT
Timeframe: 15m

Entry: 67,120.50
Stop Loss: 66,745.20  (-0.56%)
Take Profit: 67,871.10 (2R, 50%) / 68,246.40 (3R, 50%)
Risk/Reward: 2.62 (net of fees)
Risk: 10.00 (1.00% of 1000.00)
Confidence: 84/100 (strong)

Trend: trend_up / strong (ADX 31.2)
Structure: up, last break bos_bull
Volume: 2.10x average, flow +64%
ATR: 250.00 (0.373% of price)
Funding: -0.0080%

Trigger: breakout
Reason:
  • trigger: close broke the 20-bar high on volume spike
  • EMA21 > EMA50
  • MACD cross in the last 4 bars
  • volume spike 2.1x average
  • BOS confirmed 2 bars ago
  • MTF: 100% agreement (4h, 1h, 15m)

Against:
  • RSI 68 already close to overbought

Signal only - no order was placed.
==========================
```

Closed trades report PnL, R multiple, excursion, exit reason and running win
rate / profit factor / expectancy. A daily summary goes out at the configured
UTC hour.

**Rejections are logged but not sent.** A 30-second loop over several symbols
produces hundreds a day, and a chat full of "no trade" teaches you to ignore it.
They are all in `logs/signal.jsonl`.

---

## Logging

Five rotating channels plus machine-readable JSONL:

```
logs/quantbot.log     everything, human readable
logs/signal.log       one line per signal / rejection
logs/trade.log        position opens, closes, stop moves
logs/error.log        WARNING+ from anywhere
logs/telegram.log     outbound notification attempts
logs/performance.log  periodic equity / metric snapshots
logs/*.jsonl          structured events
logs/state.json       durable risk counters and open positions
```

Analyse a run in pandas:

```python
import pandas as pd
signals = pd.read_json("logs/signal.jsonl", lines=True)
signals[signals.event == "signal_rejected"].gate.value_counts()
```

---

## Deploying to a Linux VPS

Tested against Ubuntu 22.04/24.04 and Debian 12. A 1 vCPU / 1 GB instance is
enough for a handful of symbols.

```bash
# --- 1. system packages ---
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3 python3-venv python3-pip git

# --- 2. correct time. Non-negotiable: bar boundaries, the closed-bar check
#        and the daily risk reset are all computed from system time. ---
sudo timedatectl set-timezone UTC
sudo timedatectl set-ntp true
timedatectl status          # confirm "System clock synchronized: yes"

# --- 3. a dedicated unprivileged user ---
sudo useradd --system --create-home --home-dir /opt/quantbot --shell /bin/bash quantbot

# --- 4. the code ---
sudo -u quantbot git clone https://github.com/pdtrhieu2008/Baitapaimbotkiemthu.git /opt/quantbot
cd /opt/quantbot
sudo -u quantbot python3 -m venv .venv
sudo -u quantbot .venv/bin/pip install --upgrade pip
sudo -u quantbot .venv/bin/pip install -r requirements.txt

# --- 5. credentials, readable only by the service user ---
sudo -u quantbot cp .env.example .env
sudo -u quantbot nano .env
sudo chmod 600 /opt/quantbot/.env

# --- 6. verify BEFORE starting anything long-running ---
sudo -u quantbot .venv/bin/python main.py config-check
sudo -u quantbot .venv/bin/python main.py fetch
sudo -u quantbot .venv/bin/python main.py backtest
```

Optional hardening — the bot needs no inbound ports at all:

```bash
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw allow OpenSSH
sudo ufw enable
```

---

## Running 24/7

### Option A — systemd (recommended for a plain VPS)

```bash
sudo cp deploy/quantbot.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now quantbot

systemctl status quantbot
journalctl -u quantbot -f
```

The unit ([`deploy/quantbot.service`](deploy/quantbot.service)):

- runs `config-check` as `ExecStartPre`, so a bad configuration fails the deploy
  instead of crash-looping;
- waits for `time-sync.target`;
- sends `SIGTERM` and allows 45 s, so the current cycle finishes and state is
  persisted;
- gives up after 5 failures in 10 minutes, so a broken deploy is visible instead
  of hammering the exchange;
- is hardened: `ProtectSystem=strict`, `NoNewPrivileges`, a syscall filter, and
  write access to only `logs/`, `data/cache/` and `reports/`.

Update:

```bash
cd /opt/quantbot
sudo -u quantbot git pull
sudo -u quantbot .venv/bin/pip install -r requirements.txt
sudo systemctl restart quantbot
```

### Option B — Docker

```bash
cp .env.example .env && nano .env

docker compose build
docker compose run --rm bot config-check
docker compose run --rm bot fetch
docker compose run --rm bot backtest

docker compose up -d
docker compose logs -f bot
```

`config/` is mounted read-only; `logs/`, `data/cache/` and `reports/` are
mounted read-write so state, candles and the audit trail survive a rebuild. The
container runs as UID 10001, publishes no ports, and has a healthcheck that
fails if the configuration stops validating.

```bash
docker compose down                          # stop (45s grace period)
docker compose build --no-cache && docker compose up -d   # update
```

### Operational checklist

| Check | Command |
|---|---|
| Is it alive? | `systemctl status quantbot` / `docker compose ps` |
| What is it doing? | `journalctl -u quantbot -f` / `docker compose logs -f` |
| Any errors? | `tail -100 logs/error.log` |
| Why no signals? | `jq -r .gate logs/signal.jsonl \| sort \| uniq -c \| sort -rn` |
| Current risk state? | `jq . logs/state.json` |
| Is a halt active? | `jq -r .risk.halt logs/state.json` |

Back up `logs/state.json` before any update: it holds the risk counters and any
open paper positions.

---

## Design decisions and honest limitations

### Why there is no execution module

Placing orders is a different engineering problem from finding setups, and the
gap between them is where accounts die: partial fills, rejected orders, reduce-
only semantics, position-mode mismatches, leverage settings, exchange downtime
mid-position, and reconciliation after a restart. None of that is exercised by a
backtest.

So `app.mode` accepts `signal` and `paper` only, and the configuration layer
refuses `live` with an explanation rather than leaving a tempting switch.

If you add one later, the seam is clean — `RiskManager.approve()` already returns
an exact quantity. What that module must handle before touching real money:
idempotent order submission, reconciliation of exchange state against
`logs/state.json` on restart, partial-fill accounting, a kill switch, and its own
tests. Do it after a paper-trading period, not before.

### Known limitations, stated plainly

| Limitation | Detail |
|---|---|
| **Buy/sell volume is estimated** | True delta needs trade-level data (`aggTrades`). `effort_split()` distributes volume by where the close sits in the bar's range. Fine for relative comparison, not measured order flow. |
| **No liquidation feed** | Binance exposes forced liquidations only over the `!forceOrder` websocket; there is no REST history and aggregators are paid. `data.liquidations` defaults to `false` and `fetch_liquidations()` is a documented no-op rather than a scraper that breaks silently. |
| **News scoring is a lexicon** | Not sentiment analysis. CryptoPanic's own bullish/bearish votes are preferred because they avoid guessing. Off by default. |
| **Historical sentiment is not replayed** | See [Backtesting](#backtesting). |
| **Indicators span data gaps** | If the exchange has a hole, indicators are computed across it as if the bars were contiguous. The repository detects and warns about gaps. |
| **Single-venue correlation** | The correlation limit is a crude same-direction cap, not a correlation matrix. Three long alts is one bet; the bot knows that much and no more. |
| **Paper fills differ by one bar** | The backtest fills at the next bar's open; paper mode marks the signal bar's close plus slippage. Documented, not hidden. |
| **Every timeframe needs history** | A timeframe used by the MTF gate needs ~240 of its own bars to seed EMA-200. A `1d` context therefore needs ~240 days. Too little history fails **closed** — no signals — and the engine warns about it explicitly at start-up. |

---

## Testing

```bash
make check        # ruff + pytest
pytest -q         # 182 tests, ~100s
pytest tests/test_no_lookahead.py -v    # the ones that matter most
```

`tests/test_no_lookahead.py` is the heart of the suite. It computes each probe
bar **twice** — once from the full series as the backtester does, once from the
series truncated at that bar as the live bot sees it — and asserts that all 74
enriched columns, the structure report, the regime, the volume read and the
strategy's final verdict are identical. Any code that ever starts reading the
future fails there, including code added long after this was written.

Three real bugs were found by writing these tests, all fixed:

1. `update_breakers()` marked equity to market but the two brake calculations
   re-fetched equity *without* the marks — an open, underwater position was
   invisible to both circuit breakers.
2. A hammer with a near-zero body (a dragonfly — the strongest form) was
   excluded because it also matched the doji test.
3. `find_swings()` can legitimately report the same bar as both a fractal high
   and a fractal low (an outside bar), so callers must key by `(position, kind)`.

---

## Future work

Only after the basics are validated by a backtest **and** a paper-trading period.
Adding machine learning to a system with no demonstrated edge just produces a
more complicated way to lose money.

**Worth doing first — they improve the foundation:**

- Store the funding-rate series alongside candles so backtests can replay
  sentiment instead of marking it unavailable.
- Real order flow from `aggTrades`, replacing the `effort_split()` estimate.
- Portfolio-level correlation from a real return-correlation matrix.
- Monte-Carlo / bootstrap resampling of the trade sequence, to get a confidence
  interval on expectancy instead of a single number.
- Multi-symbol backtesting in one pass, so portfolio drawdown is measured rather
  than inferred from single-symbol runs.

**Optional, genuinely later:**

- **ML meta-labelling** (López de Prado): keep this rule-based system as the
  entry generator and train a classifier only on *whether to take* each signal,
  using the features already computed. Far more tractable than predicting price,
  and it degrades gracefully — a bad model just filters nothing.
- **Reinforcement learning** for position sizing or exit timing. Be aware that
  RL on financial series overfits enthusiastically and needs walk-forward
  validation at minimum.
- **Statistical strategies** — pairs / cointegration, volatility-regime models —
  as *separate* strategies behind the existing `Strategy` interface, not as
  changes to this one.

The `Strategy` ABC exists precisely so an alternative can be registered in
`strategies/__init__.py` and backtested against the same harness.

---

## Licence

MIT. Provided as-is, with no warranty. Trading carries risk of total loss; you
are responsible for anything you run.
