# Gold Bot — MT5 algo trader for XAUUSD

A Python trading bot for MetaTrader 5 that trades **gold only**, sized off a
**$500 allocated capital** that **compounds**, with a **daily loss circuit
breaker**, an **adaptive risk model** (starts at 5%/trade, steps up as you
win), and a **full trade journal** (SQLite + CSV) with daily/monthly reports.

> **Read this first.** This is educational software, not financial advice.
> Gold CFDs are leveraged products; most retail accounts lose money. A
> $200/day loss limit on $500 capital means you can lose 40% in one day —
> it is configured because you asked for it, but 5–10% ($25–$50) is the
> sane range. **No bot guarantees monthly profit.** Run on a DEMO account
> until the journal proves the edge to you.

## 1. One-time setup

1. **Install the MetaTrader 5 terminal** from your broker (or metatrader5.com)
   and log into a **DEMO account** with ~$500 balance. Any broker offering
   XAUUSD works (IC Markets, Pepperstone, Exness, FTMO-style, etc.).
2. In the terminal: **Tools → Options → Expert Advisors → check
   "Allow algorithmic trading"**.
3. Keep the terminal **running and logged in** while the bot runs.
4. Install Python dependencies (Windows only — the `MetaTrader5` package
   doesn't exist for Mac/Linux):

```bash
pip install -r requirements.txt
```

## 2. Configure

Everything lives in [config.yaml](config.yaml). The defaults implement what
you asked for:

| Setting | Value | Meaning |
|---|---|---|
| `capital.allocated` | 500 | The bot manages $500 + whatever it earns (compounding). It never sizes off your full account. |
| `risk.base_risk_pct` | 5% | Starting risk per trade (of bot equity). |
| `risk.step_up_after_wins` / `step_up_pct` | 3 wins / +1% | The "gradually increase" rule: 3 consecutive wins → 6%, then 7%… capped at `max_risk_pct` (10%). One loss → back to 5%. |
| `risk.daily_loss_limit` | 200 | Hard daily stop: realized + floating P&L ≤ −$200 → close everything, no trades until tomorrow. |
| `symbol` | XAUUSD | Your broker may call it `GOLD` or `XAUUSD.m` — run `find-symbol`. |

## 3. How to actually run it

**Every time, in this order:**

1. **Open MetaTrader 5 and log in.** The bot talks to the terminal — if MT5
   is closed, nothing works. Leave it open the whole time.
2. **Tools → Options → Expert Advisors → tick "Allow algorithmic trading".**
   One-time setting, but check it after MT5 updates.
3. **Start the bots.** Easiest way — double-click **`run_all.bat`**, which
   opens one window per strategy. Or start them individually:

```bash
python main.py live
```

```bash
python main.py live --config config_fast.yaml
```

```bash
python main.py live --config config_orb.yaml
```

4. **Leave it running.** Start Monday, stop Saturday. Each bot enforces its
   own trading hours, so running it outside them costs nothing — but the
   daily loss breaker only protects you while the bot is alive. Disable
   Windows sleep (Settings → Power → Screen and sleep → Never), or use a VPS.

**To stop:** Ctrl+C in the window, or just close it. Open positions keep
their stop-loss and take-profit on the broker's server, so they still exit
correctly with the bot off — you just lose the daily breaker and journaling
until you restart (the journal catches up automatically).

### Account size — read this before you wonder why nothing trades

The broker's smallest position is 0.01 lots = 1 ounce of gold. With gold near
$4,500 and volatility high, that one ounce risks more than a small account is
allowed to lose on a single trade, so **the bot correctly refuses to trade**
rather than over-risking you. What you'll see in the log:

```
[risk] no trade: even min lot 0.01 risks $54.01 (> 1.5x intended $14.90) — skipping trade
```

That is the safety system working, not a bug. Minimum account equity needed
at current volatility:

| Strategy | Needs (proper 5% risk) | Bare minimum |
|---|---|---|
| `ses_orb` | ~$400 | ~$260 |
| `fast_channel` | ~$885 | ~$590 |
| `trend_donchian` | ~$1,080 | ~$720 |

**For demo testing, top your demo account up to ~$3,000** (free, in your
broker's portal or by opening a new demo) **and set `capital.allocated: 3000`
in all three configs** — otherwise you'll only ever see `ses_orb` trade and
you won't learn anything about the other two.

## 4. Commands

```bash
python main.py find-symbol
```
Lists the gold symbols your broker actually offers — put the right name in
`config.yaml`.

```bash
python main.py backtest
```
Backtests the strategy on your broker's own price history (pulls ~20,000
M15 bars through the terminal), using the **exact same** strategy and risk
code as live. Every simulated trade is written to the journal with
`mode='backtest'`. **Do this first, always.**

The simulation is deliberately conservative: entries only on the very next
bar inside the session, spread paid on fills, a bar touching both SL and TP
counts as a stop, gaps through the stop fill at the open (not the stop
price), the daily breaker uses the bar's worst intrabar excursion, and
drawdown includes floating losses.

No MT5 terminal yet? You can exercise the whole pipeline offline on
synthetic bars (mechanics only — the numbers mean nothing):

```bash
python tools/make_sample_data.py
```

```bash
python tools/selftest.py
```

```bash
python main.py live
```
The trading loop. Connects to the terminal, and on every new closed M15 bar:
checks the daily breaker → session window → spread → signal → sizes the lot
off current bot equity → sends a market order **with SL and TP attached**
(so the stop lives on the broker's server even if your PC dies). Every fill
and close is journaled automatically.

```bash
python main.py status
```
Account equity, bot equity, today's P&L vs the daily stop, current adaptive
risk %, open positions, spread.

```bash
python main.py report
```
Your trade tracker: per-day and **per-month** P&L, win rates, best/worst
trades — for live and backtest separately. The raw journal is also mirrored
to `data/trades.csv` (opens in Excel).

## 5. Strategies (pluggable) and the research behind them

Strategies are modules in [bot/strategies/](bot/strategies/) selected by
`strategy.name` in the config — the same module drives live trading and
backtests. A multi-agent research program (6 strategy families, ~1000
backtests on real broker history, train/validation/locked-test splits,
independent causality audits) produced three demo-validation candidates:

| Config | Strategy | Win rate | Evidence |
|---|---|---|---|
| `config.yaml` (default) | `trend_donchian` H1 — long-only ~3-week breakout, SL 3×ATR / TP 5×ATR, London+NY | ~48% | Best: PF 1.6 train / 2.0 validation; 13.1y usable H1 history $500→$12.8k, PF 2.15, survived the 2013-15 bear |
| `config_fast.yaml` | `fast_channel` M30 — 24h channel breakout + anti-chase filter, SL 3.5×ATR / TP 5×ATR | ~47% | 4.7 trades/mo, 64% positive months, PF 1.18 over 7.8y — the fastest strategy that still works |
| `config_orb.yaml` | `ses_orb` M15 — London-open breakout of the Asia range, first two London hours only | ~60-68% | PF 1.63 train / 3.02 validation (only 19 val trades); +81% on full 4y |
| `config_rsi2.yaml` | `mr_rsi2` M30 — Asia-session RSI(3) dip buyer, TP 1×ATR / SL 2.5×ATR | ~73-79% | The high-win-rate profile — but ≈zero net edge on full 8.5y history |

Run all three simultaneously on demo (separate `magic` numbers and journals):

```bash
python main.py live
```

```bash
python main.py live --config config_fast.yaml
```

```bash
python main.py live --config config_orb.yaml
```

Session findings from the research (broker-server clock): the profitable
hours are the **first two London hours (09-11)** for breakouts and the
**quiet Asia session (01-08)** for mean reversion; the high-volatility
London/NY overlap (15-18) was unprofitable chop for every family tested;
**shorts never paid** anywhere. A config that looked spectacular on
validation (84% win rate) **failed the locked final test** (-17.7%) — which
is why none of these numbers is a promise, and demo forward-testing decides.

Research tooling: `tools/research.py` (train/val/locked-test harness),
`tools/sessions.py` (per-hour/per-session P&L), `tools/export_bars.py`.

## 6. Suggested path to real money

1. `backtest` — is the strategy even positive on recent history? Tune, repeat.
2. `live` on **demo for 4+ weeks** — does `report` show the same behavior?
3. Only then consider a small real account — and drop `daily_loss_limit`
   to $25–50 first. Never fund it with money you can't afford to lose.

## Files

```
main.py            CLI + live trading loop
config.yaml        all settings
bot/config.py      config loading/validation
bot/strategy.py    signals (EMA/RSI/ATR) — pure pandas, backtestable
bot/risk.py        compounding equity, adaptive 5%→10% risk, daily breaker, lot sizing
bot/mt5_client.py  all MetaTrader5 API calls
bot/tracker.py     SQLite + CSV trade journal, daily/monthly aggregates
bot/backtest.py    event-driven backtester reusing live strategy/risk code
data/              trades.db + trades.csv appear here once you run it
```
