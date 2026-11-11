# Round 5 — strategy factory, 7 markets (2026-11-08)

Markets (real XM costs, `research_*.yaml`): GOLD, SILVER, BTCUSD, ETHUSD, US500Cash,
US100Cash, EURUSD. Protocol: `results/FACTORY_PROTOCOL.md`. Factory: `tools/factory.py`,
registry `results/registry_v2/` (re-run after the backtester fixes below),
`tools/leaderboard.py --registry results/registry_v2`.

## Numbers
- 4,978 configurations (strategy x market x timeframe x setting), 14 families.
- Pass mark raised to train+val t >= 3.0 (≈250 lucky "winners" expected at 5%).
- Stopped at: practice 3,118 · quiz 1,485 · luck bar 235 · FINALIST 33.
- 33 finalists = 5 effects. Fair coin-flip re-check (`tools/verify_finalists.py`,
  random entries never placed where the strategy's own exit fires at once): all pass
  except EURUSD trend on 2001+ data (t 1.58 -> dropped).

## Final exams (`tools/final_exam.py`, `results/final_exams.csv`; one try each)
| Strategy | Test window | Trades | avgR | t | p_random | Verdict |
|---|---|---|---|---|---|---|
| US500 RSI2 dip-buyer (lm_meanrev, D1, no trend filter) | 2023-09 → 2026-11 | 65 | +0.108 | 3.05 | 0.000 | **PASS** |
| US100 IBS dip-buyer (trend filter) | 2023-09 → 2026-11 | 43 | +0.032 | 0.45 | 0.60 | FAIL |
| BTC H4 Donchian trail long | 2025-02 → 2026-11 | 23 | +0.047 | 0.18 | 0.47 | FAIL |
| ETH H4 Donchian trail long | 2025-02 → 2026-11 | 21 | +0.427 | 0.83 | 0.10 | FAIL |
| US100 M30 Williams breakout | 2025-02 → 2026-11 | 293 | −0.040 | −0.42 | 0.50 | FAIL |
| BTC turn of month | 2024-09 → 2026-11 | 26 | −0.165 | −1.82 | 0.93 | FAIL |

Winner over its whole history (2011–2026, 1% risk, real costs): 292 trades, 73% wins,
PF 1.88, +0.084R/trade, t 4.05, max DD 4.9%; losing year 2018 (−3R), flat 2020/2022.
$500 at the user's 5–10% compounding risk: $500 → $1,699 over 15 years (~8.5%/yr).
Config: `config_us500_rsi2.yaml` (demo first).

## Backtester fixes this round
- Fill on the next bar index (live trades a Friday-close signal at Monday's open).
- Session for a new signal judged when the next bar appears (Friday D1/H4 signals were dropped).
- Spread floor scales with price (`spread_ref_price`), like swap — old crypto history was
  charged today's $40 BTC spread.
- `session.flat_at` closes inside the bar that contains the flat time.
- Validation runs warm up indicators on the preceding train bars (`start_at`).
- Random baseline: `dir both`, borrows the strategy's exit, never enters where that exit fires at once.

## Known data caveats
EURUSD has no H4 file and its D1 history starts in 1971 (synthetic before ~1999). H1/H4
files of indices (pre-2016), metals (pre-2013) and crypto (pre-2017) start with daily filler bars.
