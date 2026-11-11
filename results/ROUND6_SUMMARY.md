# Round 6 — gold focus + stock-index extension (2026-11-08)

Protocol: `results/ROUND6_PROTOCOL.md`. Registry: `results/registry_v2/` (g6i_*, g6s_*, ix_*).
Markets added: US30, US2000, GER40, UK100, JP225, EU50, FRA40, AUS200, HK50, CA60 (17 total).

## Results
- **Gold intraday (g6i_, 455 configs):** tick-volume breakouts / climaxes / quiet-to-active,
  CPI-NFP-FOMC post-release momentum and pre-event drift, NY-open break with pre-registered
  filters, Monday gaps, round numbers, extreme-move fades — no finalist (best t 2.13).
- **Gold swing (g6s_, 154 configs):** no add-on improved trend_donchian (breakeven, trail,
  stale/opposite-channel exits, tick-volume and ATR% filters, event-day skip all lowered t).
  Baseline trend_donchian (session 09–23) scores train+val t 3.17, p_random 0 — passes every
  automatic check; no clean final exam possible (2024–26 test years were seen in round 4).
  Event union (pre-FOMC + pre-holiday, t 3.54) rejected: FOMC profit came from 13 trades whose
  fixed 21:00-server exit fell after the statement in DST-mismatch weeks; with the exit anchored
  to 14:00 New York it drops to t 2.77.
- **Indices (ix_, 310 configs):** the US500 RSI2 rule, unchanged, is positive on only 3 of 10
  new indices (sign-test p 0.95) — the US500 edge does not generalise. One finalist:
  JP225 D1 lt_pullback (RSI(3)<10 above EMA200, recovery exit SMA5, 20-bar cap).

## Final exam
| Strategy | Window | Trades | avgR | t | p_random | Verdict |
|---|---|---|---|---|---|---|
| JP225 RSI pullback (D1) | 2023-09 → 2026-11 | 11 | +0.161 | 1.27 | 0.033 | PASS (weak: 11 trades) |

## Portfolio of the best-evidence strategies (`results/r6_portfolio.py`, whole history, in-sample)
| Set | R/month | $500/mo at 1% risk | at 2% | maxDD at 2% |
|---|---|---|---|---|
| US500 RSI2 | 0.126 | $396k | $198k | 9% |
| US500 + GOLD donchian | 0.543 | $92k | $46k | 12% |
| US500 + JP225 + GOLD donchian | 0.598 | $84k | $42k | 14% |
Monthly R correlations: US500/JP225 0.03, US500/GOLD −0.16, JP225/GOLD 0.00.
Launcher: `run_portfolio.bat` (demo first). New config: `config_jp225_pullback.yaml`.

## Fix this round
MT5 `trade_tick_value` is in the symbol's PROFIT currency for non-USD indices when no conversion
is live (JP225 reported 1.0 = ¥1, true $0.0063). `bot/mt5_client.account_tick_value()` now prices a
100-tick move with `order_calc_profit`; live sizing (`economics`) and `export_bars.py` use it, and the
saved `data/*_econ.json` of AUS200, CA60, EU50, FRA40, GER40, HK50, JP225, UK100 were corrected.
R-based research results were unaffected; $-sized backtests and live sizing on those indices were wrong.
