# Round 4 — GOLD + BTCUSD, scalp / day / swing (2026-11-08)

Data: fresh MT5 exports from XM (GOLD, BTCUSD), with recorded per-bar spread.
Costs: spread floor = today's live spread (GOLD $0.55, BTC $40) or the recorded bar
spread if wider; overnight swap (GOLD long −$78.59/lot/night, short +$16.81, Wed x3;
BTC long −$32.20/BTC/night, short −$21.47, every night), scaled to the price then.
Protocol: `results/RESEARCH_PROTOCOL.md` (60/20/20 train/val/locked test, random-entry
edge test, cost stress, neighbour robustness). ~590 configurations tried in total.

## Verdict: no new strategy confirmed

| Candidate | Train | Val | Edge test (trainval) | LOCKED TEST |
|---|---|---|---|---|
| BTC H4 20-day breakout, 10-day midpoint trail, 14-day cap (`btcsw_brk`) | 47 tr, +0.77R, t 2.1 | 16 tr, +0.69R, t 1.5 | p 0.000 | **16 tr, −0.004R, t 0.0** (BTC −10% in window) |
| BTC Monday weekend-follow, flat 23:30 (`btcid_window`, M30) | 107 tr, +0.135R, t 2.0 | 36 tr, +0.14R, t 1.5 | p 0.000 | **35 tr, −0.065R, t −0.6** |

Test windows: BTC H4 2025-03-16 → 2026-11-08; BTC M30 2026-03-03 → 2026-11-08.

## What failed before the locked test
- GOLD swing (`gsw_`): 12-month momentum / MA regime gates, trailing exits, lookback
  ensembles, shorts (even with paid swap), H4 pullbacks — none passed validation.
- GOLD intraday (`gid_`): Asian-session drift (sits in the untradeable first minute after
  the 01:00 reopen), Gao intraday momentum (below the spread), London ORB with gates (all
  40 configs train-negative). Near miss: NY-open break of the London-morning range, long,
  M15 (trainval t 1.59, regime-dependent: 2023–24 ≈ 0).
- BTC intraday (`btcid_`): post-rollover window, intraday TSMOM, NYSE-close 10-day high,
  Williams volatility breakout, range breakouts — all die on costs (effects 0.05–0.1%
  gross vs ~0.05–0.15% spread).
- BTC swing (`btcsw_`): SMA regime and Donchian ensembles are profitable but their entries
  do not beat random entries with identical exits (they are just "be long BTC").

## Findings about the EXISTING gold configs (with real swap)
- `trend_donchian` H1 (config.yaml): +0.25R/trade 2013–2026, beats random entries
  (p < 0.001), but swap costs ~0.05R/trade and validation 2021-06 → 2024-02 was flat
  (+0.02R); profits concentrate in the 2019–20 and 2024–26 gold rallies.
- `fast_channel` M30 (config_fast.yaml): +0.156R → +0.073R after swap, t 1.2 = no edge.
- `ses_orb` M15 (config_orb.yaml): train-negative on 2022–24 under real costs; its profits
  are 2024–26 only.
- Swap (~7%/yr of notional on gold longs, ~15%/yr on BTC) eats most multi-week long edges.

## Infrastructure added this round
`tools/export_bars.py` (any symbol, spread + econ json), backtester swap / bar spread /
weekends / flat_at / max_hold_bars / exit_signal / start_at warm-up, live-loop parity for the
new exits, `research_gold.yaml`, `research_btc.yaml`, `tools/edge_test.py` +
`bot/strategies/rnd_baseline.py`, `tools/full_eval.py`, research.py `--config` and R/t output.
