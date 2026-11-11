# Research protocol (round 4: GOLD + BTCUSD, 2026-11-08)

Shared brief for every strategy-research agent. Working dir: `C:\00-SideProject\AI_Trading`.

## Infrastructure (do not modify — report bugs instead)
- Data, fresh MT5 exports from the user's broker (XM Global, server clock = data clock,
  most likely EET: UTC+2 winter / UTC+3 summer, rollover at server 00:00):
  `data/gold_{m1,m5,m15,m30,h1,h4}.csv`, `data/btcusd_{m1,m5,m15,m30,h1,h4}.csv`,
  columns time,open,high,low,close,tick_volume,spread(points). Contract specs in `data/*_econ.json`.
  - GOLD: M5 since 2025-06, M15 since 2022-08, M30 since 2018-05, H1 real hourly since 2013-06, H4.
  - BTCUSD: M5 since 2025-11, M15 since 2024-01, M30 since 2020-10, H1 real since 2017
    (weekend bars only from 2022; before that the broker was weekdays-only), H4.
- Base configs with REAL costs: `research_gold.yaml`, `research_btc.yaml` — fixed 1% risk on
  $100k, spread floor = today's live spread (GOLD $0.55, BTC $40) or the bar's recorded spread if
  wider, overnight swap at server midnight (GOLD long −$78.59/lot/night, short +$16.81, Wed x3;
  BTC long −$32.20/BTC/night, short −$21.47, every night incl. weekends), scaled to the price then.
- Strategy contract: see `bot/strategies/__init__.py`, helpers in `bot/strategy.py`, example
  `bot/strategies/trend_donchian.py`. Optional exit hook:
  `exit_signal(ind_window, scfg, direction, bars_held) -> bool` (True = close at next bar open).
  Signals must carry sl_distance/tp_distance (> 0); a very wide TP + exit_signal = trailing exit.
- Harness: `python tools/research.py --config research_X.yaml --strategy NAME --timeframes H1,H4 --grid-file grids/F.json --out results/F.json`
  60/20/20 train/val/TEST split by bars (`--max-bars N` keeps the most recent N bars; default 60000).
  Grid keys: plain = strategy params; `cfg.X` = StrategyCfg (atr_sl_mult, atr_tp_mult, ema_trend,
  cooldown_bars, max_hold_bars…); `session.window` "HH:MM-HH:MM"; `session.flat_at` "HH:MM";
  `session.weekends` true/false; `backtest.spread_points` (cost stress). Max 200 combos per call.
  Output includes val avgR (R per trade, costs+swap included) and t-stat.
- Edge test: `python tools/edge_test.py --config research_X.yaml --strategy NAME --timeframe TF --params '<json>' --dir long --trend 0|1 --split trainval --seeds 30`
  compares against random entries with identical session/stops/holding/costs. p_random < 0.05 required.
- Whole-file view: `python tools/full_eval.py <a trading config yaml> --base research_X.yaml`.

## Rules
1. NEVER evaluate on the TEST split (`--final`, `--split test/all`). The lead runs locked tests.
2. Your files only: new modules `bot/strategies/<prefix>_*.py`, grids `grids/<prefix>_*.json`,
   results `results/<prefix>_*`. Do not edit any other file.
3. Keep grids small and economically motivated; log how many configurations you tried in total
   (data-mining budget). Prefer few parameters.
4. A candidate must have: train AND val profitable; val t-stat ≳ 1.5 and trainval t ≳ 2;
   edge_test p_random < 0.05 on trainval; neighbouring parameters also positive (no knife-edge);
   still positive with spread +50% (`backtest.spread_points`); at least ~40 val trades
   (fewer only for swing systems, flag it).
5. Shorts: test them, but they never paid on gold in previous rounds. Gold SHORTS are now paid swap.
6. Previously dead on gold (don't redo): mechanized ICT/SMC (FVG, order blocks, sweeps, Silver
   Bullet), supply/demand, plain M1/M5 scalps, RSI2 mean reversion, VWAP/pivots, MACD/ADX trend,
   GOLD-vs-XAUEUR arb, ML gradient boosting, grid/martingale. Existing gold systems:
   trend_donchian H1 (edge confirmed, p<0.001), ses_orb M15 (t 1.6), fast_channel M30 (t 1.2 after swap).
7. Final report (≤ 700 words): top 1–3 candidates with module name, timeframe, EXACT params JSON,
   train + val metrics (trades, PF, avgR, t, posM%, maxDD), edge-test result, cost-stress result,
   total configs tried, and the honest list of what failed.
