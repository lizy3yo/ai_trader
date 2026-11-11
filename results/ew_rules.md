# ew_ — Elliott Wave rules as commonly taught (pre-registered 2026-11-08)

Written BEFORE any backtest. Every **[choice]** is a place where sources disagree or are silent;
the value was fixed here first (most common value where sources disagree). Grids only vary the
published knobs around these defaults. Code: `bot/strategies/ew_common.py` (ZigZag + wave
scanner), `ew_w3.py`, `ew_w5.py`, `ew_abc.py`. Causality: `results/ew_causal_check.py`
(truncation test). Diagnostic: `results/ew_gross.py`.

## Sources
- Hard rules (Frost & Prechter, *Elliott Wave Principle*, as summarised):
  https://www.luxalgo.com/library/concept/elliott-hard-rules.md ,
  https://wavebasis.com/docs/frequently-asked-questions/automatic-wave-counts/what-are-the-core-elliott-wave-rules/ ,
  https://www.nasdaq.com/articles/ellliott-wave-basics-variations-motive-waves-2011-06-28
- Fibonacci guidelines: https://jp.tradingview.com/chart/BTCUSD/JW9LFJKW-Elliot-Waves-Complete-Guide-Chapter-4-4-Fibonacci-Ratios
  (w2 50/61.8/76.4/85.4% of w1; w4 38.2-61.8% of w3),
  https://jp.tradingview.com/chart/CRUDEOIL1%21/u6QWFgV0-Elliot-Wave-Explanation-I-live-Example-Crudeoil
  (w4 14.6/23.6/38.2% of w3; w5 = w1 when w3 extends),
  https://kr.tradingview.com/chart/US100/hyEWr6fa-Elliott-Wave-Analysis-and-Fibonacci-Projections-for-US-100
  (w2 38.2-61.8%), https://fxopen.com/blog/en/how-to-apply-the-elliott-wave-method-in-trading/amp/
  (w2 0.618, w4 0.382, w3 target 1.618 x w1, w1 = w5 if w3 longest).
- Zigzag A-B-C: https://www.tradingview.com/chart/BTCUSD/YaMxVSes-Elliot-Waves-Complete-Guide-Chapter-4-6-ABC-Fib-Lengths/
  (C usually = A, also 0.618 / 1.618 x A), https://www.tradingview.com/chart/EURUSD/loGnMGEJ-Part-4-Corrective-Waves-Simple-Zigzags-Sharp-Corrections
  (B 38.2-78.6% of A), https://www.mql5.com/en/blogs/post/775112 .
- Trade setups (wave-3 / wave-5 entries, stops, targets): https://tradetheday-com.pages.dev/guides/elliott-wave ,
  https://www.tradingview.com/chart/BTCUSDT/Gbp9eX4O-Bitcoin-Elliot-5th-Wave-Swing-Long/ ,
  https://it.tradingview.com/chart/NIFTY/RowiB8h2-How-to-Trade-Elliott-Wave ,
  https://tr.tradingview.com/chart/RUNEUSD/dhHP6la8-RUNE-USD-Elliott-Wave-Analysis-Bullish-Wave-3 ,
  https://journalplus.co/strategies/elliott-wave-trading , https://www.asktraders.com/learn-to-trade/technical-analysis/how-to-trade-elliot-waves/
- Open-source automation: Elliott Wave [LuxAlgo] (ZigZag pivots, lengths 4/8/16)
  https://my.tradingview.com/script/KvrhsPTp-Elliott-Wave-LuxAlgo ; HKUDS/Vibe-Trading elliott-wave
  skill (zigzag swings, fib_tolerance 0.15, w2 0.5-0.618, w4 0.382, w5 = w1; swings stamped on
  the confirmation bar) https://tessl.io/registry/skills/github/HKUDS/Vibe-Trading/elliott-wave ;
  ZigZag ATR Pine library (ATR 14, multiplier 2.0 default; 3.0 = major pivots only)
  https://in.tradingview.com/script/v8XJuorH-ZigZag-ATR/ ; MT4 ZigZag 12/5/3 (repaints its last leg,
  so not used).

## 0. Swings: causal ATR ZigZag (ew_common.zigzag)
- ATR(14) (`cfg.atr_period`) at the current bar. State: leg direction and running extreme.
- Up-leg: a new high replaces the extreme; otherwise if `low <= extreme - k x ATR` the extreme is
  CONFIRMED as a pivot high at this bar and a down-leg starts (extreme = this bar's low). Mirror.
  A bar that makes a new extreme does not also reverse (intrabar order unknown, conservative).
- A pivot is known only at its confirmation bar (never at the pivot bar). No repainting: once
  confirmed it never moves. **k default 2.0** (ZigZag ATR library default); grid {2, 3}
  (3 = "major pivots"). [choice] ATR rather than % threshold so one value works on M15..D1 and
  on all three markets.

## 1. Hard rules (applied wherever the waves exist)
R1 wave 2 never retraces beyond the start of wave 1 (L2 > L0 for longs).
R2 wave 3 is never the shortest of 1/3/5. [choice] For the wave-5 trade, where wave 5 is unknown,
   require W3 >= W1 (if W3 < W1, wave 5 would have to be < W3 < W1, contradicting the
   equality target).
R3 wave 4 does not enter wave 1's territory (L4 > H1 for longs). Diagonals not modelled.

## 2. ew_w3 — "wave 3" entry (long; short = mirror)
- Pivots L0 (low), H1 (high), L2 (low) confirmed in that order. W1 = H1 - L0.
- Wave-2 retracement r = (H1 - L2)/W1 in band. **Default 0.50-0.618** (most cited "typical"
  zone); alternative 0.382-0.786 (the common wider band). r < 1 enforces R1.
- Armed when L2 is confirmed; fires at the FIRST close > H1 (break of wave 1's extreme) before the
  next pivot is confirmed (a confirmed new pivot = the rally failed -> setup dropped). One trade
  per setup. [choice] close beyond the level (the backtester has no stop orders); fill next open.
- Stop: L2 - 0.1 x ATR (beyond wave 2's extreme; [choice] pad 0.1 ATR, min stop 0.3 ATR, as ms_).
- Target: **default "fib" = L2 + 1.618 x W1** (1.618 x wave 1 projected from wave 2; most common
  w3 objective); alternatives fixed 2R and 3R. Skip if the fib target is already reached.

## 3. ew_w5 — "wave 5" entry
- Pivots L0, H1, L2, H3, L4 confirmed. R1: L2 > L0; H3 > H1; R2: W3 = H3 - L2 >= W1; R3: L4 > H1.
- Wave-4 retracement (H3 - L4)/W3 in band. **Default 0.236-0.382** (brief + 23.6/38.2 guideline);
  alternative 0.146-0.618 (union of all sources' w4 depths).
- Entry: [choice] at the bar that CONFIRMS L4 (price has turned back up by k x ATR = "wave 4 ended
  and price breaks back in the trend direction"); fill next open. No separate break level is
  defined at the same swing degree.
- Stop: L4 - 0.1 x ATR (beyond wave 4's extreme; most common w5 stop). Target: **default "eq" =
  L4 + W1** (wave-5 equality with wave 1); alternatives 2R, 3R. Skip if the target is reached.

## 4. ew_abc — A-B-C zigzag correction in an established trend (long = pullback in an uptrend)
- Pivots Lpp, Hp, Lpre, H0, La, Hb, Lc. Established uptrend [choice] = Dow structure before the
  correction: H0 > Hp (higher high) and Lpre > Lpp (higher low).
- A = H0 - La; B retraces (Hb - La)/A in **0.382-0.786** (most common B band, fixed);
  C = Hb - Lc with Lc < La (zigzag: C exceeds A's end) and Lc > Lpre (the correction does not
  undo the last impulse leg, otherwise the trend is broken) — amended to Lc > Lpp, see below.
- C/A ratio: **default 0.85-1.15 ("C = A", tolerance 0.15 as in Vibe-Trading)**; alternative
  0.618-1.618 (full published range).
- Armed when Lc is confirmed (C complete); fires at the first close > Hb (break of B) before the
  next pivot is confirmed. Stop: Lc - 0.1 x ATR (beyond C).
- Target: no common structural target; **default 2R**, alternative 3R. [choice]
- **Amendment (before any P&L was seen, trade-count check only):** "Lc > Lpre" left 6 setups in
  33k GOLD H1 train bars and contradicts the EW guideline that corrections often end near the
  previous fourth wave (i.e. around Lpre). Replaced by **Lc > Lpp** (the correction must not undo
  the start of the HH/HL structure = 100% of the trend move). Still rare (~14 setups per side per
  33k H1 bars at the default band); that rarity is itself a finding.

## Tests / budget (fixed before running)
- Markets GOLD, BTCUSD, US500Cash (listed explicitly); TF M15, H1, H4, D1; real costs
  (research_*.yaml); `--workers 6`. max_bars per TF as ms_: M15 100000, H1 55000, H4 13500,
  D1 60000 (avoids the daily-filler part of H1/H4 files).
- One study per (model, k, target) with grid {band: default, alternative} on 3 markets x 4 TFs =
  24 configs each: w3 2k x 3 targets, w5 2k x 3 targets, abc 2k x 2 targets = 16 studies, 384
  configs, side "both" (random baseline dir both). Plus long-only studies of each model's default
  (k 2, default target, both bands) = 72 configs. Total 456.
- Baseline geometry: stops/targets are structural, so per (model, k) `cfg.atr_sl_mult` = the
  median structural stop in ATRs and, for structural targets, `cfg.atr_tp_mult` = the median target
  in ATRs (measured on train data only by ew_causal_check.py); for fixed-R targets
  `cfg.atr_tp_mult = R x cfg.atr_sl_mult` and the strategy uses R = atr_tp_mult/atr_sl_mult.
  No exit_signal is used (SL/TP only), so exit_from_strategy = false.
- Zero-cost diagnostic: each model's default on train+val, real vs no spread/no swap.

## Results (added after testing, 2026-11-08; rules above unchanged except the ABC amendment)
456 configurations in 76 studies (`results/registry_v2/ew_*.csv`, log `results/ew_run.log`,
summary `python results/ew_summary.py` -> `results/ew_summary.txt`). Truncation test PASS
(`results/ew_causal_check.txt`). **0 finalists.** Stops: practice 259, quiz 185, gauntlet 12;
best train+val t 2.30 (GOLD M15 ew_w3 long-only, wide band 38.2-78.6%, k 2, 1.618 target:
471 trades, +0.125R; neighbour 0/1, below the 2.4 needed to even run the coin flip).
Only 43% of configurations were train-positive (coin-flip level). ABC is too rare to test
(median 2-23 train trades per market x TF). Zero-cost diagnostic of the defaults
(`results/ew_gross.py` -> `results/ew_gross.txt`, train+val, trade-weighted): gross avgR
w3 +0.013 (n 1611), w5 +0.079 (n 804), abc +0.022 (n 135); with real costs -0.067 / -0.032 / -0.031.
