# ms_ — market structure / SMC / supply-demand rules (pre-registered 2026-11-08)

Written BEFORE any backtest. Every value marked **[choice]** is a place where sources disagree or
are silent; the value was fixed here first and grids only vary the published knobs around it.
Code: `bot/strategies/ms_common.py` (engine), `ms_smc.py`, `ms_trend.py`, `ms_sd.py`.
Causality: `results/ms_causal_check.py` (truncation test, all columns identical on shared rows).

## Sources
- LuxAlgo "Smart Money Concepts" (open source, CC BY-NC-SA 4.0):
  https://www.tradingview.com/script/CnB3fSph-Smart-Money-Concepts-SMC-LuxAlgo/ ;
  v1 code copy: https://gist.github.com/deeppanara/93863bafe965047b9ac36cb01e7c397f ;
  LuxAlgo library pages https://www.luxalgo.com/library/indicator/smart-money-concepts-smc.md ,
  https://www.luxalgo.com/library/indicator/SRsr9SLs-smart-money-concepts (order-block concept).
- ICT "2022 model" write-up: https://www.luxalgo.com/library/concept/model-2022.md
- SMC entry model (HTF bias -> sweep -> discount -> LTF CHoCH -> OB): 
  https://backtrex.com/en/blog/what-is-smart-money-concepts-trading
- Python port with the same definitions: https://github.com/joshyattridge/smart-money-concepts
- HH/HL structure trading: https://daytradingtoolkit.com/strategies/higher-high-higher-low-market-structure-strategy
- Supply/demand (Seiden / OTA): https://www.eltraderfinanciado.com/en/blog/supply-and-demand-zones ,
  https://www.luxalgo.com/library/concept/zone-scoring-and-refinement.md ,
  https://www.luxalgo.com/library/concept/supply-and-demand-zones/
Videos could not be watched; these written sources and the Pine code are the reference.

## 1. LuxAlgo structure engine (ms_common.structure)
- **Swing point (leg)**: at bar t, `high[t-size] > highest(high, size)` (bars t-size+1..t) starts a
  bearish leg and makes bar t-size a pivot HIGH; `low[t-size] < lowest(low, size)` makes it a pivot
  LOW. Known only at the close of bar t (= `size` bars after the pivot). LuxAlgo defaults:
  **swing size 50, internal size 5**.
- **BOS / CHoCH**: close crosses above the current, not-yet-broken pivot high -> bullish; tagged
  CHoCH if the trend was bearish, BOS otherwise (mirror for lows). Each pivot breaks once.
  Internal breaks of a level equal to the swing level are ignored (LuxAlgo `extraCondition`).
  Confluence filter: LuxAlgo default OFF -> off.
- **Order block** (LuxAlgo storeOrderBlock): on a bullish break at t, the bar in [broken pivot
  bar, t-1] with the lowest "parsed low"; bars with range >= 2 x ATR(200) have high/low swapped
  (volatility filter "Atr", default). OB = that bar's high..low. Mirror for bearish.
- **Mitigation**: LuxAlgo default "High/Low" (wick through the far edge removes it). Here: setup
  invalid once low < the stop level (leg extreme, at or below the OB low).
- **FVG** (LuxAlgo): bullish at t if low[t] > high[t-2] and close[t-1] > high[t-2] and the middle
  bar's % body exceeds the auto threshold 2 x cumulative mean |%body| (default "Auto Threshold" on).
- **Premium / discount**: trailing top resets to each swing pivot high then tracks max(high);
  bottom likewise. LuxAlgo draws premium as the top 5% and discount the bottom 5% of that range;
  the YouTube/blog convention is above/below the 50% equilibrium. **[choice] 50% equilibrium**
  (most common in teaching; the 5% bands would leave almost no trades).
- Equal highs/lows (3 bars, 0.1 x ATR): documented, not used as a trigger.

## 2. SMC / ICT complete entry model (ms_smc)
1. Bias: swing-layer trend (size 50) on the chart, or **[choice]** HTF structure from completed
   higher-timeframe bars, pairing **M15->H4, H1->D1, H4->D1** (sources: H4/D1 bias with M15/M5
   entries; D1 bias with H1 entries for beginners), HTF pivot size **[choice] 5**.
2. Liquidity sweep: the leg broken by the CHoCH made an extreme beyond the previous internal pivot
   (for longs: its low is below the last internal pivot low that preceded the broken high).
   **[choice]** internal pivots as the liquidity pool (session highs/PDH sweeps were tested in lq_).
3. LTF CHoCH: internal-structure (size 5) CHoCH in the bias direction. BOS-only breaks don't trigger.
4. POI: the internal OB of that CHoCH (default) or the last LuxAlgo FVG inside the CHoCH leg.
5. Discount/premium: the POI's entry edge must be below (longs) / above (shorts) equilibrium of the
   bias dealing range at the time of the touch; bias must still agree.
6. Entry: first bar whose range reaches the POI's proximal edge and closes back inside/beyond it
   (zone held) -> fill at the next open. **[choice]** first touch only; **[choice] setup expiry
   20 bars** after the CHoCH (ICT: entry soon after displacement; LuxAlgo keeps OBs until
   mitigated). The backtester has no limit orders, so this is a touch-and-close entry, not a
   resting limit at the OB edge.
7. Stop: beyond the swept leg extreme + **[choice] 0.1 x ATR(14)** pad; floor 0.3 x ATR.
8. Target: fixed RR (**2R and 3R**, published) or "liq" = opposing liquidity = dealing-range
   extreme (skipped if < 1R).
Grid: swing size {25, 50, 100}, RR {2, 3}, bias {swing, htf (HTF size 3/5)}, TF {M15, H1, H4};
ablations sweep/discount on/off; POI OB vs FVG; target RR vs liquidity.

## 3. Classic market-structure trend (ms_trend)
- Pivots = LuxAlgo legs of size **{2 (daytradingtoolkit: 2 candles each side), 5 (LuxAlgo
  internal), 10}**.
- Long: trend bullish (latest break = close above a pivot high) AND that break came after the
  previous pivot low, AND a new pivot low is confirmed that is higher than the previous one (HL).
  Entry at the confirmation bar's close **[choice]** (source uses a buy-stop above the
  confirming candle; same bar in practice).
- Stop: below the HL - 0.1 x ATR. Target **2R (most common)** / 3R, or structure trail: exit on a
  close below the latest confirmed pivot low (source's exit rule). Shorts mirror. TF M15/H1/H4/D1.

## 4. Supply & demand, Seiden / OTA (ms_sd)
- Base: **1-6** consecutive boring candles (body <= **50%** of range). Sources: 1-6 (most common),
  3-8 (one source) -> 1-6.
- Leg-out: next candle exciting (body >= **50%** of range; sources 50-60%), in the leg direction,
  closing beyond the base extreme, range >= strength_k x ATR **[choice] k in {1.0, 1.5}** (OTA
  "strength" has no published number). Leg-in: any (RBR/DBR, DBD/RBD).
- Zone: proximal = highest base body (demand) / lowest body (supply); distal = extreme wick.
- Freshness: first return only (touch consumes); close beyond distal kills; **[choice] expiry
  200 bars**; up to 10 fresh zones per side.
- Entry: first touch that does not trade through the distal; fill next open.
- Stop: distal + 0.1 x ATR; target **3R (OTA minimum 3:1)** and 2R.
- Odds enhancers (enh "all"): trend = HTF close vs SMA(20) of HTF closes (Seiden 2012 used a
  20-period average); curve = proximal in the lower (demand) / upper (supply) third of the last
  **[choice] 50** HTF bars' range. HTF pairing as in 2. "none" = raw zones.

## Costs / tests
Real broker spread + swap (research_*.yaml), factory checks unchanged. RR is set through
cfg.atr_tp_mult / cfg.atr_sl_mult so the random baseline gets the same reward:risk;
cfg.atr_sl_mult is set to the strategy's median structural stop (~1.0-1.5 ATR, measured on GOLD
train data by ms_causal_check.py). Markets: GOLD, BTCUSD, US500Cash. Budget 300-500 configs.

## Results (added after testing, 2026-11-08; rules above unchanged)
438 configurations in 37 studies (`results/registry_v2/ms_*.csv`, logs `results/ms_run1.log`,
`results/ms_run2.log`, summary `python results/ms_summary.py`). **0 finalists**; only 9 reached
the gauntlet, best train+val t 1.86 (GOLD H4 ms_trend long, length 2, 2R; neighbours 0/1).
Run with `--workers 6` as the brief asked (ROUND6_PROTOCOL says 4).
- The complete SMC model is rare: about 1 trade per 1,000 bars (GOLD M15 104 trades in 80k bars;
  H4 5-13 train trades, which can't be tested).
- `results/ms_gross.py` -> `results/ms_gross.txt` (train+val, default settings, real vs zero cost):
  mean avgR real / gross: SMC -0.155 / +0.007, structure trend -0.101 / +0.015,
  S&D -0.238 / -0.128. No gross edge to protect; costs then turn ~0 into a clear loss
  (BTC M15/H1 worst: -0.3 to -0.6R per trade).
