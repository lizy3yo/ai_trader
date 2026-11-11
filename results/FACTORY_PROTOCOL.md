# Factory protocol (round 5: every major family x 7 markets, 2026-11-08)

Shared brief for every strategy-library agent. Working dir: `C:\00-SideProject\AI_Trading`.
Read `results/RESEARCH_PROTOCOL.md` too (round 4: data, cost model, strategy contract), and
`results/ROUND4_SUMMARY.md` (what already failed on GOLD/BTC).

## Markets (all with real broker costs, `research_*.yaml`)
| market | config | real history | notes |
|---|---|---|---|
| GOLD | research_gold.yaml | H1 2013+, M30 2018+, M15 2022+, D1 2001+ | long swap ~7%/yr, shorts paid |
| BTCUSD | research_btc.yaml | H1 2017+, M30 2020-10+, D1 2013+ | 24/7 (weekend bars 2022+), swap ~15%/yr both sides |
| ETHUSD | research_eth.yaml | H1 2017+, D1 2016-11+ | spread ~0.13% of price, swap ~15%/yr |
| US500Cash (S&P 500) | research_us500.yaml | H1 2016+, M30 2018+, D1 2011+ | swap ~6.5%/yr long, Friday x3 |
| US100Cash (Nasdaq 100) | research_us100.yaml | H1 2016+, M30 2018+, D1 2011+ | same swap shape |
| SILVER | research_silver.yaml | H1 2013+, M30 2018+, D1 2002+ | long swap ~12%/yr, shorts paid |
| EURUSD | research_eurusd.yaml | H1 2011+, M30 2018+, D1 | long swap ~2.6%/yr, shorts paid |
Timeframes available per market: M5, M15, M30, H1, H4, D1 (`data/<market>_<tf>.csv`).
Server clock = EET (UTC+2/+3), US cash open 09:30 ET = server 16:30, London open = server 10:00.

## The factory (do not modify; report bugs to the lead)
`python tools/factory.py --study grids/<prefix>_<name>.json --workers 3`
Study JSON: strategy, family, markets ("all" or list), timeframes, grid (<= 64 combos per market x TF),
dir ("long"/"short"/"both", side of the random baseline), trend (0/1), exit_from_strategy (true if your
strategy uses exit_signal), max_bars (default 60000 most recent bars).
It runs every setting on every market x timeframe, logs EVERY configuration to
`results/registry/<study>.csv`, and applies the checks automatically:
practice (train avgR>0) -> quiz (val avgR>0, val t>=1) -> luck bar (train+val t >= 3.0) ->
coin flip (beats random entries, p<0.05) -> +50% spread still positive -> >=60% of grid neighbours positive.
Passing all = FINALIST. `python tools/leaderboard.py` summarises all studies.
Use `--workers 3` (four agents share a 12-core machine).

## Rules
1. NEVER touch the locked TEST split (`tools/final_exam.py`, `research.py --final`, `--split test/all`).
   The lead runs final exams once per finalist.
2. Your files only: `bot/strategies/<prefix>_*.py`, `grids/<prefix>_*.json`, `results/<prefix>_*`.
   Do not edit anything else (factory, backtester, configs, other agents' modules).
3. Write strategies GENERIC across markets: ATR-based stops (cfg.atr_sl_mult / cfg.atr_tp_mult), a
   `side` param ("long"/"short"/"both"), no hard-coded prices. Time-of-day logic in server time.
   Use `exit_signal` for indicator/time exits, and set "exit_from_strategy": true in the study so the
   random baseline gets the same exit.
4. Keep grids economically motivated and small (every configuration raises the luck bar for everyone).
   Run each idea on ALL 7 markets unless it is market-specific by construction (say why).
5. Indicators must be causal (row i uses rows <= i only). Daily features from COMPLETED days only.
6. Report (<= 600 words): families/modules built, studies run, configurations tested (count from your
   registry files), every FINALIST with exact study/market/TF/params and its registry numbers, the
   near-misses (best train+val t that failed, and which check failed), and anything that looked broken.
