# Round 7 — macro / cross-asset drivers (2026-11-08)

New feature data from MT5 (signals only): `usdx-dec26` (dollar index futures, back-adjusted,
2012-11+), `vix-nov26` (VIX futures, 2020-12+; NOT back-adjusted 2021-01→2023-02 — roll gaps on
expiry Wednesdays, handled in `bot/strategies/mx_common.py`), `usdjpy` (2010+), `oilcash` (2010+).
XM's bond ETFs (TLT/IEF/TIP/SHY) only have 2026 history — unusable.

Result: **no new finalist** (184 configs, registry `results/registry_v2/mx_*.csv`).
- Dollar-trend gates on gold trend_donchian lowered t (3.17 → 2.0–2.5); standalone gold-after-
  dollar-drop failed (best t 1.92).
- Buy S&P after VIX spikes: only ~3–20 train trades on 2021+ data; val flat — sample too small.
- VIX-gated RSI2 (observation only): no gate improved it.
- Yen risk-off → S&P/BTC: following loses; fading ≈ plain long drift (fails coin flip).
- Oil → gold/BTC: nothing passed the quiz.

Running total ≈ 7,500 configurations across rounds 4–7. Confirmed set unchanged:
US500 RSI2 dip-buyer (exam pass), GOLD trend_donchian (all checks, no clean exam),
JP225 pullback (exam pass on 11 trades). Next proof = forward test:
`run_portfolio.bat` on demo + `python tools/forward_report.py` (expectations in
`results/forward_expectations.json`).
