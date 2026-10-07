import json

picks = [
    ("results/hyb_ema_regime_s5.json",
     {"adx_min": 25, "atr_pct_min": 0.25, "cfg.atr_sl_mult": 3.0, "cfg.atr_tp_mult": 1.25}),
    ("results/hyb_ema_regime_s5.json",
     {"adx_min": 25, "atr_pct_min": 0.35, "cfg.atr_sl_mult": 3.0, "cfg.atr_tp_mult": 1.75}),
    ("results/hyb_ema_regime_s5.json",
     {"adx_min": 25, "atr_pct_min": 0.45, "cfg.atr_sl_mult": 3.0, "cfg.atr_tp_mult": 1.25}),
    ("results/hyb_ema_regime_s5.json",
     {"adx_min": 22, "atr_pct_min": 0.25, "cfg.atr_sl_mult": 3.0, "cfg.atr_tp_mult": 1.25}),
    ("results/hyb_ema_regime_s5.json",
     {"adx_min": 22, "atr_pct_min": 0.45, "cfg.atr_sl_mult": 3.0, "cfg.atr_tp_mult": 1.25}),
    ("results/hyb_ema_regime_s3b.json",
     {"adx_min": 0, "atr_pct_min": 0.35, "mode": "long", "session.window": "15:00-19:00",
      "cfg.atr_sl_mult": 2.0, "cfg.atr_tp_mult": 1.25}),
]
for path, want in picks:
    rows = json.load(open(path, encoding="utf-8-sig"))
    for r in rows:
        if all(r["params"].get(k) == v for k, v in want.items()):
            print(json.dumps(r))
            break
    else:
        print("NOT FOUND", want)
