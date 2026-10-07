import json

picks = [
    ("results/ses_orb_r4_m15.json", {"entry_end_h": 11}),
    ("results/ses_orb_r4_m15.json", {"entry_end_h": 10}),
    ("results/ses_orb_best.json", {"entry_end_h": 12}),
    ("results/ses_orb_r2_m15.json", {"buf_atr": 0.35, "entry_end_h": 14, "cfg.atr_sl_mult": 2.0, "cfg.atr_tp_mult": 2.25}),
    ("results/ses_orb_r2_m15.json", {"buf_atr": 0.5, "entry_end_h": 12, "cfg.atr_sl_mult": 2.5, "cfg.atr_tp_mult": 2.25}),
    ("results/ses_orb_r3_m30.json", {"sides": "both", "cfg.atr_tp_mult": 1.2}),
]

for path, match in picks:
    rows = json.load(open(path))
    for r in rows:
        if all(r["params"].get(k) == v for k, v in match.items()):
            print(json.dumps(r))
            break
