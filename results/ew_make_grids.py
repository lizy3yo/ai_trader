"""Write the pre-registered ew_ factory studies (grids/ew_*.json). See results/ew_rules.md.
Baseline geometry (cfg.atr_*) = train medians from results/ew_causal_check.txt."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MK = ["GOLD", "BTCUSD", "US500Cash"]
MB = {"M15": 100000, "H1": 55000, "H4": 13500, "D1": 60000}
# (model, k) -> (median SL ATR, median structural TP ATR or None)
GEO = {("w3", 2): (3.0, 4.0), ("w3", 3): (4.25, 6.0), ("w5", 2): (2.0, 2.4), ("w5", 3): (3.0, 3.9),
       ("abc", 2): (4.0, None), ("abc", 3): (5.25, None)}
names = []


def write(name, model, grid, side_dir):
    for tf in MB:
        st = {"strategy": f"ew_{model}", "family": "elliott", "markets": MK, "timeframes": [tf],
              "grid": grid, "dir": side_dir, "trend": 0, "exit_from_strategy": False, "max_bars": MB[tf]}
        f = ROOT / "grids" / f"ew_{name}_{tf.lower()}.json"
        f.write_text(json.dumps(st), encoding="utf-8")
        names.append(f.name)


for (model, k), (sl, tp) in GEO.items():
    targets = ([("struct", tp)] if tp else []) + [("rr2", 2 * sl), ("rr3", 3 * sl)]
    for tname, tpm in targets:
        grid = {"k": [float(k)], "band": ["def", "wide"],
                "target": ["struct" if tname == "struct" else "rr"],
                "cfg.atr_sl_mult": [sl], "cfg.atr_tp_mult": [tpm]}
        write(f"{model}_k{k}_{tname}", model, grid, "both")
# long-only versions of each model's default (k 2, default target)
for model, tname in (("w3", "struct"), ("w5", "struct"), ("abc", "rr2")):
    sl, tp = GEO[(model, 2)]
    tpm = tp if tname == "struct" else 2 * sl
    grid = {"k": [2.0], "band": ["def", "wide"], "target": ["struct" if tname == "struct" else "rr"],
            "side": ["long"], "cfg.atr_sl_mult": [sl], "cfg.atr_tp_mult": [tpm]}
    write(f"{model}_long", model, grid, "long")
(ROOT / "results" / "ew_studies.txt").write_bytes(("\n".join(names) + "\n").encode("utf-8"))  # LF only
print(len(names), "studies,", len(names) * 2 * 3, "configurations")
