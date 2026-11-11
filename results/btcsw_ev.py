"""btcsw scratch evaluator: train / val / trainval metrics per combo (never test)."""
import sys, json, itertools, argparse, os
sys.path.insert(0, r"C:\00-SideProject\AI_Trading"); os.chdir(r"C:\00-SideProject\AI_Trading")
import numpy as np, pandas as pd
from bot.backtest import run_backtest, load_econ
from bot.config import load_config
from tools.research import apply_params, load_split, metrics

ap = argparse.ArgumentParser()
ap.add_argument("--strategy", required=True); ap.add_argument("--tf", required=True)
ap.add_argument("--grid", required=True); ap.add_argument("--max-bars", type=int, required=True)
ap.add_argument("--log", default=None)
a = ap.parse_args()
base = load_config("research_btc.yaml")
econ = load_econ(base.symbol)
g = json.loads(open(a.grid).read()) if a.grid.endswith(".json") else json.loads(a.grid)
keys = list(g); combos = [dict(zip(keys, v)) for v in itertools.product(*(g[k] for k in keys))]
tr, va, _ = load_split(base.symbol, a.tf, a.max_bars)
tv = pd.concat([tr, va])
def hold(res, bars):
    if not res.trade_log: return 0
    tot = sum((t["close_time"] - t["open_time"]).total_seconds() for t in res.trade_log)
    return tot / (bars.index[-1] - bars.index[0]).total_seconds() * 100
rows = []
print(f"{len(combos)} combos {a.tf} train {tr.index[0].date()}..{tr.index[-1].date()} val ..{va.index[-1].date()}")
with open(r"C:/Users/User/AppData/Local/Temp/claude/fff079dc712489f1/a5035d8d-43a1-4c60-a2e1-55cdf9b15569/scratchpad/btcsw/budget.log", "a") as f:
    f.write(f"{a.strategy} {a.tf} {a.grid} {len(combos)}\n")
for p in combos:
    cfg = apply_params(base, a.strategy, a.tf, p)
    out = {}
    for nm, b in (("tr", tr), ("va", va), ("tv", tv)):
        r = run_backtest(cfg, b, econ, verbose=False)
        m = metrics(r); m["tim"] = round(hold(r, b), 0); out[nm] = m
    short = {k: v for k, v in p.items() if len(g[k]) > 1}
    T, V, TV = out["tr"], out["va"], out["tv"]
    print(f"{json.dumps(short, separators=(',',':')):<58} TR n{T['trades']:>3} PF{T['pf']:5.2f} R{T['avg_r']:+.2f} t{T['t_stat']:+.1f} pM{T['pos_months_pct']:>3.0f} dd{T['max_dd']:>6.0f} tim{T['tim']:>3.0f}% | VA n{V['trades']:>3} PF{V['pf']:5.2f} R{V['avg_r']:+.2f} t{V['t_stat']:+.1f} pM{V['pos_months_pct']:>3.0f} dd{V['max_dd']:>6.0f} tim{V['tim']:>3.0f}% | TV t{TV['t_stat']:+.2f}", flush=True)
    rows.append({"params": p, "tf": a.tf, **out})
if a.log:
    json.dump(rows, open(a.log, "w"), indent=1, default=str)
