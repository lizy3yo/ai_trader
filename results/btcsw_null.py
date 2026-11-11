"""Random-entry null with IDENTICAL gates, stops and exit_signal (btcsw modules).
p = share of seeds with avgR >= candidate. Split trainval only."""
import sys, json, argparse, os
sys.path.insert(0, r"C:\00-SideProject\AI_Trading"); os.chdir(r"C:\00-SideProject\AI_Trading")
import numpy as np, pandas as pd
from bot.backtest import run_backtest, load_econ
from bot.config import load_config
from tools.research import apply_params, load_split, r_stats
ap = argparse.ArgumentParser()
ap.add_argument("--strategy", required=True); ap.add_argument("--tf", required=True)
ap.add_argument("--params", required=True); ap.add_argument("--max-bars", type=int, required=True)
ap.add_argument("--seeds", type=int, default=30); ap.add_argument("--split", default="trainval")
a = ap.parse_args()
base = load_config("research_btc.yaml"); econ = load_econ(base.symbol)
tr, va, _ = load_split(base.symbol, a.tf, a.max_bars)
bars = {"train": tr, "val": va, "trainval": pd.concat([tr, va])}[a.split]
P = json.loads(a.params)
res = run_backtest(apply_params(base, a.strategy, a.tf, P), bars, econ, verbose=False)
cr, ct = r_stats(res); n = res.trades
pnl = sum(t["pnl"] for t in res.trade_log)
print(f"cand [{a.split}] n {n} avgR {cr:+.3f} t {ct:+.2f} pnl {pnl:+.0f}")
bpd = {"H1": 24, "H4": 6}[a.tf]
daily = int(P.get("daily", 1))
elig = len(bars) / (bpd if daily else 1)
prob = min(0.9, n / elig * 3)
def run(seed, prob):
    Q = dict(P, null_prob=prob, null_seed=seed)
    return run_backtest(apply_params(base, a.strategy, a.tf, Q), bars, econ, verbose=False)
for _ in range(6):
    m = run(0, prob).trades
    if m == 0: prob = min(0.95, prob * 3); continue
    prob = min(0.95, prob * n / m)
rs, ns, pn = [], [], []
for s in range(1, a.seeds + 1):
    r = run(s, prob); rr, _ = r_stats(r); rs.append(rr); ns.append(r.trades); pn.append(sum(t["pnl"] for t in r.trade_log))
rs = np.array(rs); pn = np.array(pn)
print(f"null x{a.seeds} prob {prob:.3f} trades~{np.mean(ns):.0f}: mean avgR {rs.mean():+.3f} p95 {np.percentile(rs,95):+.3f} => p_avgR {(rs>=cr).mean():.3f} | pnl mean {pn.mean():+.0f} p_pnl {(pn>=pnl).mean():.3f}")
