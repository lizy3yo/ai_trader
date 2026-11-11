"""Near-miss diagnostics for lx_ crypto pairs (train+val only): long/short split
and random-entry p-values vs BOTH-side and LONG-only random baselines."""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from tools.factory import edge_p, _splits, _base
from tools.research import apply_params
from bot.backtest import run_backtest, load_econ

C = [("ETHUSD", "D1", {"cfg.atr_sl_mult":3.0,"cfg.atr_tp_mult":20.0,"entry_z":2.0,"exit_z":0.0,"lookback":60,"mode":"revert","pair":"crypto"}),
     ("ETHUSD", "D1", {"cfg.atr_sl_mult":3.0,"cfg.atr_tp_mult":20.0,"entry_z":2.5,"exit_z":0.0,"lookback":60,"mode":"revert","pair":"crypto"}),
     ("ETHUSD", "H4", {"cfg.atr_sl_mult":3.0,"cfg.atr_tp_mult":20.0,"entry_z":2.5,"exit_z":0.0,"lookback":240,"mode":"revert","pair":"crypto"}),
     ("BTCUSD", "D1", {"cfg.atr_sl_mult":3.0,"cfg.atr_tp_mult":20.0,"entry_z":1.0,"exit_z":0.0,"lookback":60,"mode":"follow","pair":"crypto"}),
     ("BTCUSD", "H4", {"cfg.atr_sl_mult":3.0,"cfg.atr_tp_mult":20.0,"entry_z":2.0,"exit_z":0.0,"lookback":240,"mode":"follow","pair":"crypto"})]
for mk, tf, p in C:
    _, _, tv = _splits(mk, tf, 60000)
    cfg = apply_params(_base(mk), "lx_pairs", tf, p)
    res = run_backtest(cfg, tv, load_econ(cfg.symbol), verbose=False)
    R = np.array([t["pnl"]/t["risk_amount"] for t in res.trade_log])
    d = np.array([t["direction"] for t in res.trade_log])
    out = {"mk": mk, "tf": tf, "n": len(R), "avgR": round(R.mean(),3),
           "long_n": int((d=="LONG").sum()), "long_avgR": round(R[d=="LONG"].mean(),3) if (d=="LONG").any() else None,
           "short_n": int((d=="SHORT").sum()), "short_avgR": round(R[d=="SHORT"].mean(),3) if (d=="SHORT").any() else None}
    for side in ("both", "long"):
        c, rm, r95, pr = edge_p(mk, tf, "lx_pairs", p, tv, side, 0, True, seeds=30)
        out[f"p_{side}"] = round(pr,3); out[f"rand_{side}"] = round(rm,3)
    print(json.dumps(out), p["lookback"], p["entry_z"], flush=True)
