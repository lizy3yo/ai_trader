"""Diagnostic 2 (TRAIN only): first-touch TP/SL outcome simulation for the
two live hypotheses — pday FOLLOW and roll FADE — across geometry pairs and
entry windows. Approximates the harness fill model (entry next open, spread
on entry, SL first on both-touch bars). Steers round-3 grids only."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bot.config import load_config
from bot.strategies import get_strategy

m = get_strategy("lq_sweep")
cfg = load_config()
scfg = cfg.strategy
scfg.timeframe = "M5"
SPREAD = 0.55
H = 96  # max bars held in the sim

df = pd.read_csv("data/gold_m5.csv", parse_dates=["time"]).set_index("time")
df = df[["open", "high", "low", "close"]].astype(float).iloc[-99999:]
train = df.iloc[: int(len(df) * 0.6)]
o = train["open"].to_numpy(); h = train["high"].to_numpy()
l = train["low"].to_numpy(); c = train["close"].to_numpy()


def sim(sig_idx, direction, atr_v, sl_m, tp_m):
    """direction: +1 long / -1 short per signal. Returns mean $PnL/trade, win%."""
    pnl = []
    for i, d in zip(sig_idx, direction):
        if i + 2 >= len(o):
            continue
        a = atr_v[i]
        entry = o[i + 1] + (SPREAD if d > 0 else 0.0)
        sl = entry - d * sl_m * a
        tp = entry + d * tp_m * a
        out = None
        for j in range(i + 1, min(i + 1 + H, len(o))):
            hj, lj = h[j], l[j]
            if d > 0:
                hit_sl = lj <= sl; hit_tp = hj >= tp
            else:
                hit_sl = hj + SPREAD >= sl; hit_tp = lj + SPREAD <= tp
            if hit_sl:
                out = (sl - entry) * d; break
            if hit_tp:
                out = (tp - entry) * d; break
        if out is None:
            j = min(i + H, len(c) - 1)
            out = (c[j] + (SPREAD if d < 0 else 0.0) - entry) * d
        pnl.append(out)
    pnl = np.array(pnl)
    return pnl.mean(), (pnl > 0).mean() * 100, len(pnl)


def signals(params):
    scfg.params = params
    ind = m.add_indicators(train, scfg)
    sh = ((ind["swept_hi"] > 0) & (ind["close"] < ind["lvl_hi"])
          & ((ind["out_hi"].shift(1) > 0) | (ind["beyond_hi"] > 0))
          & (ind["consec_out_hi"].shift(1) <= params.get("max_out_bars", 2)))
    lo = ((ind["swept_lo"] > 0) & (ind["close"] > ind["lvl_lo"])
          & ((ind["out_lo"].shift(1) > 0) | (ind["beyond_lo"] > 0))
          & (ind["consec_out_lo"].shift(1) <= params.get("max_out_bars", 2)))
    return ind, sh.to_numpy(), lo.to_numpy()


hrs = train.index.hour.to_numpy()
print("=== pday FOLLOW (long after high-sweep reclaim, short after low-sweep reclaim) ===")
ind, sh, lo = signals({"level_mode": "pday", "max_out_bars": 2})
atr_v = ind["atr"].to_numpy()
for w0, w1 in ((9, 13), (14, 20), (9, 19)):
    win = (hrs >= w0) & (hrs < w1)
    si = np.where((sh | lo) & win)[0]
    d = np.where(sh[si], 1, -1)          # follow: high sweep -> LONG
    for sl_m, tp_m in ((1.5, 0.75), (1.5, 1.25), (2.0, 1.0), (1.0, 1.5)):
        mu, wr, n = sim(si, d, atr_v, sl_m, tp_m)
        print(f"  win {w0:02d}-{w1:02d} SL{sl_m}/TP{tp_m}: ${mu:+.3f}/trade, win {wr:.1f}%, n={n}")

print("\n=== roll FADE n=24 ===")
ind, sh, lo = signals({"level_mode": "roll", "roll_n": 24, "max_out_bars": 3})
atr_v = ind["atr"].to_numpy()
win = (hrs >= 9) & (hrs < 19)
for sides in ("long", "both"):
    if sides == "long":
        si = np.where(lo & ~sh & win)[0]; d = np.ones(len(si), int)
    else:
        si = np.where((sh | lo) & win)[0]; d = np.where(sh[si], -1, 1)
    for sl_m, tp_m in ((1.5, 1.5), (3.0, 5.0), (2.0, 1.0), (2.5, 3.75)):
        mu, wr, n = sim(si, d, atr_v, sl_m, tp_m)
        print(f"  sides={sides:5s} SL{sl_m}/TP{tp_m}: ${mu:+.3f}/trade, win {wr:.1f}%, n={n}")
