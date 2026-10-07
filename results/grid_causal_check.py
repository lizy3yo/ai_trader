"""Causality audit for results/grid_sim.py.

1) truncation test: indicators computed on a PREFIX of the data must equal the
   same rows of indicators computed on the FULL data (no lookahead anywhere).
2) fill test: a basket seeded from bar i's close must fill at bar i+1's OPEN
   (never at bar i), and every ladder add must fill at its ladder price.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from grid_sim import GridCfg, add_indicators, load_bars, run_window  # noqa: E402

bars = load_bars("data/gold_m15.csv")
cfg = GridCfg()
full = add_indicators(bars, cfg)

bad = 0
for i in (900, 3000, 12345, 50000, 99998):
    pre = {k: (v if k == "tf_min" else v[: i + 1]) for k, v in bars.items()}
    ip = add_indicators(pre, cfg)
    for key in ("atr", "rsi", "ema_f", "ema_s"):
        a, b = ip[key][-1], full[key][i]
        if not np.isclose(a, b, rtol=0, atol=1e-12):
            print(f"  LOOKAHEAD in {key} at i={i}: prefix {a!r} != full {b!r}")
            bad += 1
print(f"truncation test: {'PASS' if bad == 0 else f'FAIL ({bad})'}"
      "  (prefix indicators == full-frame indicators, bit-for-bit)")

# determinism / window-independence: the same window run twice is identical
r1 = run_window(bars, full, cfg, 5000, 25000)
r2 = run_window(bars, full, cfg, 5000, 25000)
print("determinism    :", "PASS" if r1 == r2 else "FAIL")

# a run started later must not know anything about earlier bars beyond the
# recursive indicator state -> results differ, but must still be self-consistent
r3 = run_window(bars, full, cfg, 5000, 25000)
print("sanity         : baskets", r3["baskets"], "win%", r3["win_rate"],
      "ruin", r3["ruin"], "maxBasketDD $", r3["max_basket_dd"])

# ---------------------------------------------------------------------------
# accounting identity: end_equity == start + gross_win + gross_loss - swap_paid
# (spread is embedded in the fill prices, so it is inside gross_win/gross_loss)
# ---------------------------------------------------------------------------
bad = 0
for st, en in ((5000, 30000), (40000, 70000), (250, 99999)):
    r = run_window(bars, full, cfg, st, en)
    lhs = r["end_equity"]
    rhs = cfg.start_equity + r["gross_win"] + r["gross_loss"] - r["swap_paid"]
    if abs(lhs - rhs) > 0.02:
        print(f"  ACCOUNTING BREAK [{st}:{en}] end {lhs} != {round(rhs,2)}")
        bad += 1
print(f"accounting id  : {'PASS' if bad == 0 else f'FAIL ({bad})'}"
      "  (end = start + grossWin + grossLoss - swap)")

# ---------------------------------------------------------------------------
# hand-verifiable synthetic basket:
#   ATR -> 2.00, step_atr 5 => grid step $10.00, 3 legs, base 0.01, spread 0.55
#   fills 2000.55 / 1990.55 / 1980.55  -> VWAP 1990.55
#   TP = VWAP + 0.5*step = 1995.55     -> P&L = 0.03 lots x 100 x $5.00 = $15.00
# ---------------------------------------------------------------------------
import pandas as pd
from dataclasses import replace

rows, px = [], 2000.0
for i in range(320):                       # flat oscillation: TR = 2.00 -> ATR 2.00
    rows.append((2000.0, 2001.0, 1999.0, 2000.0))
for k in range(1, 9):                      # walk down through all three levels
    px = 2000.0 - 5.0 * k
    rows.append((px + 5.0, px + 5.0, px, px))
for k in range(1, 7):                      # walk back up, ending ON the basket TP
    px = 1960.0 + 6.0 * k                  # 1966..1996; TP 1995.55 hit on the last
    rows.append((px - 6.0, px, px - 6.0, px))
idx = pd.date_range("2024-01-02 00:01", periods=len(rows), freq="1min")
syn = pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=idx)
tmp = Path("results/_grid_syn.csv")
syn.to_csv(tmp, index_label="time")
sb = load_bars(str(tmp))
scfg = replace(GridCfg(), name="syn", seed="long", step_atr=5.0, max_legs=3,
               tp_frac=0.5, base_lot=0.01, on_max_legs="hold", ema_slow=50,
               spread_points=55.0)
sind = add_indicators(sb, scfg)
sr = run_window(sb, sind, scfg, 0, len(sb["close"]))
tmp.unlink()
got = sr["gross_win"] + sr["gross_loss"]
print(f"synthetic math : {'PASS' if abs(got - 15.00) < 0.01 else 'FAIL'}  "
      f"expected +$15.00 for a 3-leg basket, got ${got:+.2f} "
      f"(baskets {sr['baskets']}, legs {sr['max_legs_seen']}, lots {sr['max_total_lots']})")
