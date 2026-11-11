"""Whole-history combination of the strategies with the best evidence (fixed 1% risk,
real costs). In-sample by construction (includes each test window) — for sizing only."""
import sys, copy, json
sys.path.insert(0, ".")
import numpy as np, pandas as pd
from bot.backtest import load_csv_bars, load_econ, run_backtest
from tools.factory import _base
from tools.research import apply_params

legs = {
 "US500 RSI2 dip (D1)": ("US500Cash", "lm_meanrev", "D1", {"mode":"rsi2","rsi_th":15.0,"exit_ma":3,"exit_rsi":70.0,"max_hold":10,"side":"long","trend":0,"cfg.atr_sl_mult":3.0,"cfg.atr_tp_mult":20.0,"cfg.cooldown_bars":0,"cfg.ema_trend":200}, None),
 "JP225 RSI pullback (D1)": ("JP225Cash", "lt_pullback", "D1", {"trig":"rsi","n":3,"thr":10,"side":"long","exit":"rec","exit_n":5,"cfg.ema_trend":200,"cfg.atr_sl_mult":3.0,"cfg.atr_tp_mult":20.0,"cfg.max_hold_bars":20,"cfg.cooldown_bars":0}, None),
 "GOLD trend_donchian (H1)": ("GOLD", "trend_donchian", "H1", {"lookback":360,"long_only":1,"cfg.ema_trend":400,"cfg.atr_sl_mult":3.0,"cfg.atr_tp_mult":5.0,"session.window":"09:00-23:00"}, "2013-06-01"),
}
monthly = {}
for name, (mk, strat, tf, p, start) in legs.items():
    cfg = apply_params(_base(mk), strat, tf, p)
    bars = load_csv_bars(f"data/{mk.lower()}_{tf.lower()}.csv")
    if start: bars = bars[bars.index >= start]
    tl = pd.DataFrame(run_backtest(cfg, bars, load_econ(mk), verbose=False).trade_log)
    tl["R"] = tl.pnl / tl.risk_amount
    m = tl.groupby(pd.to_datetime(tl.close_time).dt.to_period("M")).R.sum()
    monthly[name] = m
    yrs = (bars.index[-1] - bars.index[0]).days / 365.25
    print(f"{name:<26} {bars.index[0].date()}..{bars.index[-1].date()}  trades {len(tl)} ({len(tl)/yrs/12:.1f}/mo)  avgR {tl.R.mean():+.3f}  R/yr {tl.R.sum()/yrs:+.2f}")
df = pd.DataFrame(monthly).sort_index()
df = df[df.index >= pd.Period("2013-07", "M")].fillna(0.0)
print("\nmonthly R correlation (2013-07+):\n", df.corr().round(2).to_string())
def stats(cols, label):
    s = df[cols].sum(axis=1)
    eq = s.cumsum(); dd = (eq.cummax() - eq).max()
    avg = s.mean()
    cap1 = 500 / (0.01 * avg) if avg > 0 else float("inf")
    print(f"{label:<44} R/mo {avg:+.3f}  pos months {100*(s>0).mean():4.0f}%  worst month {s.min():+.1f}R  maxDD {dd:.1f}R  "
          f"| $500/mo needs: 1% risk ${cap1:,.0f}  2% ${cap1/2:,.0f}  (maxDD at 2% = {2*dd:.0f}%)")
print()
stats(["US500 RSI2 dip (D1)"], "US500 only")
stats(["US500 RSI2 dip (D1)", "JP225 RSI pullback (D1)"], "US500 + JP225")
stats(["US500 RSI2 dip (D1)", "GOLD trend_donchian (H1)"], "US500 + GOLD donchian")
stats(list(legs), "US500 + JP225 + GOLD donchian")
