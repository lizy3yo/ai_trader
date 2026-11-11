"""Goat Funded Trader 100K 2-step, simulated on the real trade history of the
3 confirmed strategies (whole history 2013-2026, real costs; in-sample, so
optimistic). Rules from the user's screenshot (2026-11-09):
  phase 1 +10%, phase 2 +5%, daily loss 3%, max loss 8% (assumed STATIC from the
  starting balance), no time limit; funded: 80% split, consistency 40%.
Each trade's result is in R; at risk r% per trade it moves the account r% x R
of the starting balance (no compounding inside a phase, as on a prop account).
Every month from 2013-07 is a possible start; a phase ends on the first day the
target is reached, or fails on the first breach (daily loss uses the sum of
trades closed that day; open-trade drawdown is not modelled).
"""
import sys
sys.path.insert(0, ".")
import numpy as np
import pandas as pd
from bot.backtest import load_csv_bars, load_econ, run_backtest
from tools.factory import _base
from tools.research import apply_params

legs = [
    ("US500Cash", "lm_meanrev", "D1", {"mode": "rsi2", "rsi_th": 15.0, "exit_ma": 3, "exit_rsi": 70.0, "max_hold": 10, "side": "long", "trend": 0, "cfg.atr_sl_mult": 3.0, "cfg.atr_tp_mult": 20.0, "cfg.cooldown_bars": 0, "cfg.ema_trend": 200}, None),
    ("JP225Cash", "lt_pullback", "D1", {"trig": "rsi", "n": 3, "thr": 10, "side": "long", "exit": "rec", "exit_n": 5, "cfg.ema_trend": 200, "cfg.atr_sl_mult": 3.0, "cfg.atr_tp_mult": 20.0, "cfg.max_hold_bars": 20, "cfg.cooldown_bars": 0}, None),
    ("GOLD", "trend_donchian", "H1", {"lookback": 360, "long_only": 1, "cfg.ema_trend": 400, "cfg.atr_sl_mult": 3.0, "cfg.atr_tp_mult": 5.0, "session.window": "09:00-23:00"}, "2013-06-01"),
]
trades = []
for mk, strat, tf, p, start in legs:
    cfg = apply_params(_base(mk), strat, tf, p)
    bars = load_csv_bars(f"data/{mk.lower()}_{tf.lower()}.csv")
    if start:
        bars = bars[bars.index >= start]
    tl = pd.DataFrame(run_backtest(cfg, bars, load_econ(mk), verbose=False).trade_log)
    tl["R"] = tl.pnl / tl.risk_amount
    tl["day"] = pd.to_datetime(tl.close_time).dt.normalize()
    trades.append(tl[["day", "R"]])
daily = pd.concat(trades).groupby("day").R.sum().sort_index()
daily = daily[daily.index >= "2013-07-01"]
days, rs = daily.index.to_numpy(), daily.to_numpy()
end = days[-1]


def phase(start_i, risk, target, daily_lim=3.0, max_loss=8.0):
    bal = 0.0
    for i in range(start_i, len(rs)):
        d = rs[i] * risk
        if d <= -daily_lim:
            return "fail", i
        bal += d
        if bal <= -max_loss:
            return "fail", i
        if bal >= target:
            return "pass", i
    return "open", len(rs) - 1


starts = pd.date_range("2013-07-01", pd.Timestamp(end) - pd.DateOffset(months=6), freq="MS")
print(f"{'risk':>5} | {'PASS both':>9} {'fail':>5} {'still going':>11} | {'median months to get funded':>27} | funded payout/month (80%)")
avg_R_month = daily.sum() / ((pd.Timestamp(end) - pd.Timestamp(days[0])).days / 30.44)
for risk in (0.5, 0.75, 1.0, 1.25, 1.5, 2.0):
    res, months = [], []
    for s in starts:
        i0 = int(np.searchsorted(days, np.datetime64(s)))
        r1, j = phase(i0, risk, 10.0)
        if r1 != "pass":
            res.append(r1); continue
        r2, k = phase(j + 1, risk, 5.0)
        res.append(r2 if r2 != "pass" else "pass")
        if r2 == "pass":
            months.append((pd.Timestamp(days[k]) - s).days / 30.44)
    res = pd.Series(res)
    pay = avg_R_month * risk / 100 * 100_000 * 0.8
    med = f"{np.median(months):.0f}" if months else "-"
    print(f"{risk:>4}% | {100*(res=='pass').mean():>8.0f}% {100*(res=='fail').mean():>4.0f}% {100*(res=='open').mean():>10.0f}% | "
          f"{med:>27} | ~${pay:,.0f}")
print(f"\n(average {avg_R_month:.2f}R per month across the 3 strategies; worst single day {rs.min():+.1f}R)")
