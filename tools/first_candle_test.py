"""FIRST CANDLE RULE test (2026-11-10) — Casper SMC / Jesse Rogers YouTube
strategy ("The First Candle Rule Is The Easiest Way To Become Profitable FAST",
and 3 related videos the user sent). Rules taken from his published
TradingView script "Casper SMC: 5m ORB + Retest" and the video summaries.

PRE-REGISTERED (before any result was seen):
  candle   the first candle of the New York cash session, 09:30 NY time
           (server clock converted Europe/Athens -> America/New_York, so the
           EU/US daylight-saving gap weeks are handled)
  RETEST   (his script) 5m range 09:30-09:35. A 5m candle CLOSES beyond the
           range -> breakout. Then a later candle wicks back to the broken
           level while its body stays outside -> enter at the next open.
           Stop = range midpoint. Target = 2R (and 3R). Cancel the day if a
           candle closes back through the opposite side. One trade a day,
           entries until 11:00 NY, flat at 16:00 NY.
  BREAK    simpler version: enter on the first 5m close beyond the 5m range;
           stop = opposite side of the range; target 2R / 3R.
  RETEST30 as RETEST but the range is the first 30 minutes (09:30-10:00),
           entries until 12:00 NY.
  costs    bars are BID prices; longs buy at ask (bid + recorded spread),
           shorts exit at ask; recorded spread x 1.3 (recorded bar spreads
           understate live ones). If stop and target are both touched in one
           5m bar, the STOP is assumed first.
  null     the same trades with a coin-flip direction (same entry time, same
           stop/target distance): if the rule's direction calls are no better
           than a coin, there is no edge.
Data: XM M5 bars (MT5 keeps the last 100,000 = ~17 months for the indices).

Usage: python tools/first_candle_test.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MARKETS = {"US100Cash": "data/us100cash_m5.csv", "US500Cash": "data/us500cash_m5.csv",
           "GOLD": "data/gold_m5.csv"}
POINT = 0.01
SPREAD_MULT = 1.3
VARIANTS = {   # name: (range minutes, mode, entry cutoff NY, stop)
    "RETEST": (5, "retest", "11:00", "mid"),
    "BREAK": (5, "break", "11:00", "far"),
    "RETEST30": (30, "retest", "12:00", "mid"),
}


def load(path: str) -> pd.DataFrame:
    d = pd.read_csv(path, parse_dates=["time"])
    ny = d.time.dt.tz_localize("Europe/Athens", ambiguous="NaT", nonexistent="NaT").dt.tz_convert("America/New_York")
    d["ny"] = ny.dt.tz_localize(None)
    d = d.dropna(subset=["ny"]).reset_index(drop=True)
    d["sp"] = d.spread * POINT * SPREAD_MULT
    d["day"] = d.ny.dt.date
    d["t"] = d.ny.dt.strftime("%H:%M")
    return d


def trades_for(d: pd.DataFrame, rng_min: int, mode: str, cutoff: str, stop: str, rr: float,
               flip_seed: int | None = None) -> list[dict]:
    out = []
    end_rng = (pd.Timestamp("09:30") + pd.Timedelta(minutes=rng_min)).strftime("%H:%M")
    rs = np.random.default_rng(flip_seed) if flip_seed is not None else None
    for day, g in d.groupby("day"):
        g = g[(g.t >= "09:30") & (g.t < "16:00")]
        r = g[g.t < end_rng]
        if len(r) != rng_min // 5 or r.t.iloc[0] != "09:30":
            continue                                    # incomplete opening range
        hi, lo = r.high.max(), r.low.min()
        mid = (hi + lo) / 2
        rest = g[g.t >= end_rng]
        o, h, l, c, sp, tt = (rest[k].to_numpy() for k in ("open", "high", "low", "close", "sp", "t"))
        side, entry_i = 0, None
        for i in range(len(rest) - 1):
            if tt[i] >= cutoff:
                break
            if side == 0:
                if c[i] > hi:
                    side = 1
                elif c[i] < lo:
                    side = -1
                else:
                    continue
                if mode == "break":
                    entry_i = i + 1
                    break
                continue
            # waiting for the retest
            if (side == 1 and c[i] < lo) or (side == -1 and c[i] > hi):
                break                                   # full reversal: cancel the day
            if side == 1 and l[i] <= hi and min(o[i], c[i]) > hi:
                entry_i = i + 1
                break
            if side == -1 and h[i] >= lo and max(o[i], c[i]) < lo:
                entry_i = i + 1
                break
        if entry_i is None or entry_i >= len(rest):
            continue
        if rs is not None:
            side = int(rs.choice([-1, 1]))
        e_bid = o[entry_i]
        entry = e_bid + sp[entry_i] if side == 1 else e_bid
        if stop == "mid":
            dist = abs(entry - mid)
        else:
            dist = abs(entry - (lo if side == 1 else hi))
        if rs is not None:                              # coin flip keeps the rule's stop distance
            dist = abs(entry - mid) if stop == "mid" else abs(entry - (lo if side == -1 else hi))
        if dist <= 0:
            continue
        sl, tp = entry - side * dist, entry + side * rr * dist
        res = None
        for j in range(entry_i, len(rest)):
            # long exits at bid; short exits at ask (bid + spread)
            lo_x, hi_x = (l[j], h[j]) if side == 1 else (l[j] + sp[j], h[j] + sp[j])
            hit_sl = lo_x <= sl if side == 1 else hi_x >= sl
            hit_tp = hi_x >= tp if side == 1 else lo_x <= tp
            if hit_sl:
                px = min(sl, o[j]) if side == 1 else max(sl, o[j] + sp[j])   # gap through the stop
                res = (px - entry) * side / dist
                break
            if hit_tp:
                res = rr
                break
        if res is None:                                 # flat at the 16:00 NY close
            px = c[-1] if side == 1 else c[-1] + sp[-1]
            res = (px - entry) * side / dist
        out.append({"day": day, "side": side, "R": res, "risk_pts": dist,
                    "cost_R": sp[entry_i] / dist})
    return out


def stats(t: pd.DataFrame) -> str:
    if t.empty:
        return "no trades"
    r = t.R
    tstat = r.mean() / (r.std(ddof=1) / np.sqrt(len(r))) if len(r) > 1 else 0
    return (f"{len(r):>4} trades, win {100*(r>0).mean():3.0f}%, avg {r.mean():+.3f}R, t {tstat:+.1f}, "
            f"total {r.sum():+.1f}R, spread cost {t.cost_R.mean():.2f}R/trade")


def main() -> None:
    lines = ["FIRST CANDLE RULE test (XM M5, real spreads x1.3, stop-first on ambiguous bars)\n"]
    for mk, path in MARKETS.items():
        d = load(path)
        lines.append(f"== {mk}: {d.day.nunique()} days, {d.ny.iloc[0].date()} -> {d.ny.iloc[-1].date()}")
        for name, (rng_min, mode, cutoff, stop) in VARIANTS.items():
            for rr in (2.0, 3.0):
                t = pd.DataFrame(trades_for(d, rng_min, mode, cutoff, stop, rr))
                flips = [pd.DataFrame(trades_for(d, rng_min, mode, cutoff, stop, rr, s)).R.mean()
                         for s in range(1, 21)]
                beat = int((np.array(flips) < (t.R.mean() if len(t) else -9)).sum())
                half = ""
                if len(t) > 20:
                    k = len(t) // 2
                    half = f" | 1st half {t.R.iloc[:k].mean():+.3f}R, 2nd half {t.R.iloc[k:].mean():+.3f}R"
                lines.append(f"  {name:<8} 1:{rr:.0f}  {stats(t)}  | coin-flip avg {np.mean(flips):+.3f}R, "
                             f"rule beats {beat}/20{half}")
        lines.append("")
    txt = "\n".join(lines)
    print(txt)
    from pathlib import Path
    Path("results/first_candle.txt").write_text(txt, encoding="utf-8")


if __name__ == "__main__":
    main()
