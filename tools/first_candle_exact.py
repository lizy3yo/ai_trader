"""FIRST CANDLE RULE — exact rules from the 4 videos the user sent (transcribed
locally into videos/*.txt, 2026-11-11). Replaces the guessed rules in
tools/first_candle_test.py.

PRE-REGISTERED (written from the transcripts before any result was seen):
  FVG15   ("Easiest Way To Become Profitable FAST", Casper / Jesse Rogers)
          range = 15-min candle 09:30-09:45 NY. On M5, a fair value gap
          (3 candles, candle-3 low above candle-1 high for longs) with one of
          the 3 candles CLOSING outside the range. Limit order at the gap
          (filled when price dips back into it, at the gap's near edge).
          Stop = candle-1 low (high for shorts). Target fixed 2R. Trade only
          the first 90 minutes (setups until 11:00 NY), flat 16:00 NY.
  FVG5M1  ("Easiest Way To Make F*ck You Money", Casper)
          range = first 5-min candle 09:30-09:35 NY. On M1, the same FVG
          pattern through the level; enter at the CLOSE of the candle that
          confirms the gap (candle 3). Stop = candle-1 low/high. Target 2R.
  ORB30   ("The First Candle Scalping Strategy Nobody Talks About")
          range = 30-min candle 09:30-10:00 NY. On M5, a close outside the
          range, then a pullback that touches the broken level and closes
          back outside -> enter at that close. Stop = a 5-min CLOSE back
          inside the range (exit at that close); risk unit R = entry to the
          level (floored at 10% of the range). Management as described:
          take half at 1R and move the stop to break-even, the rest runs to
          a close back inside / 16:00 NY.  ORB30_2R: same entry, fixed 2R.
  RETEST5 ("This First Candle Strategy Is Boring", Casper)
          range = first 5-min candle; M5 close outside, then a retest candle
          that trades into the level but closes outside -> enter at its close.
          A close back inside resets. Stop = range midpoint. Target 2R.
  one trade per market per day, first valid setup only.
  costs   bid bars; longs pay the recorded spread x 1.3 at entry, shorts at
          exit; stop assumed first when stop and target are in the same bar.
  null    the same trades with a coin-flip direction and the same stop
          distance / target (20 seeds): direction calls must beat a coin.
Markets (user's list): US100, US500, GOLD, BTCUSD, EURUSD GBPUSD USDJPY USDCHF
USDCAD AUDUSD NZDUSD. Same 09:30 NY time in every market (as the video says).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

MARKETS = ["US100Cash", "US500Cash", "GOLD", "BTCUSD", "EURUSD", "GBPUSD", "USDJPY",
           "USDCHF", "USDCAD", "AUDUSD", "NZDUSD"]
SPREAD_MULT = 1.3
OUT = Path("results/first_candle_exact")


def point_of(sym: str) -> float:
    p = Path(f"data/{sym.lower()}_econ.json")
    return json.loads(p.read_text(encoding="utf-8"))["point"]


def load(sym: str, tf: str) -> pd.DataFrame | None:
    p = Path(f"data/{sym.lower()}_{tf}.csv")
    if not p.exists():
        return None
    d = pd.read_csv(p, parse_dates=["time"])
    ny = d.time.dt.tz_localize("Europe/Athens", ambiguous="NaT", nonexistent="NaT").dt.tz_convert("America/New_York")
    d["ny"] = ny.dt.tz_localize(None)
    d = d.dropna(subset=["ny"]).reset_index(drop=True)
    d["sp"] = d.spread * point_of(sym) * SPREAD_MULT
    d["day"] = d.ny.dt.date
    d["t"] = d.ny.dt.strftime("%H:%M")
    return d


# ---------------------------------------------------------------- simulation
def run_trade(o, h, l, c, sp, start, side, entry, stop, target, close_exit=None, half_at=None, unit=None):
    """Walk bars from `start`. Returns R (risk unit = |entry - stop| unless
    close_exit gives its own). close_exit(i) -> True when a bar CLOSE triggers
    the exit at that close. half_at = R level to bank half and move to BE."""
    dist = unit if unit is not None else abs(entry - stop)
    if dist <= 0:
        return None
    banked, size, sl = 0.0, 1.0, stop
    for j in range(start, len(o)):
        lo_x, hi_x = (l[j], h[j]) if side == 1 else (l[j] + sp[j], h[j] + sp[j])
        hit_sl = (lo_x <= sl) if side == 1 else (hi_x >= sl)
        if hit_sl:
            px = min(sl, o[j]) if side == 1 else max(sl, o[j] + sp[j])
            return banked + size * (px - entry) * side / dist
        if half_at is not None and size == 1.0:
            lvl = entry + side * half_at * dist
            if (hi_x >= lvl) if side == 1 else (lo_x <= lvl):
                banked, size, sl = 0.5 * half_at, 0.5, entry      # half off, stop to break-even
        if target is not None:
            if (hi_x >= target) if side == 1 else (lo_x <= target):
                return banked + size * (target - entry) * side / dist
        if close_exit is not None and close_exit(j):
            px = c[j] if side == 1 else c[j] + sp[j]
            return banked + size * (px - entry) * side / dist
    px = c[-1] if side == 1 else c[-1] + sp[-1]
    return banked + size * (px - entry) * side / dist


def day_bars(g, rng_min):
    g = g[(g.t >= "09:30") & (g.t < "16:00")]
    end = (pd.Timestamp("09:30") + pd.Timedelta(minutes=rng_min)).strftime("%H:%M")
    r, rest = g[g.t < end], g[g.t >= end]
    return r, rest


def fvg_rule(d, rng_min, step_min, entry_mode, rr=2.0, cutoff="11:00", flip=None):
    out = []
    rs = np.random.default_rng(flip) if flip is not None else None
    for day, g in d.groupby("day"):
        r, rest = day_bars(g, rng_min)
        if len(r) != rng_min // step_min or r.t.iloc[0] != "09:30" or len(rest) < 5:
            continue
        hi, lo = r.high.max(), r.low.min()
        o, h, l, c, sp, tt = (rest[k].to_numpy() for k in ("open", "high", "low", "close", "sp", "t"))
        trade = None
        for i in range(2, len(rest) - 1):
            if tt[i] >= cutoff:
                break
            closes_out_up = max(c[i - 2], c[i - 1], c[i]) > hi
            closes_out_dn = min(c[i - 2], c[i - 1], c[i]) < lo
            if l[i] > h[i - 2] and closes_out_up:          # bullish FVG through the high
                side, near, stop = 1, l[i], l[i - 2]
            elif h[i] < l[i - 2] and closes_out_dn:        # bearish FVG through the low
                side, near, stop = -1, h[i], h[i - 2]
            else:
                continue
            if entry_mode == "close":                       # enter at candle-3 close (next open)
                e_i = i + 1
                entry = o[e_i] + (sp[e_i] if side == 1 else 0.0)
            else:                                           # limit at the gap's near edge
                e_i = None
                for j in range(i + 1, len(rest)):
                    if tt[j] >= "16:00":
                        break
                    ask_lo = l[j] + sp[j]
                    if (side == 1 and ask_lo <= near) or (side == -1 and h[j] >= near):
                        e_i = j
                        break
                if e_i is None:
                    break                                   # never filled: no trade today
                entry = min(near, o[e_i] + sp[e_i]) if side == 1 else max(near, o[e_i])
            trade = (e_i, side, entry, stop)
            break
        if trade is None:
            continue
        e_i, side, entry, stop = trade
        dist = abs(entry - stop)
        if rs is not None:
            side = int(rs.choice([-1, 1]))
            stop = entry - side * dist
        if dist <= 0 or (side == 1 and stop >= entry) or (side == -1 and stop <= entry):
            continue
        tgt = entry + side * rr * dist
        # a limit fill bar may also hit the stop: start checking on the fill bar itself
        R = run_trade(o, h, l, c, sp, e_i, side, entry, stop, tgt)
        if R is not None:
            out.append({"day": day, "side": side, "R": R, "cost_R": sp[e_i] / dist})
    return out


def orb30_rule(d, mode, flip=None, cutoff="11:30"):
    out = []
    rs = np.random.default_rng(flip) if flip is not None else None
    for day, g in d.groupby("day"):
        r, rest = day_bars(g, 30)
        if len(r) != 6 or r.t.iloc[0] != "09:30" or len(rest) < 5:
            continue
        hi, lo = r.high.max(), r.low.min()
        o, h, l, c, sp, tt = (rest[k].to_numpy() for k in ("open", "high", "low", "close", "sp", "t"))
        side, e_i = 0, None
        for i in range(len(rest) - 1):
            if tt[i] >= cutoff:
                break
            if side == 0:
                side = 1 if c[i] > hi else (-1 if c[i] < lo else 0)
                continue
            if (side == 1 and c[i] < hi) or (side == -1 and c[i] > lo):
                break                                        # closed back inside before a retest
            if side == 1 and l[i] <= hi and c[i] > hi:
                e_i = i + 1
                break
            if side == -1 and h[i] >= lo and c[i] < lo:
                e_i = i + 1
                break
        if e_i is None:
            continue
        lvl = hi if side == 1 else lo
        entry = o[e_i] + (sp[e_i] if side == 1 else 0.0)
        floor = 0.10 * (hi - lo)
        dist = max(abs(entry - lvl), floor)
        if rs is not None:
            side = int(rs.choice([-1, 1]))
        stop_px = entry - side * dist * 3        # disaster stop far away; the close rule exits first
        if rs is None:
            inside = (lambda j, s=side: c[j] < hi if s == 1 else c[j] > lo)
        else:                                    # coin flip: exit on a close 1R against
            inside = (lambda j, s=side, e=entry, dd=dist: (c[j] - e) * s <= -dd)
        if mode == "managed":
            R = run_trade(o, h, l, c, sp, e_i, side, entry, stop_px, None, close_exit=inside, half_at=1.0, unit=dist)
        else:
            R = run_trade(o, h, l, c, sp, e_i, side, entry, stop_px, entry + side * 2 * dist, close_exit=inside, unit=dist)
        if R is not None:
            out.append({"day": day, "side": side, "R": R, "cost_R": sp[e_i] / dist})
    return out


def retest5_rule(d, flip=None, cutoff="11:00", rr=2.0):
    """'This First Candle Strategy Is Boring' (Casper): first 5-min candle
    09:30-09:35; a 5m close outside, then a retest candle that trades into
    the level but CLOSES outside -> enter at that close. A close back inside
    resets (wait for a new break, either side). Stop = range midpoint, 2R."""
    out = []
    rs = np.random.default_rng(flip) if flip is not None else None
    for day, g in d.groupby("day"):
        r, rest = day_bars(g, 5)
        if len(r) != 1 or r.t.iloc[0] != "09:30" or len(rest) < 5:
            continue
        hi, lo = r.high.max(), r.low.min()
        mid = (hi + lo) / 2
        o, h, l, c, sp, tt = (rest[k].to_numpy() for k in ("open", "high", "low", "close", "sp", "t"))
        side, e_i, broke_at = 0, None, -1
        for i in range(len(rest) - 1):
            if tt[i] >= cutoff:
                break
            if lo <= c[i] <= hi:
                side = 0                                     # back inside: reset
                continue
            if side == 0:
                side, broke_at = (1 if c[i] > hi else -1), i
                continue
            if i > broke_at and ((side == 1 and l[i] <= hi and c[i] > hi) or (side == -1 and h[i] >= lo and c[i] < lo)):
                e_i = i + 1
                break
            if (side == 1 and c[i] < lo) or (side == -1 and c[i] > hi):
                side, broke_at = -side, i                    # closed out the other side: new break
        if e_i is None:
            continue
        entry = o[e_i] + (sp[e_i] if side == 1 else 0.0)
        dist = abs(entry - mid)
        if rs is not None:
            side = int(rs.choice([-1, 1]))
        stop = entry - side * dist
        if dist <= 0:
            continue
        R = run_trade(o, h, l, c, sp, e_i, side, entry, stop, entry + side * rr * dist)
        if R is not None:
            out.append({"day": day, "side": side, "R": R, "cost_R": sp[e_i] / dist})
    return out


RULES = {
    "RETEST5 (5m candle, close+retest, stop mid, 2R)": ("m5", lambda d, f=None: retest5_rule(d, flip=f)),
    "FVG15 (15m candle, M5 gap, limit, 2R)": ("m5", lambda d, f=None: fvg_rule(d, 15, 5, "limit", flip=f)),
    "FVG5M1 (5m candle, M1 gap, at close, 2R)": ("m1", lambda d, f=None: fvg_rule(d, 5, 1, "close", flip=f)),
    "ORB30 managed (half at 1R, BE, runner)": ("m5", lambda d, f=None: orb30_rule(d, "managed", flip=f)),
    "ORB30 fixed 2R": ("m5", lambda d, f=None: orb30_rule(d, "2r", flip=f)),
}


def summary(t: pd.DataFrame) -> dict:
    if t.empty:
        return {"n": 0}
    r = t.R
    return {"n": len(r), "win": 100 * (r > 0).mean(), "avgR": r.mean(),
            "t": r.mean() / (r.std(ddof=1) / np.sqrt(len(r))) if len(r) > 1 else 0.0,
            "totR": r.sum(), "cost": t.cost_R.mean()}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    lines = ["FIRST CANDLE RULE - exact video rules, XM data, spread x1.3, stop-first\n"]
    allrows = []
    data = {(m, tf): load(m, tf) for m in MARKETS for tf in ("m5", "m1")}
    for name, (tf, fn) in RULES.items():
        lines.append(f"== {name}  [{tf.upper()} bars]")
        pooled, pooled_flip = [], []
        for m in MARKETS:
            d = data[(m, tf)]
            if d is None:
                continue
            t = pd.DataFrame(fn(d))
            if t.empty:
                lines.append(f"   {m:<10} no trades")
                continue
            t["market"] = m
            pooled.append(t)
            flips = [pd.DataFrame(fn(d, s)).R.mean() for s in range(1, 21)]
            s = summary(t)
            days = d.day.nunique()
            lines.append(f"   {m:<10} {s['n']:>4} trades/{days} days  win {s['win']:3.0f}%  avg {s['avgR']:+.3f}R  "
                         f"t {s['t']:+.1f}  total {s['totR']:+6.1f}R  spread {s['cost']:.2f}R  | coin flip {np.mean(flips):+.3f}R"
                         f" (rule beats {int((np.array(flips) < s['avgR']).sum())}/20)")
            allrows.append({"rule": name, "market": m, **s, "coin": float(np.mean(flips))})
        if pooled:
            P = pd.concat(pooled)
            s = summary(P)
            lines.append(f"   {'ALL':<10} {s['n']:>4} trades  win {s['win']:3.0f}%  avg {s['avgR']:+.3f}R  t {s['t']:+.1f}  "
                         f"total {s['totR']:+.1f}R  markets positive {sum(1 for x in pooled if x.R.mean() > 0)}/{len(pooled)}")
            P.to_csv(OUT / f"{name.split()[0]}_{'managed' if 'managed' in name else ''}trades.csv", index=False)
        lines.append("")
    txt = "\n".join(lines)
    print(txt)
    (OUT / "summary.txt").write_text(txt, encoding="utf-8")
    pd.DataFrame(allrows).to_csv(OUT / "by_market.csv", index=False)


if __name__ == "__main__":
    main()
