"""grid_sim.py — STANDALONE grid / martingale / averaging-down simulator for GOLD.

Why standalone: bot/backtest.py holds ONE position at a time. Grid EAs hold a
BASKET of positions (leg 1, leg 2, ... leg N) that are closed together. This
file therefore re-implements the fill/cost model of bot/backtest.py from
scratch, with multi-position support, margin, and a hard ruin condition.

WHAT IS SIMULATED
-----------------
  seed        a first leg is opened in some direction (RSI mean-reversion,
              EMA trend, coin-flip, or always-long) on the bar AFTER the
              signal bar closes, filled at the next bar's OPEN + spread.
  grid ladder fixed ladder from the seed price: level_k = seed -/+ k*step,
              step = step_atr * ATR(14) frozen at basket open.
  sizing      leg k lots = base_lot * mult**k
                 mult = 1.0  -> classic flat grid
                 mult = 2.0  -> martingale doubling
                 mult < 1.0  -> anti-martingale (de-escalating) control
  basket TP   whole basket closes when price reaches the volume-weighted
              average entry +/- tp_frac * step. TP RATCHETS toward price as
              legs are added -- this is the mechanism that manufactures the
              famous "90% win rate".
  max legs    on_max_legs="hold" -> stop adding, ride it (vendor default)
              on_max_legs="cut"  -> basket stop-loss (capped control)

COSTS (measured against the live broker, non-negotiable)
  spread      55 points = 0.55 price units, paid on EVERY leg (longs fill at
              ask = bid + spread; shorts exit at ask). 0.01 lot => $0.55/leg RT.
  swap        POINTS mode. The mandated figure is ~$0.10/night per 0.1 lot,
              i.e. $1.00/night per 1.0 lot; charged at every date rollover on
              every carried leg, x3 on Wednesday. --swap-per-lot-night lets you
              stress the literal MT5 points reading (96.61 pts = $96.61/lot/nt).
  margin      MT5 metals margin = lots * 100 oz * price / leverage (1:100).
              A leg that does not fit in free margin is REFUSED (the real EA
              gets "not enough money" and just floats).
  stop-out    margin level = equity / used_margin; below 50% the broker
              liquidates. Modelled as: close the whole basket at that price.
  ruin        equity <= $100 on a $500 account = account dead, run stops.

Everything is causal: indicators are computed on closed bars only and a signal
on bar i is filled at bar i+1's open. Intrabar order uses the standard bar-path
assumption (up bar: open->low->high->close, down bar: open->high->low->close),
so adverse levels are visited in the right order relative to the TP.

USAGE
  python results/grid_sim.py --mode smoke
  python results/grid_sim.py --mode sweep  --csv data/gold_m15.csv --out results/grid_sweep_m15.json
  python results/grid_sim.py --mode ruin   --csv data/gold_m30.csv --out results/grid_ruin_m30.json
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd

CONTRACT = 100.0        # oz per 1.0 lot (XAUUSD)
POINT = 0.01


# --------------------------------------------------------------------------
# config
# --------------------------------------------------------------------------
@dataclass
class GridCfg:
    name: str = "grid"
    # --- entry seed ---
    seed: str = "revert"        # revert | trend | random | long | short
    rsi_period: int = 14
    rsi_lo: float = 30.0
    rsi_hi: float = 70.0
    ema_fast: int = 20
    ema_slow: int = 50
    # --- grid geometry ---
    atr_period: int = 14
    step_atr: float = 1.0       # grid spacing in ATRs (vendor range 0.5-1.5)
    max_legs: int = 10          # vendor range 5-15
    mult: float = 1.0           # 1.0 flat / 2.0 martingale / <1 anti
    base_lot: float = 0.01      # broker minimum
    tp_frac: float = 0.5        # basket TP distance from VWAP, in grid steps
    on_max_legs: str = "hold"   # hold | cut
    # --- costs / account ---
    spread_points: float = 55.0
    swap_per_lot_night: float = 1.0      # $ per 1.0 lot per rollover ($0.10/0.1lot)
    wed_triple: bool = True
    leverage: float = 100.0
    stopout_level: float = 0.5           # 50% margin level
    start_equity: float = 500.0
    ruin_equity: float = 100.0
    rng_seed: int = 7


# --------------------------------------------------------------------------
# data + causal indicators
# --------------------------------------------------------------------------
def _ewm(x: np.ndarray, alpha: float) -> np.ndarray:
    out = np.empty_like(x)
    acc = x[0]
    out[0] = acc
    for i in range(1, len(x)):
        acc = acc + alpha * (x[i] - acc)
        out[i] = acc
    return out


def load_bars(path: str) -> dict:
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    df["time"] = pd.to_datetime(df["time"])
    df = df.set_index("time")[["open", "high", "low", "close"]].astype(float).sort_index()
    o = df["open"].to_numpy()
    h = df["high"].to_numpy()
    l = df["low"].to_numpy()
    c = df["close"].to_numpy()
    idx = df.index
    out = {
        "time": idx, "open": o, "high": h, "low": l, "close": c,
        "dayord": idx.normalize().astype("int64").to_numpy() // 86_400_000_000_000,
        "dow": idx.dayofweek.to_numpy(),
        "month": (idx.year * 12 + idx.month).to_numpy(),
    }
    out["tf_min"] = tf_minutes(out)
    return out


def tf_minutes(bars: dict) -> int:
    """Bar duration in minutes = MODE of the index diffs (a leading data gap
    must not be mistaken for the timeframe, as bot/backtest.py also guards)."""
    d = bars["time"].to_series().diff().dropna()
    return max(1, int(d.mode().iloc[0].total_seconds() // 60))


def add_indicators(bars: dict, cfg: GridCfg) -> dict:
    """Causal only: Wilder ATR/RSI and EMAs, all recursive on closed bars."""
    h, l, c = bars["high"], bars["low"], bars["close"]
    pc = np.empty_like(c)
    pc[0] = c[0]
    pc[1:] = c[:-1]
    tr = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))
    atr = _ewm(tr, 1.0 / cfg.atr_period)

    d = np.zeros_like(c)
    d[1:] = np.diff(c)
    gain = _ewm(np.clip(d, 0, None), 1.0 / cfg.rsi_period)
    loss = _ewm(np.clip(-d, 0, None), 1.0 / cfg.rsi_period)
    rs = np.divide(gain, loss, out=np.full_like(gain, np.inf), where=loss > 0)
    rsi = 100.0 - 100.0 / (1.0 + rs)
    rsi[~np.isfinite(rsi)] = 100.0

    ef = _ewm(c, 2.0 / (cfg.ema_fast + 1.0))
    es = _ewm(c, 2.0 / (cfg.ema_slow + 1.0))
    return {"atr": atr, "rsi": rsi, "ema_f": ef, "ema_s": es}


# --------------------------------------------------------------------------
# the simulator
# --------------------------------------------------------------------------
def run_window(bars: dict, ind: dict, cfg: GridCfg, start: int, end: int) -> dict:
    """Simulate [start, end) bars. Returns a metrics dict."""
    o, h, l, c = bars["open"], bars["high"], bars["low"], bars["close"]
    dayord, dow, month = bars["dayord"], bars["dow"], bars["month"]
    atr, rsi, ef, es = ind["atr"], ind["rsi"], ind["ema_f"], ind["ema_s"]

    spread = cfg.spread_points * POINT
    rng = np.random.default_rng(cfg.rng_seed + start)

    balance = cfg.start_equity
    peak_eq = balance
    peak_i = 0
    max_eq_dd = 0.0
    max_basket_dd = 0.0        # deepest FLOATING loss of any single basket ($)
    max_used_margin = 0.0
    max_total_lots = 0.0
    max_legs_seen = 0
    swap_paid = 0.0
    spread_paid = 0.0
    stopouts = 0
    margin_blocked = 0
    baskets = 0
    wins = 0
    gross_win = 0.0
    gross_loss = 0.0

    # basket state
    b_dir = 0
    b_entry = []               # fill prices (ask for long, bid for short)
    b_lots = []
    b_seed = 0.0               # seed BID price (ladder anchor)
    b_step = 0.0
    b_dd = 0.0                 # this basket's worst floating pnl
    b_open_i = 0
    tot_lots = 0.0
    vwap = 0.0

    pending = 0
    cur_day = dayord[start]
    cur_month = month[start]
    month_marks = []           # (month, equity at last bar of that month)
    dead = False
    death_i = -1

    def floating(px_bid: float) -> float:
        if b_dir == 0:
            return 0.0
        if b_dir > 0:
            return tot_lots * (px_bid - vwap) * CONTRACT
        return tot_lots * (vwap - px_bid - spread) * CONTRACT

    def used_margin(px_bid: float) -> float:
        return tot_lots * CONTRACT * px_bid / cfg.leverage

    def tp_price() -> float:
        d = cfg.tp_frac * b_step
        return vwap + d if b_dir > 0 else vwap - spread - d

    def close_basket(px_bid: float, i: int) -> None:
        nonlocal balance, baskets, wins, gross_win, gross_loss
        nonlocal b_dir, b_entry, b_lots, tot_lots, vwap, b_dd, peak_eq, max_eq_dd, peak_i
        pnl = floating(px_bid)
        balance += pnl
        baskets += 1
        if pnl > 0:
            wins += 1
            gross_win += pnl
        else:
            gross_loss += pnl
        b_dir = 0
        b_entry = []
        b_lots = []
        tot_lots = 0.0
        vwap = 0.0
        b_dd = 0.0
        if balance > peak_eq:
            peak_eq, peak_i = balance, i
        max_eq_dd = max(max_eq_dd, peak_eq - balance)

    def add_leg(fill_px_bid: float, i: int) -> bool:
        """Open leg #len(b_lots) at ladder/seed price. False if margin refuses."""
        nonlocal tot_lots, vwap, spread_paid, max_used_margin, max_total_lots, max_legs_seen
        k = len(b_lots)
        lots = round(cfg.base_lot * (cfg.mult ** k), 8)
        lots = math.floor(lots / 0.01 + 1e-9) * 0.01      # broker lot step
        if lots < 0.01:
            return False
        need = lots * CONTRACT * fill_px_bid / cfg.leverage
        eq = balance + floating(fill_px_bid)
        if eq - used_margin(fill_px_bid) - need < 0.0:
            return False                                   # "not enough money"
        fill = fill_px_bid + spread if b_dir > 0 else fill_px_bid
        b_entry.append(fill)
        b_lots.append(lots)
        tot_lots = round(tot_lots + lots, 8)
        vwap = sum(e * v for e, v in zip(b_entry, b_lots)) / tot_lots
        spread_paid += lots * spread * CONTRACT
        max_used_margin = max(max_used_margin, used_margin(fill_px_bid))
        max_total_lots = max(max_total_lots, tot_lots)
        max_legs_seen = max(max_legs_seen, len(b_lots))
        return True

    def _mark(px: float, i: int = -1) -> None:
        nonlocal b_dd, max_basket_dd, peak_eq, max_eq_dd, peak_i
        f = floating(px)
        b_dd = min(b_dd, f)
        max_basket_dd = max(max_basket_dd, -b_dd)
        eq = balance + f
        if eq > peak_eq:
            peak_eq, peak_i = eq, i
        max_eq_dd = max(max_eq_dd, peak_eq - eq)

    def _kill_prices():
        """(ruin price, stop-out price): the exact prices at which equity hits
        $ruin_equity, and at which margin level hits stopout_level."""
        A = tot_lots * CONTRACT
        if A <= 0:
            return None, None
        s = cfg.stopout_level / cfg.leverage
        if b_dir > 0:
            p_ruin = vwap + (cfg.ruin_equity - balance) / A
            p_stop = (A * vwap - balance) / (A * (1.0 - s))
        else:
            p_ruin = vwap - spread - (cfg.ruin_equity - balance) / A
            p_stop = (balance + A * (vwap - spread)) / (A * (1.0 + s))
        return p_ruin, p_stop

    def adverse_segment(px_extreme: float, i: int) -> None:
        """Price moved AGAINST the basket, monotonically, to px_extreme (bid).
        Every event (ladder add, basket cut, broker stop-out, account ruin) is
        resolved at the EXACT price it occurs, in order — a bar that blows
        through five levels and the stop-out is handled level by level, not at
        the extreme. Uses y = -direction * price, so adverse == increasing y."""
        nonlocal margin_blocked, stopouts, dead, death_i
        if b_dir == 0:
            return
        sgn = -b_dir
        y_ext = sgn * px_extreme
        blocked = False
        while b_dir != 0:
            y_seed = sgn * b_seed
            if not blocked and len(b_lots) < cfg.max_legs:
                y_evt, evt = y_seed + len(b_lots) * b_step, "add"
            elif cfg.on_max_legs == "cut":
                y_evt, evt = y_seed + cfg.max_legs * b_step, "cut"
            else:
                y_evt, evt = float("inf"), None
            p_ruin, p_stop = _kill_prices()
            if p_ruin is None:
                y_k, kind = float("inf"), None
            else:
                yr, ys = sgn * p_ruin, sgn * p_stop
                y_k, kind = (yr, "ruin") if yr <= ys else (ys, "stopout")
            if y_k <= y_ext and y_k <= y_evt:
                px = sgn * y_k
                _mark(px, i)
                close_basket(px, i)
                if kind == "ruin":
                    dead = True
                    death_i = i
                else:
                    stopouts += 1
                return
            if evt is not None and y_evt <= y_ext:
                px = sgn * y_evt
                if evt == "cut":
                    _mark(px, i)
                    close_basket(px, i)
                    return
                if not add_leg(px, i):
                    margin_blocked += 1
                    blocked = True
                continue
            break
        if b_dir != 0:
            _mark(px_extreme, i)

    def favourable_segment(px_extreme: float, i: int) -> None:
        if b_dir == 0:
            return
        tp = tp_price()
        hit = px_extreme >= tp if b_dir > 0 else px_extreme <= tp
        if hit:
            close_basket(tp, i)

    warm = max(cfg.atr_period, cfg.rsi_period, cfg.ema_slow) * 5
    i0 = max(start, warm)

    for i in range(i0, end):
        # ---- nightly swap on every carried leg ----
        if dayord[i] != cur_day:
            if b_dir != 0 and tot_lots > 0:
                nights = 0.0
                for dd_ in range(int(cur_day) + 1, int(dayord[i]) + 1):
                    # 1970-01-01 was a Thursday -> dow = (ord+3) % 7 (Mon=0)
                    nights += 3.0 if (cfg.wed_triple and (dd_ + 3) % 7 == 2) else 1.0
                chg = nights * cfg.swap_per_lot_night * tot_lots
                balance -= chg
                swap_paid += chg
            cur_day = dayord[i]

        # ---- fill the pending seed at this bar's open ----
        if pending != 0 and b_dir == 0:
            a = atr[i - 1]
            if np.isfinite(a) and a > 0:
                b_dir = pending
                b_seed = o[i]
                b_step = cfg.step_atr * a
                b_entry = []
                b_lots = []
                tot_lots = 0.0
                vwap = 0.0
                b_dd = 0.0
                b_open_i = i
                if not add_leg(o[i], i):
                    b_dir = 0
                    margin_blocked += 1
            pending = 0

        # ---- walk the bar path ----
        if b_dir != 0:
            # a gap can open the bar already past the (ratcheted) basket TP
            favourable_segment(o[i], i)
        if b_dir != 0:
            up_bar = c[i] >= o[i]
            path = (l[i], h[i], c[i]) if up_bar else (h[i], l[i], c[i])
            adverse_low = b_dir > 0            # long baskets suffer on lows
            for k, px in enumerate(path):
                if b_dir == 0:
                    break
                if up_bar:
                    seg_down = (k == 0 or k == 2)
                else:
                    seg_down = (k == 1)
                if seg_down == adverse_low:
                    adverse_segment(px, i)
                else:
                    favourable_segment(px, i)
            if dead:
                death_i = i
                break

        # ---- month mark (equity, floating included) ----
        if month[i] != cur_month:
            month_marks.append(balance + floating(c[i]))
            cur_month = month[i]

        # ---- seed decision on this CLOSED bar, filled next bar's open ----
        if b_dir == 0 and pending == 0:
            s = cfg.seed
            if s == "revert":
                if rsi[i] < cfg.rsi_lo:
                    pending = 1
                elif rsi[i] > cfg.rsi_hi:
                    pending = -1
            elif s == "trend":
                pending = 1 if ef[i] > es[i] else -1
            elif s == "random":
                pending = 1 if rng.random() < 0.5 else -1
            elif s == "long":
                pending = 1
            elif s == "short":
                pending = -1

    # close whatever is left at the last close
    end_i = min(end, len(c)) - 1
    if b_dir != 0:
        close_basket(c[end_i], end_i)
    equity = balance
    month_marks.append(equity)

    marks = np.array(month_marks, dtype=float)
    m_pnl = np.diff(np.concatenate(([cfg.start_equity], marks)))
    months = len(m_pnl)
    bars_alive = (death_i if dead else end_i) - i0 + 1
    days_alive = bars_alive * bars["tf_min"] / (60.0 * 24.0)

    return {
        "cfg": cfg.name, "seed": cfg.seed, "step_atr": cfg.step_atr,
        "max_legs": cfg.max_legs, "mult": cfg.mult, "tp_frac": cfg.tp_frac,
        "on_max_legs": cfg.on_max_legs,
        "start_bar": int(i0), "end_bar": int(end_i),
        "start_date": str(bars["time"][i0].date()),
        "end_date": str(bars["time"][death_i if dead else end_i].date()),
        "baskets": baskets,
        "win_rate": round(100.0 * wins / baskets, 1) if baskets else 0.0,
        "end_equity": round(equity, 2),
        "return_pct": round(100.0 * (equity / cfg.start_equity - 1.0), 1),
        "avg_win": round(gross_win / wins, 2) if wins else 0.0,
        "avg_loss": round(gross_loss / (baskets - wins), 2) if baskets - wins else 0.0,
        "profit_factor": round(gross_win / abs(gross_loss), 3) if gross_loss else float("inf"),
        "gross_win": round(gross_win, 2),
        "gross_loss": round(gross_loss, 2),
        "max_basket_dd": round(max_basket_dd, 2),
        "max_equity_dd": round(max_eq_dd, 2),
        "peak_equity": round(peak_eq, 2),
        "days_to_peak": round(max(peak_i - i0, 0) * bars["tf_min"] / 1440.0, 1),
        # the vendor's marketing number: $/month during the climb to the peak
        "usd_per_month_while_climbing": round(
            (peak_eq - cfg.start_equity) / max((peak_i - i0) * bars["tf_min"] / 43829.0, 1e-9), 2)
        if peak_i > i0 else 0.0,
        "max_used_margin": round(max_used_margin, 2),
        "margin_pct_of_500": round(100.0 * max_used_margin / cfg.start_equity, 1),
        "max_total_lots": round(max_total_lots, 2),
        "max_legs_seen": max_legs_seen,
        "margin_blocked": margin_blocked,
        "stopouts": stopouts,
        "swap_paid": round(swap_paid, 2),
        "spread_paid": round(spread_paid, 2),
        "ruin": bool(dead),
        "days_alive": round(days_alive, 1),
        "months": months,
        "avg_month_usd": round(float(m_pnl.mean()), 2) if months else 0.0,
        "pos_months_pct": round(100.0 * float((m_pnl > 0).mean()), 0) if months else 0.0,
        "worst_month": round(float(m_pnl.min()), 2) if months else 0.0,
        "trades_per_month": round(baskets / months, 1) if months else 0.0,
        # baskets per 30.44 days ALIVE (unbiased by a short final month)
        "baskets_per_month_rate": round(baskets / (days_alive / 30.44), 1) if days_alive else 0.0,
    }


# --------------------------------------------------------------------------
# drivers
# --------------------------------------------------------------------------
def ruin_study(bars: dict, cfg: GridCfg, n_offsets: int, horizon_bars: int) -> dict:
    ind = add_indicators(bars, cfg)
    n = len(bars["close"])
    warm = max(cfg.atr_period, cfg.rsi_period, cfg.ema_slow) * 5
    last_start = n - horizon_bars
    if last_start <= warm:
        n_offsets, last_start = 1, warm
    starts = np.linspace(warm, max(warm, last_start), n_offsets).astype(int)
    runs = [run_window(bars, ind, cfg, int(s), min(n, int(s) + horizon_bars)) for s in starts]
    ruined = [r for r in runs if r["ruin"]]
    alive = [r for r in runs if not r["ruin"]]
    ttr = sorted(r["days_alive"] for r in ruined)
    return {
        "runs": len(runs),
        "ruin_pct": round(100.0 * len(ruined) / len(runs), 1),
        "median_days_to_ruin": round(float(np.median(ttr)), 1) if ttr else None,
        "min_days_to_ruin": round(ttr[0], 1) if ttr else None,
        "median_win_rate": round(float(np.median([r["win_rate"] for r in runs])), 1),
        "median_return_pct": round(float(np.median([r["return_pct"] for r in runs])), 1),
        "median_avg_month_usd": round(float(np.median([r["avg_month_usd"] for r in runs])), 2),
        "median_avg_month_usd_survivors": (
            round(float(np.median([r["avg_month_usd"] for r in alive])), 2) if alive else None),
        "max_basket_dd_worst": round(max(r["max_basket_dd"] for r in runs), 2),
        "max_used_margin_worst": round(max(r["max_used_margin"] for r in runs), 2),
        "median_baskets_per_month": round(
            float(np.median([r["trades_per_month"] for r in runs])), 1),
        "median_max_legs_seen": int(np.median([r["max_legs_seen"] for r in runs])),
        "margin_blocked_total": int(sum(r["margin_blocked"] for r in runs)),
    }


def capacity_study(bars: dict, cfgs: list, horizons_days: tuple) -> list:
    """The structural argument, independent of any entry signal.

    For a fully-loaded basket, solve for D_death = the adverse price distance
    from the seed at which equity hits $100 on a $500 account:
        loss(p) = 100 * S * (p - vwap)  with S = total lots
        D_death = step * sum(k*lot_k)/S - spread + (start_eq - ruin_eq)/(100*S)
    Then measure EMPIRICALLY how often gold actually travels that far, from a
    random starting bar, inside a plausible basket lifetime. That product --
    P(death per basket) x baskets per month -- is the risk of ruin, and it does
    not depend on the entry rule at all.
    """
    ind = add_indicators(bars, cfgs[0])
    atr = ind["atr"]
    c, hi, lo = bars["close"], bars["high"], bars["low"]
    bpd = max(1, 1440 // bars["tf_min"])
    fwd = {}
    for H in horizons_days:
        n = H * bpd
        smin = pd.Series(lo)[::-1].rolling(n, min_periods=1).min()[::-1].shift(-1)
        smax = pd.Series(hi)[::-1].rolling(n, min_periods=1).max()[::-1].shift(-1)
        fwd[H] = (c - smin.to_numpy(), smax.to_numpy() - c)   # drop, rise

    out = []
    for cfg in cfgs:
        lots = []
        for k in range(cfg.max_legs):
            v = math.floor(cfg.base_lot * (cfg.mult ** k) / 0.01 + 1e-9) * 0.01
            if v < 0.01:
                break
            lots.append(v)
        S = sum(lots)
        wk = sum(k * v for k, v in enumerate(lots))
        spread = cfg.spread_points * POINT
        D = cfg.step_atr * atr * (wk / S) - spread + (cfg.start_equity - cfg.ruin_equity) / (100.0 * S)
        px = np.nanmedian(c)
        row = {
            "cfg": cfg.name, "legs_fundable_by_ladder": len(lots),
            "total_lots": round(S, 2),
            "margin_at_deepest_leg": round(S * float(px), 2),
            "margin_pct_of_500": round(100.0 * S * float(px) / cfg.start_equity, 1),
            "median_D_death_usd": round(float(np.nanmedian(D)), 1),
            "median_D_death_pct_of_price": round(100.0 * float(np.nanmedian(D)) / float(px), 2),
        }
        for H in horizons_days:
            drop, rise = fwd[H]
            m = np.isfinite(drop) & np.isfinite(rise) & np.isfinite(D)
            p_long = float((drop[m] >= D[m]).mean())    # a LONG basket dies on drops
            p_short = float((rise[m] >= D[m]).mean())   # a SHORT basket dies on rises
            row[f"P_death_{H}d_pct"] = round(100.0 * 0.5 * (p_long + p_short), 1)
            row[f"P_death_{H}d_long_pct"] = round(100.0 * p_long, 1)
            row[f"P_death_{H}d_short_pct"] = round(100.0 * p_short, 1)
        out.append(row)
    return out


def _fmt(r: dict) -> str:
    return (f"{r['cfg']:<26}{r['baskets']:>6}{r['win_rate']:>8.1f}"
            f"{r['return_pct']:>10.1f}{r['avg_month_usd']:>10.2f}"
            f"{r['max_basket_dd']:>11.2f}{r['max_used_margin']:>10.2f}"
            f"{r['max_legs_seen']:>6}{'YES' if r['ruin'] else 'no':>7}"
            f"{r['days_alive']:>9.0f}")


HDR = (f"{'config':<26}{'bskts':>6}{'win%':>8}{'ret%':>10}{'$/mo':>10}"
       f"{'maxBskDD':>11}{'maxMargin':>10}{'legs':>6}{'RUIN':>7}{'days':>9}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="smoke",
                    choices=["smoke", "sweep", "ruin", "capacity", "swapstress", "seeds"])
    ap.add_argument("--csv", default="data/gold_m15.csv")
    ap.add_argument("--out")
    ap.add_argument("--offsets", type=int, default=40)
    ap.add_argument("--horizon-days", type=int, default=365)
    ap.add_argument("--swap-per-lot-night", type=float, default=1.0)
    ap.add_argument("--max-bars", type=int, default=0)
    args = ap.parse_args()

    bars = load_bars(args.csv)
    if args.max_bars and len(bars["close"]) > args.max_bars:
        for k in ("time", "open", "high", "low", "close", "dayord", "dow", "month"):
            bars[k] = bars[k][-args.max_bars:]
        bars["tf_min"] = tf_minutes(bars)
    tf_min = bars["tf_min"]
    bars_per_day = max(1, 1440 // tf_min)
    print(f"data {args.csv}: {len(bars['close'])} bars  {bars['time'][0]} -> {bars['time'][-1]}  "
          f"(TF={tf_min}m)\n")

    base = replace(GridCfg(), swap_per_lot_night=args.swap_per_lot_night)

    # --------------------------------------------------------------
    if args.mode == "smoke":
        cfg = replace(base, name="smoke-flat-grid")
        ind = add_indicators(bars, cfg)
        r = run_window(bars, ind, cfg, 0, min(len(bars["close"]), 20000))
        print(HDR)
        print(_fmt(r))
        print(json.dumps(r, indent=1))
        return

    # --------------------------------------------------------------
    if args.mode == "sweep":
        cfgs = []
        # (a) CLASSIC FLAT GRID — vendor-realistic geometry
        for step in (0.5, 1.0, 1.5):
            for legs in (5, 10, 15):
                for tp in (0.3, 0.5):
                    cfgs.append(replace(base, name=f"flat s{step} L{legs} tp{tp}",
                                        mult=1.0, step_atr=step, max_legs=legs,
                                        tp_frac=tp, on_max_legs="hold"))
        # (b) MARTINGALE — doubling and the softer 1.5x vendors prefer
        for m in (1.5, 2.0):
            for legs in (5, 8, 10):
                cfgs.append(replace(base, name=f"mart x{m} L{legs}", mult=m,
                                    step_atr=1.0, max_legs=legs, tp_frac=0.5,
                                    on_max_legs="hold"))
        # (c) CAPPED / ANTI-MARTINGALE CONTROLS
        for m, legs, om in ((1.0, 5, "cut"), (1.0, 10, "cut"), (1.0, 1, "cut")):
            cfgs.append(replace(base, name=f"ctl x{m} L{legs} {om}", mult=m,
                                step_atr=1.0, max_legs=legs, tp_frac=0.5,
                                on_max_legs=om))
        # anti-martingale needs headroom ABOVE the 0.01 broker minimum to
        # de-escalate at all: base 0.04 -> 0.02 -> 0.01 -> refused (self-capping)
        for om in ("hold", "cut"):
            cfgs.append(replace(base, name=f"ctl anti x0.5 b0.04 {om}", mult=0.5,
                                base_lot=0.04, step_atr=1.0, max_legs=10,
                                tp_frac=0.5, on_max_legs=om))
        # (d) SEED NULL TEST — does the win rate come from the signal?
        for s in ("random", "trend", "long"):
            cfgs.append(replace(base, name=f"seed={s} flat L10", seed=s, mult=1.0,
                                step_atr=1.0, max_legs=10, tp_frac=0.5,
                                on_max_legs="hold"))

        ind_cache = {}
        rows = []
        print(HDR)
        for cfg in cfgs:
            key = (cfg.atr_period, cfg.rsi_period, cfg.ema_slow)
            if key not in ind_cache:
                ind_cache[key] = add_indicators(bars, cfg)
            r = run_window(bars, ind_cache[key], cfg, 0, len(bars["close"]))
            r["full_cfg"] = asdict(cfg)
            rows.append(r)
            print(_fmt(r))
        if args.out:
            Path(args.out).write_text(json.dumps(rows, indent=1), encoding="utf-8")
            print(f"\n-> {args.out}")
        return

    # --------------------------------------------------------------
    if args.mode == "capacity":
        cfgs = [
            replace(base, name="flat L15 s0.5", mult=1.0, step_atr=0.5, max_legs=15),
            replace(base, name="flat L10 s1.0", mult=1.0, step_atr=1.0, max_legs=10),
            replace(base, name="flat L10 s1.5", mult=1.0, step_atr=1.5, max_legs=10),
            replace(base, name="flat L5  s1.5", mult=1.0, step_atr=1.5, max_legs=5),
            replace(base, name="flat L3  s3.0", mult=1.0, step_atr=3.0, max_legs=3),
            replace(base, name="mart x1.5 L8 s1.0", mult=1.5, step_atr=1.0, max_legs=8),
            replace(base, name="mart x2.0 L5 s1.0", mult=2.0, step_atr=1.0, max_legs=5),
            replace(base, name="single 0.01 no grid", mult=1.0, step_atr=1.0, max_legs=1),
        ]
        rows = capacity_study(bars, cfgs, (5, 20, 60))
        print(f"{'geometry':<22}{'legs':>5}{'lots':>7}{'margin$':>9}{'%of500':>8}"
              f"{'D_death$':>10}{'%px':>7}{'P5d%':>7}{'P20d%':>7}{'P60d%':>7}")
        for r in rows:
            print(f"{r['cfg']:<22}{r['legs_fundable_by_ladder']:>5}{r['total_lots']:>7.2f}"
                  f"{r['margin_at_deepest_leg']:>9.0f}{r['margin_pct_of_500']:>8.0f}"
                  f"{r['median_D_death_usd']:>10.1f}{r['median_D_death_pct_of_price']:>7.2f}"
                  f"{r['P_death_5d_pct']:>7.1f}{r['P_death_20d_pct']:>7.1f}"
                  f"{r['P_death_60d_pct']:>7.1f}")
        if args.out:
            Path(args.out).write_text(json.dumps(rows, indent=1), encoding="utf-8")
            print(f"\n-> {args.out}")
        return

    if args.mode == "seeds":
        # Does the ruin come from the SIGNAL or from the GRID? Same geometry,
        # five different entry rules including a coin flip.
        horizon = args.horizon_days * bars_per_day
        out = []
        print(f"{'seed rule':<24}{'runs':>5}{'RUIN%':>7}{'medTTR_d':>9}{'medWin%':>8}"
              f"{'medRet%':>9}{'med$/mo':>9}{'worstBskDD':>12}{'bsk/mo':>8}")
        for sd in ("revert", "trend", "random", "long", "short"):
            cfg = replace(base, name=f"seed={sd} flat L10 s1.0", seed=sd, mult=1.0,
                          step_atr=1.0, max_legs=10, tp_frac=0.5, on_max_legs="hold")
            st = ruin_study(bars, cfg, args.offsets, horizon)
            st["cfg"] = cfg.name
            st["full_cfg"] = asdict(cfg)
            out.append(st)
            print(f"{cfg.name:<24}{st['runs']:>5}{st['ruin_pct']:>7.1f}"
                  f"{(st['median_days_to_ruin'] or 0):>9.0f}{st['median_win_rate']:>8.1f}"
                  f"{st['median_return_pct']:>9.1f}{st['median_avg_month_usd']:>9.2f}"
                  f"{st['max_basket_dd_worst']:>12.2f}{st['median_baskets_per_month']:>8.1f}")
        if args.out:
            Path(args.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
            print(f"\n-> {args.out}")
        return

    if args.mode in ("ruin", "swapstress"):
        horizon = args.horizon_days * bars_per_day
        cfgs = [
            replace(base, name="flat L10 s1.0 tp0.5", mult=1.0, step_atr=1.0,
                    max_legs=10, tp_frac=0.5, on_max_legs="hold"),
            replace(base, name="flat L15 s0.5 tp0.3", mult=1.0, step_atr=0.5,
                    max_legs=15, tp_frac=0.3, on_max_legs="hold"),
            replace(base, name="flat L5 s1.5 tp0.5", mult=1.0, step_atr=1.5,
                    max_legs=5, tp_frac=0.5, on_max_legs="hold"),
            replace(base, name="mart x2.0 L5 s1.0", mult=2.0, step_atr=1.0,
                    max_legs=5, tp_frac=0.5, on_max_legs="hold"),
            replace(base, name="mart x1.5 L8 s1.0", mult=1.5, step_atr=1.0,
                    max_legs=8, tp_frac=0.5, on_max_legs="hold"),
            replace(base, name="ctl flat L10 CUT", mult=1.0, step_atr=1.0,
                    max_legs=10, tp_frac=0.5, on_max_legs="cut"),
            replace(base, name="flat L3 s3.0 tp0.5", mult=1.0, step_atr=3.0,
                    max_legs=3, tp_frac=0.5, on_max_legs="hold"),
            replace(base, name="ctl anti x0.5 b0.04 CUT", mult=0.5, base_lot=0.04,
                    step_atr=1.0, max_legs=10, tp_frac=0.5, on_max_legs="cut"),
            replace(base, name="ctl single L1 CUT", mult=1.0, step_atr=1.0,
                    max_legs=1, tp_frac=0.5, on_max_legs="cut"),
        ]
        out = []
        print(f"{'config':<24}{'runs':>5}{'RUIN%':>7}{'medTTR_d':>9}{'medWin%':>8}"
              f"{'medRet%':>9}{'med$/mo':>9}{'worstBskDD':>12}{'worstMgn':>9}{'bsk/mo':>8}")
        for cfg in cfgs:
            s = ruin_study(bars, cfg, args.offsets, horizon)
            s["cfg"] = cfg.name
            s["full_cfg"] = asdict(cfg)
            s["horizon_days"] = args.horizon_days
            out.append(s)
            print(f"{cfg.name:<24}{s['runs']:>5}{s['ruin_pct']:>7.1f}"
                  f"{(s['median_days_to_ruin'] or 0):>9.0f}{s['median_win_rate']:>8.1f}"
                  f"{s['median_return_pct']:>9.1f}{s['median_avg_month_usd']:>9.2f}"
                  f"{s['max_basket_dd_worst']:>12.2f}{s['max_used_margin_worst']:>9.2f}"
                  f"{s['median_baskets_per_month']:>8.1f}")
        if args.out:
            Path(args.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
            print(f"\n-> {args.out}")
        return


if __name__ == "__main__":
    main()
