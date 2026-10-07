"""ICT Fair Value Gap retrace / fade (research family prefix fvg_).

A bullish FVG is the 3-bar imbalance  bar1.high < bar3.low  (price never
traded in [bar1.high, bar3.low] during the pattern); bearish is the mirror.
The zone becomes known ONLY when bar3 closes. We track the most recent
qualifying zone per direction with a forward-only state machine (row i uses
rows <= i exclusively — no lookahead, no full-frame statistics) and fire a
trigger on the FIRST closed bar that retraces into the zone.

mode "cont": enter in the gap's direction on the retrace (ICT continuation
             entry; restricted to a 1-hour killzone via the harness key
             "session.window" this is the "Silver Bullet" template).
mode "fade": enter AGAINST the gap as price returns to fill it (gap-fill
             bet), optionally targeting the far edge (tp_mode "gap").

Params (scfg.params):
    min_gap_atr : gap height must be >= this x ATR at formation (noise filter)
    disp_atr    : middle-bar body must be >= this x ATR (displacement filter)
    max_age     : zone expires after this many bars without a touch
    min_age     : touches before this age kill the zone without a trigger
                  (fade variant: only OLD unfilled gaps are worth fading)
    entry_depth : fraction of the gap price must penetrate to trigger
                  (0 = touch the near edge, 0.5 = ICT "consequent encroachment")
    confirm     : 1 = the trigger bar must CLOSE back beyond the trigger level
                  (rejection); 0 = any touch triggers
    mode        : "cont" | "fade"
    trend_filter: 1 = only trade aligned with EMA(scfg.ema_trend)
    sides       : "both" | "long" | "short"
    sl_mode     : "atr" (scfg.atr_sl_mult x ATR) | "zone" (far zone edge + pad)
    zone_pad_atr: pad beyond the zone edge for sl_mode "zone", in ATRs
    tp_mode     : "atr" (scfg.atr_tp_mult x ATR) | "gap" (fade only: target
                  the full gap fill = far edge of the zone)
    min_tp_atr  : floor for the "gap" target, in ATRs (avoid spread-sized TPs)

SL/TP baseline: scfg.atr_sl_mult / scfg.atr_tp_mult x ATR (shared knobs).
A blow-through bar (close beyond the far edge on first touch) never triggers:
the imbalance was consumed, the zone dies. Triggered zones are consumed too —
one trade per zone. Entries whose fill bar would open in the 00:00-01:00
rollover hour are refused regardless of session config.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import TIMEFRAME_MINUTES, StrategyCfg
from ..strategy import LONG, SHORT, Signal, atr, ema, flat, valid_atr

DEFAULTS = {
    "min_gap_atr": 0.3,
    "disp_atr": 0.0,
    "max_age": 24,
    "min_age": 0,
    "entry_depth": 0.0,
    "confirm": 0,
    "mode": "cont",
    "trend_filter": 0,
    "sides": "both",
    "sl_mode": "atr",
    "zone_pad_atr": 0.25,
    "tp_mode": "atr",
    "min_tp_atr": 0.5,
}


def _bar_minutes(scfg: StrategyCfg) -> int:
    return TIMEFRAME_MINUTES.get(scfg.timeframe.strip().upper(), 30)


def add_indicators(df: pd.DataFrame, scfg: StrategyCfg) -> pd.DataFrame:
    p = {**DEFAULTS, **scfg.params}
    out = df.copy()
    out["atr"] = atr(out, scfg.atr_period)
    out["ema_trend"] = ema(out["close"], scfg.ema_trend)

    o = out["open"].to_numpy(float)
    h = out["high"].to_numpy(float)
    l = out["low"].to_numpy(float)
    c = out["close"].to_numpy(float)
    a = out["atr"].to_numpy(float)
    n = len(out)
    body = np.abs(c - o)

    min_gap = float(p["min_gap_atr"])
    disp = float(p["disp_atr"])
    max_age = int(p["max_age"])
    min_age = int(p["min_age"])
    depth = float(p["entry_depth"])
    confirm = bool(int(p["confirm"]))

    trig_dir = np.zeros(n, dtype=np.int8)   # +1 bull-zone trigger, -1 bear
    trig_top = np.full(n, np.nan)
    trig_bot = np.full(n, np.nan)

    bull = None  # (top, bot, age) — zone below price, formed by an up-move
    bear = None  # (top, bot, age) — zone above price, formed by a down-move

    # Forward-only state machine: bar i first tests the zones known BEFORE i,
    # then may register a zone formed BY bar i (usable from bar i+1 on).
    for i in range(2, n):
        fired_bull = fired_bear = None

        if bull is not None:
            top, bot, age = bull
            age += 1
            lvl = top - depth * (top - bot)
            if l[i] <= lvl:                       # reached trigger depth
                ok = c[i] >= bot                  # not blown through the gap
                if confirm:
                    ok = ok and c[i] >= lvl       # closed back above
                if ok and age >= min_age:
                    fired_bull = (top, bot)
                bull = None                       # consumed or dead either way
            elif age >= max_age:
                bull = None
            else:
                bull = (top, bot, age)

        if bear is not None:
            top, bot, age = bear
            age += 1
            lvl = bot + depth * (top - bot)
            if h[i] >= lvl:
                ok = c[i] <= top
                if confirm:
                    ok = ok and c[i] <= lvl
                if ok and age >= min_age:
                    fired_bear = (top, bot)
                bear = None
            elif age >= max_age:
                bear = None
            else:
                bear = (top, bot, age)

        # both directions triggering on one giant bar is ambiguous — skip
        if fired_bull is not None and fired_bear is None:
            trig_dir[i] = 1
            trig_top[i], trig_bot[i] = fired_bull
        elif fired_bear is not None and fired_bull is None:
            trig_dir[i] = -1
            trig_top[i], trig_bot[i] = fired_bear

        # ---- formation by bar i (bar i = "bar3" of the pattern) ----
        ai = a[i]
        if np.isfinite(ai) and ai > 0:
            if l[i] > h[i - 2]:                   # bullish imbalance
                gap = l[i] - h[i - 2]
                if gap >= min_gap * ai and body[i - 1] >= disp * ai:
                    bull = (l[i], h[i - 2], 0)
            elif h[i] < l[i - 2]:                 # bearish imbalance
                gap = l[i - 2] - h[i]
                if gap >= min_gap * ai and body[i - 1] >= disp * ai:
                    bear = (l[i - 2], h[i], 0)

    out["trig_dir"] = trig_dir
    out["trig_top"] = trig_top
    out["trig_bot"] = trig_bot
    return out


def warmup(scfg: StrategyCfg) -> int:
    p = {**DEFAULTS, **scfg.params}
    # 5x spans keep the ewm seeds negligible; zones need max_age of history
    return max(5 * scfg.ema_trend, 5 * scfg.atr_period, int(p["max_age"]) + 5) + 10


def evaluate(ind_window: pd.DataFrame, scfg: StrategyCfg) -> Signal:
    p = {**DEFAULTS, **scfg.params}
    if len(ind_window) < warmup(scfg):
        return flat("not enough bars")
    last = ind_window.iloc[-1]
    if not valid_atr(last):
        return flat("bad ATR")

    zdir = int(last["trig_dir"])
    if zdir == 0:
        return flat("no FVG trigger")

    # entry fills at the NEXT bar open == this bar's close time
    entry_ts = ind_window.index[-1] + pd.Timedelta(minutes=_bar_minutes(scfg))
    if entry_ts.hour == 0:
        return flat("rollover hour forbidden")

    fade = str(p["mode"]) == "fade"
    direction = -zdir if fade else zdir

    sides = str(p["sides"])
    if (direction == LONG and sides == "short") or (direction == SHORT and sides == "long"):
        return flat("side disabled")

    c = float(last["close"])
    if int(p["trend_filter"]):
        trend = float(last["ema_trend"])
        if direction == LONG and c <= trend:
            return flat("against trend filter")
        if direction == SHORT and c >= trend:
            return flat("against trend filter")

    a = float(last["atr"])
    top, bot = float(last["trig_top"]), float(last["trig_bot"])

    # ---- stop distance ----
    if str(p["sl_mode"]) == "zone":
        pad = float(p["zone_pad_atr"]) * a
        if direction == LONG:
            sl_dist = c - (bot - pad)     # stop beyond the lower zone edge
        else:
            sl_dist = (top + pad) - c     # stop beyond the upper zone edge
        sl_dist = max(sl_dist, 0.5 * a)   # floor: never a spread-sized stop
    else:
        sl_dist = scfg.atr_sl_mult * a

    # ---- target distance ----
    if fade and str(p["tp_mode"]) == "gap":
        # target the completed gap fill = far edge of the zone
        tp_dist = (c - bot) if direction == SHORT else (top - c)
        tp_dist = max(tp_dist, float(p["min_tp_atr"]) * a)
    else:
        tp_dist = scfg.atr_tp_mult * a

    zone = "bull" if zdir == LONG else "bear"
    side = "long" if direction == LONG else "short"
    kind = "fade" if fade else "cont"
    return Signal(direction, c, sl_dist, tp_dist, a,
                  f"FVG {kind} {side}: {zone} zone [{bot:.2f},{top:.2f}]")
