"""Shared causal zone bookkeeping for the ob_ family (order blocks / supply-demand).

The core object is a "zone": a price band [lo, hi] created at the CLOSE of some
bar (the displacement / leave bar), tradable from the NEXT bar on, until it
expires, is consumed by its first touch, or is invalidated by a close through
its far edge. `run_zone_machine` is a single O(n) forward pass over numpy
arrays: at each bar i it first evaluates zones created at bars < i against bar
i's OHLC (bar i is a CLOSED bar when the strategy sees it), and only then
ingests a zone created at bar i itself. Strictly causal by construction.

Signal semantics (bull/demand zone; supply is the mirror):
- price left the zone upward at creation (enforced by the caller)
- a "touch episode" starts when a bar's low first re-enters (low <= zone_hi)
- entry_mode "touch":   signal if that bar CLOSES at/above zone_lo (zone held)
- entry_mode "confirm": signal only if it closes back ABOVE zone_hi (rejection)
- one_touch: the first touch episode consumes the zone either way
- close below zone_lo invalidates the zone; with breaker=True the invalidated
  zone flips into an opposite-direction "breaker" zone (and only flipped zones
  emit signals in breaker mode).
"""
from __future__ import annotations

import numpy as np

from ..strategy import LONG, SHORT, Signal, flat, valid_atr


def run_zone_machine(
    low: np.ndarray,
    high: np.ndarray,
    close: np.ndarray,
    bull_new_lo: np.ndarray,
    bull_new_hi: np.ndarray,
    bear_new_lo: np.ndarray,
    bear_new_hi: np.ndarray,
    expiry: int,
    one_touch: bool,
    entry_mode: str,
    breaker: bool,
):
    """Returns (long_sig, short_sig, sig_lo, sig_hi); sig_* carry the edges of
    the zone that produced the signal (NaN elsewhere) for structural stops."""
    n = len(close)
    long_sig = np.zeros(n, dtype=bool)
    short_sig = np.zeros(n, dtype=bool)
    sig_lo = np.full(n, np.nan)
    sig_hi = np.full(n, np.nan)

    # one active zone slot per side — the most recent displacement wins
    bl_lo = bl_hi = np.nan
    bl_born = -1
    bl_alive = bl_in = bl_brk = False
    br_lo = br_hi = np.nan
    br_born = -1
    br_alive = br_in = br_brk = False

    confirm = str(entry_mode) == "confirm"

    for i in range(n):
        l = low[i]
        h = high[i]
        c = close[i]

        # ---------- demand (bull) zone: created before bar i ----------
        if bl_alive and i - bl_born > expiry:
            bl_alive = False
        # invalidation FIRST: a close through the far edge kills (or flips) the
        # zone before any trigger logic — a violating bar can never signal
        # anyway (its close fails both touch and confirm conditions)
        if bl_alive and c < bl_lo:
            bl_alive = False
            if breaker and not bl_brk:
                # failed demand flips into supply; price is below it now, so
                # start "in touch" — the episode arms once price leaves the band
                br_lo, br_hi, br_born = bl_lo, bl_hi, i
                br_alive, br_in, br_brk = True, True, True
        if bl_alive:
            inz = l <= bl_hi
            if inz and not bl_in:
                ok = (c >= bl_hi) if confirm else (c >= bl_lo)
                if ok and (bl_brk or not breaker):
                    long_sig[i] = True
                    sig_lo[i] = bl_lo
                    sig_hi[i] = bl_hi
                # in breaker mode a PLAIN zone must survive touches so it can
                # still fail and flip later; only signal-emitting zones consume
                if one_touch and (bl_brk or not breaker):
                    bl_alive = False
            bl_in = inz

        # ---------- supply (bear) zone ----------
        if br_alive and i - br_born > expiry:
            br_alive = False
        if br_alive and c > br_hi:
            br_alive = False
            if breaker and not br_brk:
                bl_lo, bl_hi, bl_born = br_lo, br_hi, i
                bl_alive, bl_in, bl_brk = True, True, True
        if br_alive:
            inz = h >= br_lo
            if inz and not br_in:
                ok = (c <= br_lo) if confirm else (c <= br_hi)
                if ok and (br_brk or not breaker):
                    short_sig[i] = True
                    sig_lo[i] = br_lo
                    sig_hi[i] = br_hi
                if one_touch and (br_brk or not breaker):
                    br_alive = False
            br_in = inz

        # ---------- ingest zones created at THIS bar's close ----------
        if np.isfinite(bull_new_lo[i]):
            bl_lo, bl_hi, bl_born = bull_new_lo[i], bull_new_hi[i], i
            bl_alive, bl_in, bl_brk = True, False, False
        if np.isfinite(bear_new_lo[i]):
            br_lo, br_hi, br_born = bear_new_lo[i], bear_new_hi[i], i
            br_alive, br_in, br_brk = True, False, False

    return long_sig, short_sig, sig_lo, sig_hi


def zone_evaluate(ind_window, scfg, p: dict, min_bars: int, tag: str) -> Signal:
    """Shared evaluate(): read the precomputed causal signal flags off the last
    CLOSED bar and turn them into a Signal with ATR or structural stops."""
    if len(ind_window) < min_bars:
        return flat("not enough bars")
    last = ind_window.iloc[-1]
    if not valid_atr(last):
        return flat("bad ATR")

    go_long = float(last["long_sig"]) > 0
    go_short = float(last["short_sig"]) > 0
    if not (go_long or go_short):
        return flat("no zone retest")
    if go_long and go_short:
        return flat("conflicting zone signals")

    direction = LONG if go_long else SHORT
    sides = str(p["sides"])
    if (direction == LONG and sides == "short") or (direction == SHORT and sides == "long"):
        return flat("side disabled")

    c = float(last["close"])
    a = float(last["atr"])
    if p["trend_filter"]:
        t = float(last["ema_trend"])
        if direction == LONG and c <= t:
            return flat("against trend filter")
        if direction == SHORT and c >= t:
            return flat("against trend filter")

    zlo = float(last["sig_zlo"])
    zhi = float(last["sig_zhi"])
    if str(p["stop_mode"]) == "zone":
        pad = float(p["zone_pad_atr"]) * a
        sl_dist = (c - (zlo - pad)) if direction == LONG else ((zhi + pad) - c)
        sl_dist = max(sl_dist, float(p["min_sl_atr"]) * a)
        tp_dist = float(p["rr"]) * sl_dist
    else:
        sl_dist = scfg.atr_sl_mult * a
        tp_dist = scfg.atr_tp_mult * a
    if not (np.isfinite(sl_dist) and sl_dist > 0 and np.isfinite(tp_dist) and tp_dist > 0):
        return flat("bad SL/TP distance")

    side = "long" if direction == LONG else "short"
    return Signal(direction, c, sl_dist, tp_dist, a,
                  f"{tag} {side} retest of zone {zlo:.2f}-{zhi:.2f}")
