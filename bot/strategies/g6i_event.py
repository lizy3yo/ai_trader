"""US event days for gold: post-release momentum and pre-event drift (g6i_).

Events (g6i_common.events, server time via zoneinfo, DST-mismatch weeks
handled): NFP + CPI at 08:30 ET (15:30 server, 14:30 in mismatch weeks),
FOMC statement 14:00 ET (21:00 / 20:00). ev = nfp|cpi|fomc|data|all.

mode "post": on the bar whose CLOSE = release + k minutes, the move since
             the release (that close minus the OPEN of the bar starting at
             the release = last pre-release price) decides the side:
             dirmode "follow" (momentum) or "fade". |move| >= mm x ATR.
             Fill at release + k; exit via session.flat_at (before the
             daily close) or hold_min cap, plus wide ATR stop/target.
             k must be a multiple of the timeframe.
mode "pre":  on event days, enter at the bar closing at entry_hm (server)
             and exit at the first bar close >= release - pre_buf minutes
             (state exit column; on non-event days the exit clock is the
             normal release clock, so the random baseline gets the same
             time exit). side picks the direction ("long"/"short").
params: mode, ev, k, dirmode, mm, entry_hm, pre_buf, hold_min, side
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import StrategyCfg
from ..strategy import Signal, atr
from . import g6i_common as G
from . import ls_common as C

DEFAULTS = {"mode": "post", "ev": "data", "k": 30, "dirmode": "follow", "mm": 0.0,
            "entry_hm": "10:00", "pre_buf": 0, "hold_min": 0, "side": "both"}


def _default_clock(ev: str) -> int:
    return 21 * 60 if ev == "fomc" else 15 * 60 + 30


def add_indicators(df: pd.DataFrame, scfg: StrategyCfg) -> pd.DataFrame:
    p = C.params(scfg, DEFAULTS)
    out = df.copy()
    out["atr"] = atr(out, scfg.atr_period)
    step = C.tf_minutes(scfg)
    ev = G.events(str(p["ev"]))
    idx = out.index
    close_t = idx + pd.Timedelta(minutes=step)
    sig = np.zeros(len(out))
    if p["mode"] == "post":
        k = int(p["k"])
        # reference = open of the bar that starts exactly at the release
        opens = out["open"]
        ref = opens.reindex(ev)                      # NaN when no bar at release
        dec = ev + pd.Timedelta(minutes=k)           # decision bar close time
        pos = pd.Index(close_t).get_indexer(dec)
        ok = (pos >= 0) & np.isfinite(ref.to_numpy())
        c = out["close"].to_numpy()
        a = out["atr"].to_numpy()
        for j, r in zip(pos[ok], ref.to_numpy()[ok]):
            mv = c[j] - r
            if mv == 0 or not np.isfinite(a[j]) or abs(mv) < float(p["mm"]) * a[j]:
                continue
            d = np.sign(mv)
            sig[j] = d if p["dirmode"] == "follow" else -d
    elif p["mode"] == "pre":
        days = pd.DatetimeIndex(ev.normalize())
        rel = pd.Series(ev, index=days)
        rel = rel[~rel.index.duplicated()]
        bday = idx.normalize()
        is_ev = bday.isin(rel.index)
        eh, em = (int(x) for x in str(p["entry_hm"]).split(":"))
        at_entry = (close_t.hour * 60 + close_t.minute == eh * 60 + em) & (close_t.normalize() == bday)
        relt = pd.DatetimeIndex(rel.reindex(bday).to_numpy())
        before = relt > close_t
        sig = np.where(at_entry & is_ev & before, 1.0 if p["side"] != "short" else -1.0, 0.0)
        # exit target per bar: today's release - buf (event day) or the normal clock
        dflt = bday + pd.Timedelta(minutes=_default_clock(str(p["ev"])))
        tgt = pd.DatetimeIndex(np.where(is_ev, relt, dflt)) - pd.Timedelta(minutes=int(p["pre_buf"]))
        x = (close_t >= tgt).astype(float)
        out["ls_xl"] = x
        out["ls_xs"] = x
    else:
        raise ValueError(f"g6i_event: unknown mode {p['mode']!r}")
    out["ls_sig"] = sig
    return out


def warmup(scfg: StrategyCfg) -> int:
    return 5 * scfg.atr_period + 10


def evaluate(ind_window: pd.DataFrame, scfg: StrategyCfg) -> Signal:
    p = C.params(scfg, DEFAULTS)
    return C.signal_from_col(ind_window, scfg, p, warmup(scfg), f"event {p['mode']} {p['ev']}")


def exit_signal(ind_window: pd.DataFrame, scfg: StrategyCfg, direction: int, bars_held: int) -> bool:
    p = C.params(scfg, DEFAULTS)
    return C.exit_from_cols(ind_window, direction, bars_held,
                            G.hold_bars(p["hold_min"], C.tf_minutes(scfg)))
