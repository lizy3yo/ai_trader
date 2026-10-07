"""Diagnostic (TRAIN split only): forward expectancy of raw sweep-reclaim
signals per level_mode, fade vs follow direction, several horizons.
Not a backtest — just directional expectancy to steer the grid design."""
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

df = pd.read_csv("data/gold_m5.csv", parse_dates=["time"]).set_index("time")
df = df[["open", "high", "low", "close"]].astype(float).iloc[-99999:]
n = len(df)
train = df.iloc[: int(n * 0.6)]

HORIZONS = [3, 6, 12, 24, 48]

for mode in ("pday", "asia", "roll"):
    scfg.params = {"level_mode": mode, "sweep_atr": 0.0, "max_beyond": 3,
                   "max_out_bars": 2, "roll_n": 24}
    ind = m.add_indicators(train, scfg)
    a = ind["atr"]
    sh = ((ind["swept_hi"] > 0) & (ind["close"] < ind["lvl_hi"])
          & ((ind["out_hi"].shift(1) > 0) | (ind["beyond_hi"] > 0))
          & (ind["consec_out_hi"].shift(1) <= 2))
    lo = ((ind["swept_lo"] > 0) & (ind["close"] > ind["lvl_lo"])
          & ((ind["out_lo"].shift(1) > 0) | (ind["beyond_lo"] > 0))
          & (ind["consec_out_lo"].shift(1) <= 2))
    hrs = ind.index.hour
    win = (hrs >= 9) & (hrs < 19) if mode != "asia" else (hrs >= 9) & (hrs < 12)
    sh &= win
    lo &= win
    # entry at next bar open, fade direction PnL before costs, in ATR units
    entry = ind["open"].shift(-1)
    print(f"\n== {mode}: {int(sh.sum())} short sigs, {int(lo.sum())} long sigs (window) ==")
    for hz in HORIZONS:
        exit_px = ind["close"].shift(-hz)
        fade_short = ((entry - exit_px) / a)[sh]          # short: entry - exit
        fade_long = ((exit_px - entry) / a)[lo]
        both = pd.concat([fade_short, fade_long])
        print(f"  hz {hz:3d} bars: fade mean {both.mean():+.4f} ATR "
              f"(short {fade_short.mean():+.4f}, long {fade_long.mean():+.4f}) "
              f"| median {both.median():+.4f} | n {len(both)}")
    # MFE / MAE over 24 bars in fade direction (geometry guidance)
    hz = 24
    fmax = ind["high"].rolling(hz).max().shift(-hz)
    fmin = ind["low"].rolling(hz).min().shift(-hz)
    mfe_s = ((entry - fmin) / a)[sh]; mae_s = ((fmax - entry) / a)[sh]
    mfe_l = ((fmax - entry) / a)[lo]; mae_l = ((entry - fmin) / a)[lo]
    mfe = pd.concat([mfe_s, mfe_l]); mae = pd.concat([mae_s, mae_l])
    print(f"  24-bar MFE median {mfe.median():.3f} ATR / MAE median {mae.median():.3f} ATR "
          f"(spread = {(0.55 / a[sh | lo]).median():.3f} ATR median)")
