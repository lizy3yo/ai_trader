"""Pluggable strategy registry.

A strategy is a module in this package exposing exactly three functions:

    add_indicators(df: pd.DataFrame, scfg: StrategyCfg) -> pd.DataFrame
        df has columns open/high/low/close (index = bar open time, closed bars
        only). Return a COPY with whatever indicator columns you need added.
        Must be causal: row i may only use rows <= i.

    evaluate(ind_window: pd.DataFrame, scfg: StrategyCfg) -> Signal
        ind_window is the add_indicators() output sliced so its LAST row is
        the newest CLOSED bar. Look only at the tail rows (iloc[-1], [-2], ...).
        Return Signal(direction=LONG/SHORT/FLAT, ..., sl_distance, tp_distance).

    warmup(scfg: StrategyCfg) -> int
        Minimum closed bars needed before evaluate() is meaningful.

Strategy-specific tuning knobs come from scfg.params (a free-form dict from
config.yaml); read them with  p = {**DEFAULTS, **scfg.params}.
The shared knobs (timeframe, atr_*, cooldown_bars) stay on scfg itself.

The same module drives the live loop AND the backtester, so anything that
wins research can be traded by setting  strategy.name  in config.yaml.
"""
from __future__ import annotations

import importlib

REQUIRED = ("add_indicators", "evaluate", "warmup")


def get_strategy(name: str):
    if not name.replace("_", "").isalnum():
        raise ValueError(f"bad strategy name {name!r}")
    try:
        mod = importlib.import_module(f"bot.strategies.{name}")
    except ImportError as e:
        raise ValueError(
            f"strategy '{name}' not found in bot/strategies/ ({e})"
        ) from e
    missing = [a for a in REQUIRED if not hasattr(mod, a)]
    if missing:
        raise ValueError(f"strategy '{name}' is missing {missing}")
    return mod
