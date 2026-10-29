"""Load and validate config.yaml into plain dataclasses."""
from __future__ import annotations

import dataclasses
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

# Supported timeframes in minutes.
TIMEFRAME_MINUTES = {"M1": 1, "M5": 5, "M15": 15, "M30": 30,
                     "H1": 60, "H4": 240, "D1": 1440}


@dataclass
class MT5Cfg:
    login: int = 0
    password: str = ""
    server: str = ""
    terminal_path: str = ""


@dataclass
class CapitalCfg:
    allocated: float = 500.0
    hard_max: float = 500.0


@dataclass
class RiskCfg:
    base_risk_pct: float = 5.0
    max_risk_pct: float = 10.0
    step_up_pct: float = 1.0
    step_up_after_wins: int = 3
    drop_to_base_on_loss: bool = True
    daily_loss_limit: float = 200.0
    close_positions_on_limit: bool = True
    max_open_positions: int = 1
    max_spread_points: float = 60.0
    min_equity_stop: float = 100.0


@dataclass
class StrategyCfg:
    name: str = "ema_cross"      # module in bot/strategies/
    params: dict = field(default_factory=dict)  # strategy-specific knobs
    timeframe: str = "M15"
    ema_fast: int = 9
    ema_slow: int = 21
    ema_trend: int = 200
    rsi_period: int = 14
    rsi_overbought: float = 70.0
    rsi_oversold: float = 30.0
    atr_period: int = 14
    atr_sl_mult: float = 1.5
    atr_tp_mult: float = 2.25
    cooldown_bars: int = 3


@dataclass
class SessionCfg:
    trade_start: str = "07:00"
    trade_end: str = "20:00"
    friday_close: str = "19:00"
    poll_seconds: int = 10


@dataclass
class TrackingCfg:
    db_path: str = "data/trades.db"
    csv_path: str = "data/trades.csv"


@dataclass
class BacktestCfg:
    data_csv: str = ""
    bars: int = 20000
    spread_points: float = 30.0
    commission_per_lot: float = 0.0


@dataclass
class Config:
    mt5: MT5Cfg = field(default_factory=MT5Cfg)
    symbol: str = "XAUUSD"
    magic: int = 620031
    capital: CapitalCfg = field(default_factory=CapitalCfg)
    risk: RiskCfg = field(default_factory=RiskCfg)
    strategy: StrategyCfg = field(default_factory=StrategyCfg)
    session: SessionCfg = field(default_factory=SessionCfg)
    tracking: TrackingCfg = field(default_factory=TrackingCfg)
    backtest: BacktestCfg = field(default_factory=BacktestCfg)


_COERCERS = {"int": int, "float": float, "str": str}


def _coerce(dc_name: str, key: str, ftype: str, value):
    """Coerce a YAML value to the dataclass field's declared type, with a
    readable error naming the offending config key."""
    try:
        if ftype == "bool":
            if isinstance(value, bool):
                return value
            if isinstance(value, str) and value.strip().lower() in ("true", "false"):
                return value.strip().lower() == "true"
            raise ValueError("expected true/false")
        if ftype in _COERCERS:
            return _COERCERS[ftype](value)
        return value
    except (TypeError, ValueError) as e:
        raise ValueError(
            f"config value {dc_name}.{key}: {value!r} is not a valid {ftype} ({e})"
        ) from None


def _merge(dc_cls, raw: dict):
    """Build dataclass from dict, keeping defaults for missing keys and
    ignoring unknown keys (so config typos don't crash, they just warn)."""
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError(
            f"config section '{dc_cls.__name__}' must be a mapping of keys, got {raw!r}"
        )
    names = {f.name: f for f in dataclasses.fields(dc_cls)}
    kwargs = {}
    for key, value in raw.items():
        if key not in names:
            print(f"[config] warning: unknown key '{key}' in {dc_cls.__name__} — ignored")
            continue
        kwargs[key] = _coerce(dc_cls.__name__, key, str(names[key].type), value)
    return dc_cls(**kwargs)


def load_config(path: str | Path = "config.yaml") -> Config:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Copy config.yaml next to main.py and edit it."
        )
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    cfg = Config(
        mt5=_merge(MT5Cfg, raw.get("mt5")),
        symbol=str(raw.get("symbol", "XAUUSD")),
        magic=int(raw.get("magic", 620031)),
        capital=_merge(CapitalCfg, raw.get("capital")),
        risk=_merge(RiskCfg, raw.get("risk")),
        strategy=_merge(StrategyCfg, raw.get("strategy")),
        session=_merge(SessionCfg, raw.get("session")),
        tracking=_merge(TrackingCfg, raw.get("tracking")),
        backtest=_merge(BacktestCfg, raw.get("backtest")),
    )
    _validate(cfg)
    return cfg


def _parse_hhmm(value: str) -> int | None:
    """'HH:MM' -> minutes since midnight, or None if malformed."""
    if not isinstance(value, str) or not re.fullmatch(r"\d{1,2}:\d{2}", value.strip()):
        return None
    h, m = (int(x) for x in value.strip().split(":"))
    if not (0 <= h <= 23 and 0 <= m <= 59):
        return None
    return h * 60 + m


def _validate(cfg: Config) -> None:
    r = cfg.risk
    problems = []
    if not (0 < r.base_risk_pct <= 100):
        problems.append("risk.base_risk_pct must be in (0, 100]")
    if r.max_risk_pct < r.base_risk_pct:
        problems.append("risk.max_risk_pct must be >= base_risk_pct")
    if r.step_up_pct > 0 and r.step_up_after_wins < 1:
        problems.append("risk.step_up_after_wins must be >= 1 "
                        "(set step_up_pct: 0 to disable the step-up)")
    if r.daily_loss_limit <= 0:
        problems.append("risk.daily_loss_limit must be > 0")
    if cfg.capital.allocated <= 0:
        problems.append("capital.allocated must be > 0")
    if cfg.capital.allocated > cfg.capital.hard_max:
        problems.append("capital.allocated must be <= capital.hard_max")
    if cfg.strategy.ema_fast >= cfg.strategy.ema_slow:
        problems.append("strategy.ema_fast must be < ema_slow")
    if cfg.strategy.atr_sl_mult <= 0 or cfg.strategy.atr_tp_mult <= 0:
        problems.append("strategy ATR multipliers must be > 0")

    cfg.strategy.timeframe = cfg.strategy.timeframe.strip().upper()
    if cfg.strategy.timeframe not in TIMEFRAME_MINUTES:
        problems.append(
            f"strategy.timeframe '{cfg.strategy.timeframe}' unknown — "
            f"use one of {list(TIMEFRAME_MINUTES)}"
        )
    if not isinstance(cfg.strategy.params, dict):
        problems.append("strategy.params must be a mapping of knob: value")

    times = {}
    for key in ("trade_start", "trade_end", "friday_close"):
        times[key] = _parse_hhmm(getattr(cfg.session, key))
        if times[key] is None:
            problems.append(f"session.{key} must be 'HH:MM' (24h), "
                            f"got {getattr(cfg.session, key)!r}")
    if None not in times.values():
        if times["trade_start"] >= times["trade_end"]:
            problems.append("session.trade_start must be earlier than trade_end")
        if times["friday_close"] <= times["trade_start"]:
            problems.append("session.friday_close is not after trade_start — "
                            "that disables Fridays entirely; use trade_start < friday_close")
    if cfg.session.poll_seconds < 1:
        problems.append("session.poll_seconds must be >= 1")

    if problems:
        raise ValueError("Bad config:\n  - " + "\n  - ".join(problems))

    if r.daily_loss_limit > 0.2 * cfg.capital.allocated:
        pct = 100.0 * r.daily_loss_limit / cfg.capital.allocated
        print(
            f"[config] WARNING: daily_loss_limit ${r.daily_loss_limit:.0f} is "
            f"{pct:.0f}% of allocated capital. That allows losing the account "
            f"in a few bad days. Consider 5-10% (${0.05*cfg.capital.allocated:.0f}"
            f"-{0.10*cfg.capital.allocated:.0f})."
        )
