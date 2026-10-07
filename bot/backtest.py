"""Event-driven backtester that reuses the EXACT live strategy and risk code.

Fill model (deliberately conservative):
- signal is evaluated on closed bar i -> entry at bar i+1 OPEN
- spread is paid on entry (price worsened by spread_points)
- SL/TP are checked bar-by-bar against high/low
- if a bar touches BOTH SL and TP, the SL is assumed to fill first (worst case)
- the daily loss breaker is checked on every bar close using floating P&L,
  and a tripped breaker closes the position at that bar's close
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config
from .risk import RiskManager, SymbolEconomics
from .strategies import get_strategy
from .strategy import FLAT, LONG
from .tracker import TradeTracker

# XAUUSD economics used when backtesting from CSV (no broker connection):
# 1.0 lot = 100 oz, so a $1.00/oz move = $100 per lot.
DEFAULT_GOLD_ECON = SymbolEconomics(
    tick_size=0.01, tick_value=1.0, volume_min=0.01,
    volume_max=100.0, volume_step=0.01, point=0.01,
)


@dataclass
class OpenPos:
    direction: int
    lots: float
    entry: float
    sl: float
    tp: float
    open_time: pd.Timestamp
    risk_pct: float
    risk_amount: float
    reason: str
    trade_id: int


@dataclass
class BTResult:
    start_equity: float
    end_equity: float
    trades: int = 0
    wins: int = 0
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    max_drawdown: float = 0.0
    breaker_days: int = 0
    equity_curve: list = field(default_factory=list)
    monthly: pd.DataFrame | None = None
    trade_log: list = field(default_factory=list)   # one dict per closed trade

    @property
    def win_rate(self) -> float:
        return 100.0 * self.wins / self.trades if self.trades else 0.0

    @property
    def profit_factor(self) -> float:
        return self.gross_profit / abs(self.gross_loss) if self.gross_loss else float("inf")


def load_csv_bars(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    need = {"time", "open", "high", "low", "close"}
    if not need.issubset(df.columns):
        raise ValueError(f"CSV must have columns {sorted(need)}, got {list(df.columns)}")
    df["time"] = pd.to_datetime(df["time"])
    return df.set_index("time")[["open", "high", "low", "close"]].astype(float)


def run_backtest(
    cfg: Config,
    bars: pd.DataFrame,
    econ: SymbolEconomics = DEFAULT_GOLD_ECON,
    tracker: TradeTracker | None = None,
    verbose: bool = True,
) -> BTResult:
    scfg = cfg.strategy
    risk = RiskManager(cfg.risk, cfg.capital)
    spread = cfg.backtest.spread_points * econ.point
    commission = cfg.backtest.commission_per_lot

    per_unit = econ.money_per_price_unit_per_lot()
    equity = cfg.capital.allocated
    peak = equity
    result = BTResult(start_equity=equity, end_equity=equity)

    # Precompute indicators once over the full frame, then slice per-bar
    # (identical math to live: live recomputes on closed bars each loop, and
    # indicators use only past data so there is no lookahead).
    strat = get_strategy(scfg.name)
    warmup = strat.warmup(scfg)
    ind = strat.add_indicators(bars, scfg)

    pos: OpenPos | None = None
    pending_entry = None      # signal decided on previous close, fills this open
    last_close_idx = -(10**9)
    realized_today = 0.0
    cur_day = None
    breaker_days: set = set()

    # one bar's duration, inferred from the data itself (mode of index diffs)
    bar_step = bars.index.to_series().diff().mode().iloc[0]

    def day_key(ts: pd.Timestamp) -> str:
        return str(ts.date())

    def in_session(ts: pd.Timestamp) -> bool:
        """Live evaluates a bar seconds AFTER it closes, so callers pass the
        bar's CLOSE time here; the end bound is exclusive to match that."""
        if ts.weekday() >= 5:
            return False
        sh, sm = (int(x) for x in cfg.session.trade_start.split(":"))
        eh, em = (int(x) for x in cfg.session.trade_end.split(":"))
        minutes = ts.hour * 60 + ts.minute
        end = eh * 60 + em
        if ts.weekday() == 4:
            fh, fm = (int(x) for x in cfg.session.friday_close.split(":"))
            end = min(end, fh * 60 + fm)
        return sh * 60 + sm <= minutes < end

    def record_close(i: int, ts: pd.Timestamp, price: float, exit_reason: str) -> None:
        nonlocal pos, equity, realized_today, last_close_idx, peak
        assert pos is not None
        move = (price - pos.entry) * pos.direction
        pnl = move * pos.lots * per_unit - commission * pos.lots
        equity += pnl
        realized_today += pnl
        result.trades += 1
        if pnl > 0:
            result.wins += 1
            result.gross_profit += pnl
        else:
            result.gross_loss += pnl
        risk.record_trade_result(pnl)
        peak = max(peak, equity)
        result.max_drawdown = max(result.max_drawdown, peak - equity)
        result.equity_curve.append((ts, equity))
        result.trade_log.append({
            "open_time": pos.open_time, "close_time": ts,
            "direction": "LONG" if pos.direction == LONG else "SHORT",
            "lots": pos.lots, "pnl": round(pnl, 2),
            "risk_pct": pos.risk_pct, "exit": exit_reason,
        })
        if tracker is not None:
            tracker.close_trade(
                pos.trade_id, close_time=str(ts), close_price=round(price, 2),
                pnl=round(pnl, 2), equity_after=round(equity, 2), exit_reason=exit_reason,
            )
        last_close_idx = i
        pos = None

    times = bars.index
    for i in range(warmup, len(bars)):
        ts = times[i]
        bar = bars.iloc[i]
        dk = day_key(ts)
        if dk != cur_day:
            cur_day = dk
            realized_today = 0.0

        # ---- fill a pending entry at this bar's open ----
        # Only fill on the bar immediately after the signal bar (a weekend or
        # data gap between them means live would never have taken this trade)
        # and only inside the session, matching the live order-send moment.
        if (
            pending_entry is not None and pos is None
            and in_session(ts) and (ts - pending_entry[4]) == bar_step
        ):
            sig, lots, risk_pct, risk_amount, _sig_ts = pending_entry
            # longs buy at ask (bid data + spread); shorts sell at bid
            entry = float(bar["open"]) + (spread if sig.direction == LONG else 0.0)
            sl = entry - sig.direction * sig.sl_distance
            tp = entry + sig.direction * sig.tp_distance
            trade_id = 0
            if tracker is not None:
                trade_id = tracker.open_trade(
                    ticket=None, symbol=cfg.symbol,
                    direction="LONG" if sig.direction == LONG else "SHORT",
                    lots=lots, open_time=str(ts), open_price=round(entry, 2),
                    sl=round(sl, 2), tp=round(tp, 2), risk_pct=risk_pct,
                    risk_amount=round(risk_amount, 2), reason=sig.reason, mode="backtest",
                )
            pos = OpenPos(sig.direction, lots, entry, sl, tp, ts, risk_pct, risk_amount,
                          sig.reason, trade_id)
        pending_entry = None

        # ---- manage open position against this bar's range ----
        hi, lo = float(bar["high"]), float(bar["low"])
        if pos is not None:
            bar_open = float(bar["open"])
            # shorts exit by BUYING at ask: worsen exit by spread
            if pos.direction == LONG:
                hit_sl = lo <= pos.sl
                hit_tp = hi >= pos.tp
                # gap through the stop fills at the open, not at the SL price
                sl_px, tp_px = min(bar_open, pos.sl), pos.tp
            else:
                hit_sl = hi + spread >= pos.sl
                hit_tp = lo + spread <= pos.tp
                sl_px, tp_px = max(bar_open + spread, pos.sl), pos.tp
            if hit_sl:                       # worst case first
                record_close(i, ts, sl_px, "sl")
            elif hit_tp:
                record_close(i, ts, tp_px, "tp")

        # ---- daily circuit breaker ----
        # Live checks every few seconds, so approximate intrabar behavior with
        # the bar's ADVERSE extreme, and close at the price where the daily
        # budget was actually breached (clamped into the bar's range).
        worst_floating = 0.0
        if pos is not None:
            if pos.direction == LONG:
                worst_floating = (lo - pos.entry) * pos.lots * per_unit
            else:
                worst_floating = (pos.entry - (hi + spread)) * pos.lots * per_unit
        if risk.check_daily_breaker(dk, realized_today, worst_floating):
            breaker_days.add(dk)
            if pos is not None and cfg.risk.close_positions_on_limit:
                budget = -abs(cfg.risk.daily_loss_limit) - realized_today
                move = budget / (pos.lots * per_unit)
                if pos.direction == LONG:
                    exit_px = min(max(pos.entry + move, lo), hi)
                else:
                    exit_px = min(max(pos.entry - move, lo + spread), hi + spread)
                record_close(i, ts, exit_px, "breaker")
            peak = _mark_drawdown(result, equity, pos, bar, spread, per_unit, peak)
            continue  # no new entries today

        # ---- mark-to-market drawdown (floating losses count) ----
        peak = _mark_drawdown(result, equity, pos, bar, spread, per_unit, peak)

        if equity < cfg.risk.min_equity_stop:
            if verbose:
                print(f"[backtest] equity ${equity:.2f} fell below min_equity_stop — halted at {ts}")
            break

        # ---- evaluate entry signal on this closed bar ----
        # Session is judged at the bar's CLOSE (= when live evaluates it).
        if pos is None and in_session(ts + bar_step) and i - last_close_idx > scfg.cooldown_bars:
            window = ind.iloc[: i + 1]
            sig = strat.evaluate(window, scfg)
            if sig.direction != FLAT:
                # same equity rule as live: size off bot_equity, not raw equity
                size_eq = risk.bot_equity(equity - cfg.capital.allocated)
                decision = risk.size_position(size_eq, sig.sl_distance, econ, open_positions=0)
                if decision.lots > 0:
                    pending_entry = (sig, decision.lots, decision.risk_pct,
                                     decision.risk_amount, ts)

    # close any position left at the end
    if pos is not None:
        last_bar = bars.iloc[-1]
        exit_px = float(last_bar["close"]) + (spread if pos.direction == -1 else 0.0)
        record_close(len(bars) - 1, times[-1], exit_px, "eod")

    result.end_equity = equity
    result.breaker_days = len(breaker_days)
    result.monthly = _monthly_table(result.equity_curve, result.start_equity)
    return result


def _mark_drawdown(result: BTResult, equity: float, pos: "OpenPos | None",
                   bar, spread: float, per_unit: float, peak: float) -> float:
    """Update max_drawdown with FLOATING P&L included; returns the new peak."""
    floating = 0.0
    if pos is not None:
        mark = float(bar["close"]) + (spread if pos.direction != LONG else 0.0)
        floating = (mark - pos.entry) * pos.direction * pos.lots * per_unit
    mtm = equity + floating
    new_peak = max(peak, mtm)
    result.max_drawdown = max(result.max_drawdown, new_peak - mtm)
    return new_peak


def _monthly_table(curve: list, start_equity: float) -> pd.DataFrame:
    if not curve:
        return pd.DataFrame(columns=["month", "pnl", "equity_end"])
    df = pd.DataFrame(curve, columns=["time", "equity"])
    df["month"] = df["time"].dt.to_period("M").astype(str)
    grouped = df.groupby("month")["equity"].last().reset_index(name="equity_end")
    prev = start_equity
    pnls = []
    for _, row in grouped.iterrows():
        pnls.append(row["equity_end"] - prev)
        prev = row["equity_end"]
    grouped["pnl"] = [round(p, 2) for p in pnls]
    return grouped[["month", "pnl", "equity_end"]]


def print_report(result: BTResult, cfg: Config) -> None:
    print()
    print("=" * 58)
    print("  BACKTEST REPORT — reuses the live strategy & risk code")
    print("=" * 58)
    print(f"  start equity      : ${result.start_equity:,.2f}")
    print(f"  end equity        : ${result.end_equity:,.2f}")
    ret = 100.0 * (result.end_equity / result.start_equity - 1.0)
    print(f"  total return      : {ret:+.1f}%")
    print(f"  trades            : {result.trades}  (win rate {result.win_rate:.1f}%)")
    print(f"  profit factor     : {result.profit_factor:.2f}")
    print(f"  gross profit/loss : ${result.gross_profit:,.2f} / ${result.gross_loss:,.2f}")
    print(f"  max drawdown      : ${result.max_drawdown:,.2f}")
    print(f"  daily-breaker hit : {result.breaker_days} day(s) "
          f"(limit ${cfg.risk.daily_loss_limit:.0f})")
    if result.monthly is not None and len(result.monthly):
        print("-" * 58)
        print("  MONTHLY P&L")
        for _, row in result.monthly.iterrows():
            print(f"    {row['month']}   {row['pnl']:>+10.2f}   equity ${row['equity_end']:,.2f}")
    print("=" * 58)
    print("  Past performance in a backtest does NOT guarantee future profit.")
    print("=" * 58)
