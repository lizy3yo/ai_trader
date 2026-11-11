"""Event-driven backtester reusing live strategy and risk logic."""
from __future__ import annotations

import json
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
    fill_idx: int = 0
    flat_dt: pd.Timestamp | None = None
    swap: float = 0.0
    init_sl_dist: float = 0.0
    partial_done: bool = False
    partial_pnl: float = 0.0


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
    swap_total: float = 0.0
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
    """OHLC plus, when the export has them, MT5 tick_volume and spread (points)."""
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    need = {"time", "open", "high", "low", "close"}
    if not need.issubset(df.columns):
        raise ValueError(f"CSV must have columns {sorted(need)}, got {list(df.columns)}")
    df["time"] = pd.to_datetime(df["time"])
    cols = ["open", "high", "low", "close"] + [c for c in ("tick_volume", "spread") if c in df]
    return df.set_index("time")[cols].astype(float)


def load_econ(symbol: str) -> SymbolEconomics:
    """Broker contract specs saved by tools/export_bars.py (falls back to gold)."""
    path = Path(f"data/{symbol.lower()}_econ.json")
    if not path.exists():
        return DEFAULT_GOLD_ECON
    e = json.loads(path.read_text(encoding="utf-8"))
    return SymbolEconomics(
        tick_size=e["tick_size"], tick_value=e["tick_value"], volume_min=e["volume_min"],
        volume_max=e["volume_max"], volume_step=e["volume_step"], point=e["point"],
    )


def _hhmm(s: str) -> tuple[int, int]:
    h, m = (int(x) for x in s.split(":"))
    return h, m


def run_backtest(
    cfg: Config,
    bars: pd.DataFrame,
    econ: SymbolEconomics = DEFAULT_GOLD_ECON,
    tracker: TradeTracker | None = None,
    verbose: bool = True,
    start_at: pd.Timestamp | None = None,
) -> BTResult:
    """start_at: bars before it only warm up indicators (no trades) — lets a
    test window start with fully formed long-lookback filters."""
    scfg = cfg.strategy
    bt = cfg.backtest
    risk = RiskManager(cfg.risk, cfg.capital)
    fixed_spread = bt.spread_points * econ.point
    commission = bt.commission_per_lot

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
    exit_hook = getattr(strat, "exit_signal", None)

    # per-bar spread: never below the configured floor, wider when MT5
    # recorded a wider spread on that bar (news, rollover, weekends)
    o_arr, h_arr = bars["open"].to_numpy(), bars["high"].to_numpy()
    l_arr, c_arr = bars["low"].to_numpy(), bars["close"].to_numpy()
    # the floor is today's spread; a spread is a % of price, so when the floor
    # was measured at spread_ref_price it scales down for cheaper history
    floor = np.full(len(bars), fixed_spread)
    if bt.spread_ref_price > 0:
        floor = fixed_spread * o_arr / bt.spread_ref_price
    if bt.use_bar_spread and "spread" in bars:
        spreads = np.maximum(floor, bars["spread"].to_numpy() * econ.point)
    else:
        spreads = floor

    pos: OpenPos | None = None
    pending_entry = None      # signal decided on previous close, fills this open
    pending_exit = None       # strategy exit decided on previous close
    last_close_idx = -(10**9)
    realized_today = 0.0
    cur_day = None
    breaker_days: set = set()
    flat_hm = _hhmm(cfg.session.flat_at) if cfg.session.flat_at else None

    # one bar's duration, inferred from the data itself (mode of index diffs)
    bar_step = bars.index.to_series().diff().mode().iloc[0]

    def day_key(ts: pd.Timestamp) -> str:
        return str(ts.date())

    def in_session(ts: pd.Timestamp) -> bool:
        """Live evaluates a bar seconds AFTER it closes, so callers pass the
        bar's CLOSE time here; the end bound is exclusive to match that."""
        if ts.weekday() >= 5 and not cfg.session.weekends:
            return False
        sh, sm = _hhmm(cfg.session.trade_start)
        eh, em = _hhmm(cfg.session.trade_end)
        minutes = ts.hour * 60 + ts.minute
        end = eh * 60 + em
        if ts.weekday() == 4 and not cfg.session.weekends:
            fh, fm = _hhmm(cfg.session.friday_close)
            end = min(end, fh * 60 + fm)
        return sh * 60 + sm <= minutes < end

    def swap_nights(prev: pd.Timestamp, now: pd.Timestamp) -> int:
        """Rollovers charged between two server dates (MT5 rolls at 00:00)."""
        n = 0
        d = prev.normalize()
        while d < now.normalize():
            if bt.swap_every_day:
                n += 1
            elif d.weekday() < 5:          # the night ENDING day d
                n += 3 if d.weekday() == bt.swap_triple_weekday else 1
            d += pd.Timedelta(days=1)
        return n

    def record_close(i: int, ts: pd.Timestamp, price: float, exit_reason: str) -> None:
        nonlocal pos, equity, realized_today, last_close_idx, peak
        assert pos is not None
        move = (price - pos.entry) * pos.direction
        pnl = move * pos.lots * per_unit - commission * pos.lots + pos.swap + pos.partial_pnl
        equity += pnl
        realized_today += pnl
        result.trades += 1
        if pnl > 0:
            result.wins += 1
            result.gross_profit += pnl
        else:
            result.gross_loss += pnl
        result.swap_total += pos.swap
        risk.record_trade_result(pnl)
        peak = max(peak, equity)
        result.max_drawdown = max(result.max_drawdown, peak - equity)
        result.equity_curve.append((ts, equity))
        result.trade_log.append({
            "open_time": pos.open_time, "close_time": ts,
            "direction": "LONG" if pos.direction == LONG else "SHORT",
            "lots": pos.lots, "pnl": round(pnl, 2), "swap": round(pos.swap, 2),
            "risk_amount": round(pos.risk_amount, 2),
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
    prev_ts = None
    first = warmup
    if start_at is not None:
        first = max(warmup, int(bars.index.searchsorted(start_at)))
    for i in range(first, len(bars)):
        ts = times[i]
        spread = spreads[i]
        dk = day_key(ts)
        if dk != cur_day:
            cur_day = dk
            realized_today = 0.0
            # overnight financing on a position carried through midnight
            if pos is not None and prev_ts is not None:
                pts = bt.swap_long_points if pos.direction == LONG else bt.swap_short_points
                if bt.swap_ref_price > 0:   # financing is a % of notional: scale to the price then
                    pts *= float(c_arr[i - 1]) / bt.swap_ref_price
                pos.swap += swap_nights(prev_ts, ts) * pts * econ.point * per_unit * pos.lots
        prev_ts = ts
        bar_open = float(o_arr[i])

        # ---- time / strategy exits fill at this bar's open ----
        if pos is not None:
            why = None
            if pending_exit:
                why = pending_exit
            elif pos.flat_dt is not None and ts >= pos.flat_dt:
                why = "flat_at"
            elif scfg.max_hold_bars > 0 and i - pos.fill_idx >= scfg.max_hold_bars:
                why = "time"
            if why:
                px = bar_open + (spread if pos.direction != LONG else 0.0)
                record_close(i, ts, px, why)
        pending_exit = None

        # ---- fill a pending entry at this bar's open ----
        # Fill on the NEXT bar after the signal bar, inside the session. Live
        # only sees a bar as closed once the next one exists, so a Friday-close
        # signal is traded at Monday's open there too (gaps included).
        if (
            pending_entry is not None and pos is None
            and in_session(ts) and pending_entry[4] == i - 1
        ):
            sig, lots, risk_pct, risk_amount, _sig_ts = pending_entry
            # longs buy at ask (bid data + spread); shorts sell at bid
            entry = bar_open + (spread if sig.direction == LONG else 0.0)
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
            flat_dt = None
            if flat_hm is not None:
                flat_dt = ts.normalize() + pd.Timedelta(hours=flat_hm[0], minutes=flat_hm[1])
                if flat_dt <= ts:
                    flat_dt += pd.Timedelta(days=1)
            pos = OpenPos(sig.direction, lots, entry, sl, tp, ts, risk_pct, risk_amount,
                          sig.reason, trade_id, fill_idx=i, flat_dt=flat_dt,
                          init_sl_dist=sig.sl_distance)
        pending_entry = None

        # ---- manage open position against this bar's range ----
        hi, lo = float(h_arr[i]), float(l_arr[i])
        if pos is not None:
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
            partial_due = (scfg.partial_at_r > 0 and not pos.partial_done)
            if hit_sl:                       # worst case first
                record_close(i, ts, sl_px, "sl")
            elif partial_due:
                # scale-out: close part at +partial_at_r R, then (optionally) stop
                # to break-even. Same-bar ambiguity is resolved conservatively:
                # if the bar also traded back to break-even, the rest exits there.
                lvl = pos.entry + pos.direction * scfg.partial_at_r * pos.init_sl_dist
                reached = hi >= lvl if pos.direction == LONG else lo + spread <= lvl
                if reached:
                    step = econ.volume_step or 0.01
                    part = int(pos.lots * scfg.partial_frac / step + 1e-9) * step
                    if part >= econ.volume_min and pos.lots - part >= econ.volume_min - 1e-12:
                        pos.partial_pnl += ((lvl - pos.entry) * pos.direction * part * per_unit
                                            - commission * part)
                        pos.lots = round(pos.lots - part, 8)
                    pos.partial_done = True
                    if scfg.be_after_partial:
                        pos.sl = pos.entry
                        back = lo <= pos.sl if pos.direction == LONG else hi + spread >= pos.sl
                        if back:
                            record_close(i, ts, pos.sl, "be")
                    if pos is not None and hit_tp:
                        record_close(i, ts, tp_px, "tp")
                elif hit_tp:
                    record_close(i, ts, tp_px, "tp")
            elif hit_tp:
                record_close(i, ts, tp_px, "tp")
            # flat_at falling inside this bar: live closes at that moment,
            # approximated by this bar's close (a flat time inside a market
            # closure is still handled at the next open, as live would)
            if pos is not None and pos.flat_dt is not None and ts < pos.flat_dt <= ts + bar_step:
                px = float(c_arr[i]) + (spread if pos.direction != LONG else 0.0)
                record_close(i, ts, px, "flat_at")

        # ---- daily circuit breaker ----
        # Live checks every few seconds, so approximate intrabar behavior with
        # the bar's ADVERSE extreme, and close at the price where the daily
        # budget was actually breached (clamped into the bar's range).
        worst_floating = 0.0
        if pos is not None:
            if pos.direction == LONG:
                worst_floating = (lo - pos.entry) * pos.lots * per_unit + pos.swap
            else:
                worst_floating = (pos.entry - (hi + spread)) * pos.lots * per_unit + pos.swap
        if risk.check_daily_breaker(dk, realized_today, worst_floating):
            breaker_days.add(dk)
            if pos is not None and cfg.risk.close_positions_on_limit:
                budget = -abs(cfg.risk.daily_loss_limit) - realized_today - pos.swap
                move = budget / (pos.lots * per_unit)
                if pos.direction == LONG:
                    exit_px = min(max(pos.entry + move, lo), hi)
                else:
                    exit_px = min(max(pos.entry - move, lo + spread), hi + spread)
                record_close(i, ts, exit_px, "breaker")
            peak = _mark_drawdown(result, equity, pos, float(c_arr[i]), spread, per_unit, peak)
            continue  # no new entries today

        # ---- mark-to-market drawdown (floating losses count) ----
        peak = _mark_drawdown(result, equity, pos, float(c_arr[i]), spread, per_unit, peak)

        if equity < cfg.risk.min_equity_stop:
            if verbose:
                print(f"[backtest] equity ${equity:.2f} fell below min_equity_stop — halted at {ts}")
            break

        # ---- strategy-driven exit, decided on this closed bar ----
        if pos is not None and exit_hook is not None:
            if exit_hook(ind.iloc[: i + 1], scfg, pos.direction, i - pos.fill_idx + 1):
                pending_exit = "signal"

        # ---- evaluate entry signal on this closed bar ----
        # Session is judged at the bar's CLOSE (= when live evaluates it).
        # Live acts when the NEXT bar appears (a Friday-close bar is acted on
        # at Monday's open), so judge the session at that moment.
        decide_ts = times[i + 1] if i + 1 < len(bars) else ts + bar_step
        if pos is None and in_session(decide_ts) and i - last_close_idx > scfg.cooldown_bars:
            window = ind.iloc[: i + 1]
            sig = strat.evaluate(window, scfg)
            if sig.direction != FLAT:
                # same equity rule as live: size off bot_equity, not raw equity
                size_eq = risk.bot_equity(equity - cfg.capital.allocated)
                decision = risk.size_position(size_eq, sig.sl_distance, econ, open_positions=0)
                if decision.lots > 0:
                    pending_entry = (sig, decision.lots, decision.risk_pct,
                                     decision.risk_amount, i)

    # close any position left at the end
    if pos is not None:
        exit_px = float(c_arr[-1]) + (spreads[-1] if pos.direction == -1 else 0.0)
        record_close(len(bars) - 1, times[-1], exit_px, "eod")

    result.end_equity = equity
    result.breaker_days = len(breaker_days)
    result.monthly = _monthly_table(result.equity_curve, result.start_equity)
    return result


def _mark_drawdown(result: BTResult, equity: float, pos: "OpenPos | None",
                   close: float, spread: float, per_unit: float, peak: float) -> float:
    """Update max_drawdown with FLOATING P&L included; returns the new peak."""
    floating = 0.0
    if pos is not None:
        mark = close + (spread if pos.direction != LONG else 0.0)
        floating = (mark - pos.entry) * pos.direction * pos.lots * per_unit + pos.swap
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
