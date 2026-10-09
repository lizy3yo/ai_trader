"""Gold Bot — MT5 algorithmic trading for XAUUSD with strict risk control.

Commands
--------
python main.py status        Show account, bot equity, risk state, open trades
python main.py find-symbol   List the gold symbol names your broker uses
python main.py backtest      Backtest the strategy (MT5 history or CSV)
python main.py live          Run the trading loop (use a DEMO account first!)
python main.py report        Daily + monthly P&L report from the trade journal

DISCLAIMER: This software is provided for educational purposes. Trading gold
CFDs on margin carries a high risk of losing money. Nothing here is financial
advice, and no strategy guarantees profit. Always run on a DEMO account until
YOU have verified the behavior and are comfortable with the risk settings.
"""
from __future__ import annotations

import argparse
import sys
import time
import traceback
from datetime import datetime, timedelta, timezone

from bot import mt5_client as mc
from bot.backtest import DEFAULT_GOLD_ECON, load_csv_bars, print_report, run_backtest
from bot.config import TIMEFRAME_MINUTES as TF_MINUTES
from bot.config import Config, load_config
from bot.risk import RiskManager
from bot.strategies import get_strategy
from bot.strategy import FLAT, LONG
from bot.tracker import TradeTracker


def _server_naive(epoch_seconds: float) -> datetime:
    """Broker epoch seconds -> naive datetime on the broker-server clock."""
    return datetime.fromtimestamp(epoch_seconds, tz=timezone.utc).replace(tzinfo=None)


def _bar_floor(dt: datetime, tf_min: int) -> datetime:
    """Floor a timestamp to its bar's open on the tf_min grid."""
    minutes_into_day = dt.hour * 60 + dt.minute
    floored = (minutes_into_day // tf_min) * tf_min
    return dt.replace(hour=floored // 60, minute=floored % 60, second=0, microsecond=0)


# ======================================================================
# helpers
# ======================================================================

def _parse_hhmm(s: str) -> tuple[int, int]:
    h, m = s.split(":")
    return int(h), int(m)


def in_session(cfg: Config, now: datetime) -> tuple[bool, str]:
    """Is the broker-server clock inside the allowed trading window?"""
    wd = now.weekday()  # Mon=0 ... Sun=6
    if wd >= 5:
        return False, "weekend"
    sh, sm = _parse_hhmm(cfg.session.trade_start)
    eh, em = _parse_hhmm(cfg.session.trade_end)
    start = now.replace(hour=sh, minute=sm, second=0, microsecond=0)
    end = now.replace(hour=eh, minute=em, second=0, microsecond=0)
    if wd == 4:  # Friday: stop earlier
        fh, fm = _parse_hhmm(cfg.session.friday_close)
        end = min(end, now.replace(hour=fh, minute=fm, second=0, microsecond=0))
    if start <= now <= end:
        return True, "open"
    return False, f"outside session {cfg.session.trade_start}-{cfg.session.trade_end}"


def restore_risk_state(risk: RiskManager, tracker: TradeTracker) -> None:
    risk.consecutive_wins = int(tracker.load_state("consecutive_wins", 0))
    risk.current_risk_pct = tracker.load_state(
        "current_risk_pct", risk.cfg.base_risk_pct
    )
    # never restore beyond the configured ceiling / below base
    risk.current_risk_pct = min(max(risk.current_risk_pct, risk.cfg.base_risk_pct),
                                risk.cfg.max_risk_pct)


def persist_risk_state(risk: RiskManager, tracker: TradeTracker) -> None:
    tracker.save_state(
        consecutive_wins=risk.consecutive_wins,
        current_risk_pct=risk.current_risk_pct,
    )


def _exit_reason_from_deal(deal, pnl: float) -> str:
    """Prefer MT5's authoritative deal.reason; fall back to a pnl heuristic."""
    reason = getattr(deal, "reason", None)
    mapping = {
        getattr(mc.mt5, "DEAL_REASON_SL", 4): "sl",
        getattr(mc.mt5, "DEAL_REASON_TP", 5): "tp",
        getattr(mc.mt5, "DEAL_REASON_EXPERT", 3): "bot",
        getattr(mc.mt5, "DEAL_REASON_CLIENT", 0): "manual",
        getattr(mc.mt5, "DEAL_REASON_MOBILE", 1): "manual",
        getattr(mc.mt5, "DEAL_REASON_WEB", 2): "manual",
    }
    if reason in mapping:
        return mapping[reason]
    return "tp" if pnl > 0 else "sl"


def sync_closed_positions(cfg: Config, tracker: TradeTracker, risk: RiskManager,
                          state: dict | None = None, tf_min: int = 0) -> None:
    """Settle journal trades whose MT5 position no longer exists, and journal
    any bot-magic position MT5 holds that we somehow never recorded."""
    positions = mc.bot_positions(cfg.symbol, cfg.magic)
    live_tickets = {p.ticket for p in positions}

    # reconciliation: a filled order that crashed before journaling still gets
    # a row, so its close will feed the risk ladder like any other trade
    for p in positions:
        if tracker.find_open_trade_by_ticket(p.ticket) is None:
            tracker.open_trade(
                ticket=p.ticket, symbol=cfg.symbol,
                direction="LONG" if p.type == mc.mt5.POSITION_TYPE_BUY else "SHORT",
                lots=p.volume, open_time=_server_naive(p.time).isoformat(),
                open_price=p.price_open, sl=p.sl, tp=p.tp,
                risk_pct=risk.current_risk_pct, risk_amount=0.0,
                reason="reconciled from MT5", mode="live",
            )
            print(f"[sync] journaled untracked position #{p.ticket}")

    for row in tracker.open_trades(mode="live"):
        ticket = row["ticket"]
        if not ticket or ticket in live_tickets:
            continue
        try:
            deals = mc.mt5.history_deals_get(position=ticket) or []
            out_deals = [
                d for d in deals
                if d.entry in (mc.mt5.DEAL_ENTRY_OUT, mc.mt5.DEAL_ENTRY_OUT_BY)
            ]
            if not out_deals:
                continue  # not visible in history yet; retry next loop
            pnl = sum(d.profit + d.commission + d.swap + getattr(d, "fee", 0.0)
                      for d in out_deals)
            # entry deals also carry commission on some brokers
            pnl += sum(d.commission + getattr(d, "fee", 0.0)
                       for d in deals if d.entry == mc.mt5.DEAL_ENTRY_IN)
            last = max(out_deals, key=lambda d: d.time)
            exit_reason = _exit_reason_from_deal(last, pnl)
            cum = mc.cumulative_bot_pnl(cfg.symbol, cfg.magic)
            equity_after = risk.bot_equity(cum, mc.account_equity())
            close_dt = _server_naive(last.time)
            tracker.close_trade(
                row["id"],
                close_time=close_dt.isoformat(),
                close_price=last.price, pnl=round(pnl, 2),
                equity_after=round(equity_after, 2), exit_reason=exit_reason,
            )
            risk.record_trade_result(pnl)
            persist_risk_state(risk, tracker)
            # cooldown anchors to the CLOSE bar, mirroring the backtest rule
            if state is not None and tf_min > 0:
                bar_open = _bar_floor(close_dt, tf_min)
                state["cooldown_until"] = bar_open + timedelta(
                    minutes=tf_min * (cfg.strategy.cooldown_bars + 1))
            print(f"[trade] closed ticket {ticket}: {pnl:+.2f} ({exit_reason}) | "
                  f"bot equity ${equity_after:.2f} | next risk {risk.current_risk_pct:.1f}%")
        except mc.MT5Error:
            raise
        except Exception:
            print(f"[sync] failed to settle ticket {ticket} — will retry:\n"
                  f"{traceback.format_exc()}")


# ======================================================================
# commands
# ======================================================================

def cmd_live(cfg: Config) -> None:
    print("DISCLAIMER: educational software. Demo-test first; "
          "no strategy guarantees profit.\n")
    mc.connect(cfg.mt5)
    try:
        mc.ensure_symbol(cfg.symbol)
        econ = mc.economics(cfg.symbol)
        tracker = TradeTracker(cfg.tracking.db_path, cfg.tracking.csv_path)
        risk = RiskManager(cfg.risk, cfg.capital)
        restore_risk_state(risk, tracker)

        tf_min = TF_MINUTES[cfg.strategy.timeframe]
        strat = get_strategy(cfg.strategy.name)
        need = strat.warmup(cfg.strategy) + 50
        state = {"last_bar": None, "cooldown_until": None}

        # a daily-breaker halt must survive restarts for the rest of that day
        halted_day = int(tracker.load_state("halted_day", 0))
        today = mc.server_now(cfg.symbol).date()
        if halted_day == int(today.strftime("%Y%m%d")):
            risk.halted_for_day = str(today)
            print(f"[live] daily loss limit was already hit today ({today}) — "
                  "no trading until the next broker day.")

        print(f"[live] {cfg.symbol} {cfg.strategy.timeframe} strategy={cfg.strategy.name} | "
              f"risk {risk.current_risk_pct:.1f}% (base {cfg.risk.base_risk_pct}%, "
              f"max {cfg.risk.max_risk_pct}%) | daily stop ${cfg.risk.daily_loss_limit:.0f} | "
              f"allocated ${cfg.capital.allocated:.0f}")
        print("[live] Ctrl+C to stop.\n")

        while True:
            try:
                _live_tick(cfg, econ, tracker, risk, tf_min, need, state, strat)
            except mc.MT5Error as e:
                print(f"[live] MT5 hiccup: {e} — retrying in 30s")
                time.sleep(30)
                continue
            except Exception:
                # never let a single bad tick kill the supervisor while a
                # position may be open — log, wait, retry
                print(f"[live] unexpected error — retrying in 30s:\n{traceback.format_exc()}")
                time.sleep(30)
                continue
            time.sleep(cfg.session.poll_seconds)
    except KeyboardInterrupt:
        print("\n[live] stopped by user. Open positions were LEFT RUNNING "
              "(they still have SL/TP attached). Run 'python main.py status' to see them.")
    finally:
        mc.shutdown()


def _live_tick(cfg, econ, tracker, risk, tf_min, need, state, strat) -> None:
    now = mc.server_now(cfg.symbol)
    day = str(now.date())

    # 1) settle anything MT5 closed for us (SL/TP hit while we slept)
    sync_closed_positions(cfg, tracker, risk, state, tf_min)

    # 2) daily circuit breaker — checked EVERY tick, not just on new bars
    day_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    realized_today = mc.realized_pnl_since(cfg.symbol, cfg.magic, day_start)
    floating = mc.floating_pnl(cfg.symbol, cfg.magic)
    already_halted = risk.halted_for_day == day
    if risk.check_daily_breaker(day, realized_today, floating):
        if not already_halted:  # persist the trip so restarts stay halted
            tracker.save_state(halted_day=float(now.strftime("%Y%m%d")))
        positions = mc.bot_positions(cfg.symbol, cfg.magic)
        if positions and cfg.risk.close_positions_on_limit:
            print(f"[risk] DAILY LOSS LIMIT HIT (realized {realized_today:+.2f}, "
                  f"floating {floating:+.2f}). Flattening.")
            for p in positions:
                res = mc.close_position(p)
                print(f"[risk] closed ticket {p.ticket}: retcode {res.retcode} {res.comment}")
            sync_closed_positions(cfg, tracker, risk, state, tf_min)
        return  # no trading for the rest of the day

    # 3) session window
    ok, why = in_session(cfg, now)
    if not ok:
        return

    # 4) act once per NEW closed bar
    bars = mc.closed_bars(cfg.symbol, cfg.strategy.timeframe, need)
    newest = bars.index[-1]
    if state["last_bar"] is None:
        # startup: don't act on a bar that closed before the bot was running
        state["last_bar"] = newest
        return
    if newest <= state["last_bar"]:
        return

    # 5) cooldown after a closed trade / position limit (these consume the bar)
    open_now = mc.bot_positions(cfg.symbol, cfg.magic)
    if state["cooldown_until"] and newest < state["cooldown_until"]:
        state["last_bar"] = newest
        return
    if len(open_now) >= cfg.risk.max_open_positions:
        state["last_bar"] = newest
        return

    # 6) spread guard — spikes are brief, so retry without consuming the bar
    spr = mc.spread_points(cfg.symbol)
    if spr > cfg.risk.max_spread_points:
        print(f"[live] spread {spr:.0f}pts > {cfg.risk.max_spread_points:.0f} — retrying")
        return
    state["last_bar"] = newest

    # 7) signal
    sig = strat.evaluate(strat.add_indicators(bars, cfg.strategy), cfg.strategy)
    if sig.direction == FLAT:
        return

    cum = mc.cumulative_bot_pnl(cfg.symbol, cfg.magic)
    equity = risk.bot_equity(cum, mc.account_equity())
    decision = risk.size_position(equity, sig.sl_distance, econ, len(open_now))
    if decision.lots <= 0:
        print(f"[risk] no trade: {decision.reason}")
        return

    tick = mc.current_tick(cfg.symbol)
    px = tick.ask if sig.direction == LONG else tick.bid
    sl = px - sig.direction * sig.sl_distance
    tp = px + sig.direction * sig.tp_distance
    side = "LONG" if sig.direction == LONG else "SHORT"
    print(f"[signal] {side} {cfg.symbol} @ ~{px:.2f} | SL {sl:.2f} TP {tp:.2f} | "
          f"{decision.lots} lots (risk {decision.risk_pct:.1f}% = ${decision.risk_amount:.2f}) | {sig.reason}")

    pre_tickets = {p.ticket for p in open_now}
    res = mc.market_order(cfg.symbol, sig.direction, decision.lots, sl, tp,
                          cfg.magic)
    if not res.ok:
        print(f"[order] REJECTED retcode={res.retcode} '{res.comment}'")
        return

    # find the NEW position (ticket not present before the order)
    new_pos = None
    for _ in range(5):
        time.sleep(1)
        fresh = [p for p in mc.bot_positions(cfg.symbol, cfg.magic)
                 if p.ticket not in pre_tickets]
        if fresh:
            new_pos = fresh[0]
            break
    if new_pos is None:
        print("[order] filled but position not visible yet — "
              "it will be journaled by the reconciler next tick")
    else:
        tracker.open_trade(
            ticket=new_pos.ticket, symbol=cfg.symbol, direction=side,
            lots=new_pos.volume, open_time=_server_naive(new_pos.time).isoformat(),
            open_price=new_pos.price_open, sl=sl, tp=tp,
            risk_pct=decision.risk_pct, risk_amount=decision.risk_amount,
            reason=sig.reason, mode="live",
        )
        tracker.export_csv()
        print(f"[order] FILLED ticket {new_pos.ticket} @ {new_pos.price_open:.2f}")
    state["cooldown_until"] = now + timedelta(minutes=tf_min * cfg.strategy.cooldown_bars)


def cmd_backtest(cfg: Config) -> None:
    if cfg.backtest.data_csv:
        print(f"[backtest] loading bars from {cfg.backtest.data_csv}")
        bars = load_csv_bars(cfg.backtest.data_csv)
        econ = DEFAULT_GOLD_ECON
    else:
        print(f"[backtest] pulling {cfg.backtest.bars} bars of "
              f"{cfg.symbol} {cfg.strategy.timeframe} from MT5")
        mc.connect(cfg.mt5)
        mc.ensure_symbol(cfg.symbol)
        bars = mc.closed_bars(cfg.symbol, cfg.strategy.timeframe, cfg.backtest.bars)
        econ = mc.economics(cfg.symbol)
        mc.shutdown()
    print(f"[backtest] {len(bars)} bars: {bars.index[0]} -> {bars.index[-1]}")
    tracker = TradeTracker(cfg.tracking.db_path, cfg.tracking.csv_path)
    tracker.conn.execute("DELETE FROM trades WHERE mode='backtest'")
    tracker.conn.commit()
    result = run_backtest(cfg, bars, econ, tracker=tracker)
    print_report(result, cfg)
    tracker.export_csv()
    print(f"[backtest] every simulated trade logged to {cfg.tracking.db_path} "
          f"and {cfg.tracking.csv_path} (mode='backtest')")


def cmd_report(cfg: Config) -> None:
    tracker = TradeTracker(cfg.tracking.db_path, cfg.tracking.csv_path)
    for mode in ("live", "backtest"):
        closed = tracker.all_closed(mode=mode)
        if not closed:
            continue
        print(f"\n================ {mode.upper()} TRADES ================")
        total = tracker.cumulative_pnl(mode=mode)
        wins = sum(1 for r in closed if (r["pnl"] or 0) > 0)
        print(f"closed trades: {len(closed)} | wins: {wins} "
              f"({100.0*wins/len(closed):.1f}%) | total P&L: {total:+.2f}")
        print("\n--- daily ---")
        for row in tracker.daily_summary(mode=mode):
            print(f"  {row['day']}  trades {row['trades']:>3}  wins {row['wins']:>3}  "
                  f"P&L {row['pnl']:>+10.2f}")
        print("\n--- monthly ---")
        for row in tracker.monthly_summary(mode=mode):
            print(f"  {row['month']}  trades {row['trades']:>3}  wins {row['wins']:>3}  "
                  f"P&L {row['pnl']:>+10.2f}  (best {row['best']:+.2f} / worst {row['worst']:+.2f})")
    state_risk = tracker.load_state("current_risk_pct", cfg.risk.base_risk_pct)
    state_wins = int(tracker.load_state("consecutive_wins", 0))
    print(f"\nadaptive risk: {state_risk:.1f}% per trade "
          f"({state_wins} consecutive wins banked)")


def cmd_status(cfg: Config) -> None:
    mc.connect(cfg.mt5)
    try:
        mc.ensure_symbol(cfg.symbol)
        now = mc.server_now(cfg.symbol)
        cum = mc.cumulative_bot_pnl(cfg.symbol, cfg.magic)
        tracker = TradeTracker(cfg.tracking.db_path, cfg.tracking.csv_path)
        risk = RiskManager(cfg.risk, cfg.capital)
        restore_risk_state(risk, tracker)
        equity = risk.bot_equity(cum, mc.account_equity())
        day_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        today = mc.realized_pnl_since(cfg.symbol, cfg.magic, day_start)
        floating = mc.floating_pnl(cfg.symbol, cfg.magic)
        ok, why = in_session(cfg, now)
        print(f"\nserver time     : {now}  (session: {why})")
        print(f"account equity  : ${mc.account_equity():,.2f}")
        print(f"bot equity      : ${equity:,.2f}  (allocated ${cfg.capital.allocated:.0f} "
              f"{cum:+.2f} lifetime P&L)")
        print(f"today           : realized {today:+.2f}, floating {floating:+.2f} "
              f"(daily stop -${cfg.risk.daily_loss_limit:.0f})")
        print(f"risk next trade : {risk.current_risk_pct:.1f}% "
              f"({risk.consecutive_wins} consecutive wins)")
        positions = mc.bot_positions(cfg.symbol, cfg.magic)
        if positions:
            print("open positions  :")
            for p in positions:
                side = "LONG" if p.type == 0 else "SHORT"
                print(f"  #{p.ticket} {side} {p.volume} lots @ {p.price_open:.2f} "
                      f"SL {p.sl:.2f} TP {p.tp:.2f} P&L {p.profit:+.2f}")
        else:
            print("open positions  : none")
        spr = mc.spread_points(cfg.symbol)
        print(f"spread          : {spr:.0f} points (max {cfg.risk.max_spread_points:.0f})")
    finally:
        mc.shutdown()


def cmd_find_symbol(cfg: Config) -> None:
    mc.connect(cfg.mt5)
    try:
        names = mc.find_gold_symbols()
        if not names:
            print("No gold-like symbols found. Ask your broker for the XAUUSD symbol name.")
            return
        print("Gold symbols at this broker:")

        def score(n: str) -> int:
            """Spot gold vs USD: 100oz contract, XAU*USD* or plain GOLD name."""
            info = mc.mt5.symbol_info(n)
            up = n.upper()
            s = 0
            if info and info.trade_contract_size == 100:
                s += 4  # standard spot-gold contract (stocks/minis are not 100)
            if up.startswith("XAU") and "USD" in up:
                s += 3
            if up in ("GOLD", "XAUUSD") or up.startswith("XAUUSD"):
                s += 2
            return s

        for n in names:
            info = mc.mt5.symbol_info(n)
            extra = ""
            if info:
                extra = (f"  (min lot {info.volume_min}, contract {info.trade_contract_size}, "
                         f"spread {info.spread}pts)")
            print(f"  {n}{extra}")
        best = max(names, key=score)
        print(f"\nBest match for spot gold vs USD -> config.yaml symbol: \"{best}\"")
    finally:
        mc.shutdown()


def main() -> None:
    parser = argparse.ArgumentParser(description="Gold Bot — MT5 XAUUSD algo trader")
    parser.add_argument("command",
                        choices=["live", "backtest", "report", "status", "find-symbol"])
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    try:
        cfg = load_config(args.config)
        {
            "live": cmd_live,
            "backtest": cmd_backtest,
            "report": cmd_report,
            "status": cmd_status,
            "find-symbol": cmd_find_symbol,
        }[args.command](cfg)
    except (mc.MT5Error, ValueError, FileNotFoundError) as e:
        print(f"\nERROR: {e}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
