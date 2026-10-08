"""Thin wrapper around the official MetaTrader5 Python package.

Everything MT5-specific lives here so the rest of the bot stays testable
without a terminal. Import of MetaTrader5 is lazy: backtests on CSV data
run fine on machines without MT5 installed.

Time note: MT5 hands out BROKER-SERVER timestamps (epoch seconds). We use
that server clock consistently for session windows, day boundaries and the
daily-loss reset, so behavior matches what you see in the MT5 terminal.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import pandas as pd

from .config import MT5Cfg
from .risk import SymbolEconomics

mt5 = None  # populated by connect()

TIMEFRAMES = {
    "M1": 1, "M5": 5, "M15": 15, "M30": 30, "H1": 16385, "H4": 16388, "D1": 16408,
}


class MT5Error(RuntimeError):
    pass


def _tf_const(name: str):
    tf = getattr(mt5, f"TIMEFRAME_{name.upper()}", None)
    if tf is None:
        raise MT5Error(f"Unknown timeframe '{name}'. Use one of {list(TIMEFRAMES)}")
    return tf


def connect(cfg: MT5Cfg) -> None:
    """Attach to a running/installed MT5 terminal. Raises MT5Error on failure."""
    global mt5
    try:
        import MetaTrader5 as _mt5
    except ImportError as e:
        raise MT5Error(
            "MetaTrader5 package not installed. Run: pip install MetaTrader5 "
            "(Windows only, and the MT5 terminal app must be installed)."
        ) from e
    mt5 = _mt5

    kwargs = {}
    if cfg.terminal_path:
        kwargs["path"] = cfg.terminal_path
    if cfg.login:
        kwargs.update(login=int(cfg.login), password=cfg.password, server=cfg.server)

    if not mt5.initialize(**kwargs):
        raise MT5Error(
            f"mt5.initialize() failed: {mt5.last_error()}. Is the MT5 terminal "
            "installed and (ideally) already running and logged in?"
        )

    info = mt5.account_info()
    if info is None:
        raise MT5Error(f"Connected but no account info: {mt5.last_error()}")
    mode = {
        mt5.ACCOUNT_TRADE_MODE_DEMO: "DEMO",
        mt5.ACCOUNT_TRADE_MODE_CONTEST: "CONTEST",
        mt5.ACCOUNT_TRADE_MODE_REAL: "REAL",
    }.get(info.trade_mode, "?")
    print(
        f"[mt5] connected: account {info.login} ({mode}) | balance {info.balance:.2f} "
        f"{info.currency} | equity {info.equity:.2f} | leverage 1:{info.leverage}"
    )
    if mode == "REAL":
        print(
            "[mt5] *** WARNING: this is a REAL-MONEY account. "
            "Test on a DEMO account first. ***"
        )


def shutdown() -> None:
    if mt5 is not None:
        mt5.shutdown()


# ---------------- symbols ----------------

def _symbol_info_or_raise(symbol: str):
    """symbol_info returns None on transient terminal/IPC failures too —
    surface that as MT5Error so the live loop's retry path handles it."""
    info = mt5.symbol_info(symbol)
    if info is None:
        raise MT5Error(f"symbol_info({symbol}) returned None: {mt5.last_error()}")
    return info


def ensure_symbol(symbol: str) -> None:
    info = mt5.symbol_info(symbol)
    if info is None:
        raise MT5Error(
            f"Symbol '{symbol}' not found at this broker. "
            "Run: python main.py find-symbol"
        )
    if not info.visible and not mt5.symbol_select(symbol, True):
        raise MT5Error(f"Could not add '{symbol}' to Market Watch")


def find_gold_symbols() -> list[str]:
    """Return likely gold symbols at this broker."""
    out = []
    for s in mt5.symbols_get() or []:
        name = s.name.upper()
        if "XAU" in name or name.startswith("GOLD"):
            out.append(s.name)
    return out


def economics(symbol: str) -> SymbolEconomics:
    info = _symbol_info_or_raise(symbol)
    return SymbolEconomics(
        tick_size=info.trade_tick_size,
        tick_value=info.trade_tick_value,
        volume_min=info.volume_min,
        volume_max=info.volume_max,
        volume_step=info.volume_step,
        point=info.point,
    )


# ---------------- market data ----------------

def closed_bars(symbol: str, timeframe: str, count: int) -> pd.DataFrame:
    """Last `count` CLOSED bars (skips the currently-forming bar at pos 0)."""
    rates = mt5.copy_rates_from_pos(symbol, _tf_const(timeframe), 1, count)
    if rates is None or len(rates) == 0:
        raise MT5Error(f"No bars for {symbol}: {mt5.last_error()}")
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    return df.set_index("time")[["open", "high", "low", "close", "tick_volume"]]


def current_tick(symbol: str):
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        raise MT5Error(f"No tick for {symbol}: {mt5.last_error()}")
    return tick


def server_now(symbol: str) -> datetime:
    """Broker-server 'now', derived from the freshest tick timestamp."""
    return datetime.fromtimestamp(current_tick(symbol).time, tz=timezone.utc).replace(tzinfo=None)


def spread_points(symbol: str) -> float:
    info = _symbol_info_or_raise(symbol)
    tick = current_tick(symbol)
    if info.point <= 0:
        return 0.0
    return (tick.ask - tick.bid) / info.point


# ---------------- account / positions ----------------

def account_equity() -> float:
    info = mt5.account_info()
    if info is None:
        raise MT5Error(f"account_info failed: {mt5.last_error()}")
    return float(info.equity)


def bot_positions(symbol: str, magic: int) -> list:
    positions = mt5.positions_get(symbol=symbol)
    if positions is None:
        # None means the CALL failed (empty tuple means no positions). Treating
        # a failure as "flat" would blind the breaker and the position limit.
        code, desc = mt5.last_error()
        raise MT5Error(f"positions_get failed ({code}): {desc}")
    return [p for p in positions if p.magic == magic]


def floating_pnl(symbol: str, magic: int) -> float:
    return sum(p.profit + p.swap for p in bot_positions(symbol, magic))


def realized_pnl_since(symbol: str, magic: int, since_server_time: datetime) -> float:
    """Net realized P&L of the bot's deals since `since_server_time` (server clock).

    Counts entry-deal commissions/fees too, and attributes a deal to the bot by
    the POSITION it belongs to, not just the deal's own magic: a position the
    bot opened but the user closed manually gets a closing deal with magic=0,
    and a magic-only filter would hide that loss from the daily breaker and
    permanently overstate the compounding equity.
    """
    # IN deals may predate the window (bot holds one position at a time, so a
    # 30-day lookback for entries is more than enough); filter OUTs precisely.
    frm = since_server_time - timedelta(days=30)
    to = datetime.now(timezone.utc).replace(tzinfo=None) + timedelta(days=2)
    deals = mt5.history_deals_get(frm, to)
    if deals is None:
        raise MT5Error(f"history_deals_get failed: {mt5.last_error()}")
    cutoff = since_server_time.replace(tzinfo=timezone.utc).timestamp()
    bot_position_ids = {
        d.position_id for d in deals
        if d.entry == mt5.DEAL_ENTRY_IN and d.magic == magic and d.symbol == symbol
    }
    total = 0.0
    for d in deals:
        if d.symbol != symbol or d.time < cutoff:
            continue
        if d.magic != magic and d.position_id not in bot_position_ids:
            continue
        if d.entry == mt5.DEAL_ENTRY_IN:
            total += d.commission + getattr(d, "fee", 0.0)  # entry costs
        elif d.entry in (mt5.DEAL_ENTRY_OUT, mt5.DEAL_ENTRY_OUT_BY, mt5.DEAL_ENTRY_INOUT):
            total += d.profit + d.commission + d.swap + getattr(d, "fee", 0.0)
    return total


def cumulative_bot_pnl(symbol: str, magic: int) -> float:
    """All-time realized P&L for this bot (for compounding equity)."""
    return realized_pnl_since(symbol, magic, datetime(2000, 1, 1))


# ---------------- orders ----------------

ORDER_COMMENT = "gb"

@dataclass
class OrderResult:
    ok: bool
    ticket: int
    price: float
    lots: float
    retcode: int
    comment: str


def _try_send(request: dict) -> "OrderResult | None":
    result = mt5.order_send(request)
    if result is None:
        return None
    ok = result.retcode in (mt5.TRADE_RETCODE_DONE, mt5.TRADE_RETCODE_DONE_PARTIAL)
    return OrderResult(
        ok=ok,
        ticket=getattr(result, "order", 0),
        price=getattr(result, "price", 0.0) or 0.0,
        lots=getattr(result, "volume", 0.0) or 0.0,
        retcode=result.retcode,
        comment=getattr(result, "comment", ""),
    )


def market_order(
    symbol: str, direction: int, lots: float, sl: float, tp: float,
    magic: int, comment: str = ORDER_COMMENT,
) -> OrderResult:
    """direction: +1 buy, -1 sell. Tries each filling mode the broker allows."""
    tick = current_tick(symbol)
    is_buy = direction > 0
    price = tick.ask if is_buy else tick.bid
    digits = _symbol_info_or_raise(symbol).digits
    base = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": float(lots),
        "type": mt5.ORDER_TYPE_BUY if is_buy else mt5.ORDER_TYPE_SELL,
        "price": price,
        "sl": round(sl, digits) if sl else 0.0,
        "tp": round(tp, digits) if tp else 0.0,
        "deviation": 30,
        "magic": magic,
        "comment": comment[:31],
        "type_time": mt5.ORDER_TIME_GTC,
    }
    last = None
    for filling in (mt5.ORDER_FILLING_IOC, mt5.ORDER_FILLING_FOK, mt5.ORDER_FILLING_RETURN):
        result = _try_send({**base, "type_filling": filling})
        if result is None:
            continue
        last = result
        if result.ok or result.retcode != mt5.TRADE_RETCODE_INVALID_FILL:
            return result
    if last is not None:
        return last
    raise MT5Error(f"order_send failed outright: {mt5.last_error()}")


def close_position(position, comment: str = ORDER_COMMENT) -> OrderResult:
    """Close an open position with an opposite market deal."""
    tick = current_tick(position.symbol)
    is_long = position.type == mt5.POSITION_TYPE_BUY
    base = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": position.symbol,
        "volume": position.volume,
        "type": mt5.ORDER_TYPE_SELL if is_long else mt5.ORDER_TYPE_BUY,
        "position": position.ticket,
        "price": tick.bid if is_long else tick.ask,
        "deviation": 30,
        "magic": position.magic,
        "comment": comment[:31],
        "type_time": mt5.ORDER_TIME_GTC,
    }
    last = None
    for filling in (mt5.ORDER_FILLING_IOC, mt5.ORDER_FILLING_FOK, mt5.ORDER_FILLING_RETURN):
        result = _try_send({**base, "type_filling": filling})
        if result is None:
            continue
        last = result
        if result.ok or result.retcode != mt5.TRADE_RETCODE_INVALID_FILL:
            return result
    if last is not None:
        return last
    raise MT5Error(f"close order_send failed outright: {mt5.last_error()}")
