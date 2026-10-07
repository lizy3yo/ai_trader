"""Risk manager: compounding equity, adaptive per-trade risk, daily circuit breaker.

Pure logic (no MT5 imports) so the identical rules run live and in backtests.

Definitions
-----------
bot equity     = capital.allocated + cumulative closed P&L of THIS bot's trades,
                 never more than the real account equity. This is what compounds:
                 risk is a % of bot equity, so winners grow position size and
                 losers shrink it automatically.
daily breaker  = when (today's realized P&L + floating P&L) <= -daily_loss_limit,
                 trading stops until the next broker day.
adaptive risk  = starts at base_risk_pct; after `step_up_after_wins` consecutive
                 wins it steps up by step_up_pct (capped at max_risk_pct);
                 a losing trade drops it back to base (if drop_to_base_on_loss).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .config import CapitalCfg, RiskCfg


@dataclass
class SymbolEconomics:
    """Everything needed to convert a price distance into money per lot."""
    tick_size: float       # smallest price increment (e.g. 0.01 for XAUUSD)
    tick_value: float      # account-currency value of one tick per 1.0 lot
    volume_min: float      # e.g. 0.01
    volume_max: float      # e.g. 100.0
    volume_step: float     # e.g. 0.01
    point: float           # broker 'point' (price units)

    def money_per_price_unit_per_lot(self) -> float:
        """$ P&L for a 1.0 price-unit move with 1.0 lot."""
        if self.tick_size <= 0:
            return 0.0
        return self.tick_value / self.tick_size


@dataclass
class LotDecision:
    lots: float
    risk_amount: float
    risk_pct: float
    reason: str


class RiskManager:
    def __init__(self, risk_cfg: RiskCfg, capital_cfg: CapitalCfg):
        self.cfg = risk_cfg
        self.capital = capital_cfg
        self.consecutive_wins = 0
        self.current_risk_pct = risk_cfg.base_risk_pct
        self.halted_for_day: str | None = None  # date string when breaker tripped

    # ---------------- equity / compounding ----------------

    def bot_equity(self, cumulative_bot_pnl: float, account_equity: float | None = None) -> float:
        """Capital the bot is allowed to treat as its own.

        hard_max caps the allocated BASE only; the full cumulative P&L (losses
        included) always flows through, so losses shrink equity monotonically
        and profits compound on top of the capped base.
        """
        eq = min(self.capital.allocated, self.capital.hard_max) + cumulative_bot_pnl
        if account_equity is not None:
            eq = min(eq, account_equity)   # can't manage money that isn't there
        return max(eq, 0.0)

    # ---------------- adaptive risk ----------------

    def record_trade_result(self, pnl: float) -> None:
        """Call once per CLOSED trade. Adjusts the adaptive risk %."""
        if pnl > 0:
            self.consecutive_wins += 1
            if (
                self.cfg.step_up_pct > 0
                and self.cfg.step_up_after_wins > 0
                and self.consecutive_wins % self.cfg.step_up_after_wins == 0
            ):
                self.current_risk_pct = min(
                    self.current_risk_pct + self.cfg.step_up_pct, self.cfg.max_risk_pct
                )
        else:
            self.consecutive_wins = 0
            if self.cfg.drop_to_base_on_loss:
                self.current_risk_pct = self.cfg.base_risk_pct

    # ---------------- daily circuit breaker ----------------

    def check_daily_breaker(self, day_key: str, realized_today: float, floating: float) -> bool:
        """Returns True when trading must STOP for `day_key` (e.g. '2026-08-30').

        Once tripped, it stays tripped for the whole day even if P&L recovers.
        """
        if self.halted_for_day == day_key:
            return True
        if self.halted_for_day is not None and self.halted_for_day != day_key:
            self.halted_for_day = None  # new day, reset
        if realized_today + floating <= -abs(self.cfg.daily_loss_limit):
            self.halted_for_day = day_key
            return True
        return False

    # ---------------- position sizing ----------------

    def size_position(
        self,
        equity: float,
        sl_distance: float,
        econ: SymbolEconomics,
        open_positions: int = 0,
    ) -> LotDecision:
        """Risk `current_risk_pct` of equity across `sl_distance` of price."""
        if open_positions >= self.cfg.max_open_positions:
            return LotDecision(0.0, 0.0, self.current_risk_pct, "max open positions reached")
        if equity < self.cfg.min_equity_stop:
            return LotDecision(
                0.0, 0.0, self.current_risk_pct,
                f"equity ${equity:.2f} below min_equity_stop ${self.cfg.min_equity_stop:.2f} — bot stopped",
            )
        if sl_distance <= 0:
            return LotDecision(0.0, 0.0, self.current_risk_pct, "invalid stop distance")

        per_unit = econ.money_per_price_unit_per_lot()
        if per_unit <= 0:
            return LotDecision(0.0, 0.0, self.current_risk_pct, "bad symbol economics")

        risk_amount = equity * self.current_risk_pct / 100.0
        raw_lots = risk_amount / (sl_distance * per_unit)

        # round DOWN to the broker's lot step (epsilon guards the float
        # division: 0.29/0.01 is 28.999... in IEEE doubles)
        step = econ.volume_step if econ.volume_step > 0 else 0.01
        lots = math.floor(raw_lots / step + 1e-9) * step
        lots = round(lots, 8)

        if lots < econ.volume_min:
            # Smallest allowed lot risks more than we want. Only take it if the
            # overshoot is modest (<=1.5x intended risk), otherwise skip.
            min_lot_risk = econ.volume_min * sl_distance * per_unit
            if min_lot_risk <= 1.5 * risk_amount:
                return LotDecision(
                    econ.volume_min, min_lot_risk, self.current_risk_pct,
                    f"raised to broker minimum {econ.volume_min} lots "
                    f"(risks ${min_lot_risk:.2f} vs intended ${risk_amount:.2f})",
                )
            return LotDecision(
                0.0, 0.0, self.current_risk_pct,
                f"even min lot {econ.volume_min} risks ${min_lot_risk:.2f} "
                f"(> 1.5x intended ${risk_amount:.2f}) — skipping trade",
            )

        lots = min(lots, econ.volume_max)
        actual_risk = lots * sl_distance * per_unit
        return LotDecision(lots, actual_risk, self.current_risk_pct, "ok")
