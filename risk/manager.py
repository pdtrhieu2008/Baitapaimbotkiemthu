"""Position sizing, circuit breakers and in-trade stop management.

This is the module that decides how much money is exposed, and it is
intentionally the most conservative code in the project. Its priorities, in
order:

1. never risk more than the configured fraction of equity on one trade;
2. stop trading when the day, the streak or the drawdown says so;
3. only then, take the trade.

Sizing is derived from the **stop distance**, not from a fixed notional. That is
what makes ``risk_per_trade_pct`` mean what it says: a wider stop produces a
smaller position, so the cash lost when the stop is hit is the same either way.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from enum import Enum

import numpy as np

from config.settings import RiskConfig
from risk.portfolio import ExitReason, Portfolio, Position, Trade
from strategies.base import Signal, SignalSide
from utils.helpers import safe_div, utc_now
from utils.logger import get_logger

__all__ = ["HaltReason", "RiskDecision", "RiskManager", "RiskState"]

_log = get_logger("risk.manager")


class HaltReason(str, Enum):
    """Why trading is suspended."""

    NONE = "none"
    DAILY_LOSS = "daily_loss_limit"
    MAX_DRAWDOWN = "max_drawdown"
    CONSECUTIVE_LOSSES = "consecutive_losses"

    @property
    def is_permanent(self) -> bool:
        """Whether the halt survives the day boundary.

        A daily-loss halt lifts at the next UTC day; a max-drawdown halt does
        not, because it means the strategy's assumptions are no longer holding
        and that needs a human, not a midnight timer.
        """
        return self is HaltReason.MAX_DRAWDOWN


@dataclass(slots=True)
class RiskDecision:
    """Verdict on one signal.

    Attributes:
        approved: whether the trade may be taken.
        quantity: size in base units (``0`` when rejected).
        risk_amount: cash at risk if the stop is hit.
        risk_pct: that amount as a percentage of current equity.
        notional: position value at entry.
        reason: why it was rejected, or a note when it was resized.
    """

    approved: bool
    quantity: float = 0.0
    risk_amount: float = 0.0
    risk_pct: float = 0.0
    notional: float = 0.0
    reason: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "approved": self.approved,
            "quantity": self.quantity,
            "risk_amount": round(self.risk_amount, 4),
            "risk_pct": round(self.risk_pct, 4),
            "notional": round(self.notional, 2),
            "reason": self.reason,
        }


@dataclass(slots=True)
class RiskState:
    """Mutable risk counters, persisted across restarts.

    Attributes:
        peak_equity: high-water mark, for the drawdown brake.
        daily_start_equity: equity at the start of the current UTC day.
        current_day: the day those counters belong to.
        consecutive_losses: losing trades since the last winner.
        halt: active halt, if any.
    """

    peak_equity: float
    daily_start_equity: float
    current_day: date
    consecutive_losses: int = 0
    halt: HaltReason = HaltReason.NONE
    trades_today: int = 0
    realised_today: float = 0.0

    def to_dict(self) -> dict[str, object]:
        return {
            "peak_equity": self.peak_equity,
            "daily_start_equity": self.daily_start_equity,
            "current_day": self.current_day.isoformat(),
            "consecutive_losses": self.consecutive_losses,
            "halt": self.halt.value,
            "trades_today": self.trades_today,
            "realised_today": round(self.realised_today, 4),
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> RiskState:
        """Rebuild from persisted JSON, tolerating a missing or stale file."""
        return cls(
            peak_equity=float(data.get("peak_equity", 0.0) or 0.0),
            daily_start_equity=float(data.get("daily_start_equity", 0.0) or 0.0),
            current_day=date.fromisoformat(str(data.get("current_day", date.today().isoformat()))),
            consecutive_losses=int(data.get("consecutive_losses", 0) or 0),
            halt=HaltReason(str(data.get("halt", "none"))),
            trades_today=int(data.get("trades_today", 0) or 0),
            realised_today=float(data.get("realised_today", 0.0) or 0.0),
        )


class RiskManager:
    """Approve, size and manage trades.

    Args:
        cfg: the ``risk`` configuration section.
        portfolio: the ledger this manager governs.
    """

    def __init__(self, cfg: RiskConfig, portfolio: Portfolio) -> None:
        self.cfg = cfg
        self.portfolio = portfolio
        equity = portfolio.equity()
        self.state = RiskState(
            peak_equity=equity,
            daily_start_equity=equity,
            current_day=utc_now().date(),
        )

    # -- day boundary -------------------------------------------------------
    def roll_day(self, moment: datetime) -> None:
        """Reset the daily counters when the UTC day changes.

        Called on every processed bar by both the live loop and the backtester,
        so a daily-loss halt lifts at exactly the same point in both.
        """
        today = moment.date()
        if today == self.state.current_day:
            return
        equity = self.portfolio.equity()
        self.state.current_day = today
        self.state.daily_start_equity = equity
        self.state.trades_today = 0
        self.state.realised_today = 0.0
        if self.state.halt is not HaltReason.NONE and not self.state.halt.is_permanent:
            _log.info("daily halt (%s) lifted at the %s boundary", self.state.halt.value, today)
            self.state.halt = HaltReason.NONE

    # -- breakers -----------------------------------------------------------
    @property
    def is_halted(self) -> bool:
        return self.state.halt is not HaltReason.NONE

    def daily_loss_pct(self, marks: dict[str, float] | None = None) -> float:
        """Loss so far today, as a positive percentage of the day's opening equity.

        Args:
            marks: latest prices. Pass them whenever they are available — an
                open, deeply underwater position contributes nothing to realised
                PnL, so omitting the marks lets a losing day slip past the limit.
        """
        equity = self.portfolio.equity(marks)
        change = equity - self.state.daily_start_equity
        return -100.0 * safe_div(change, self.state.daily_start_equity) if change < 0 else 0.0

    def drawdown_pct(self, marks: dict[str, float] | None = None) -> float:
        """Current drawdown from the high-water mark, as a positive percentage.

        Args:
            marks: latest prices, for the same reason as
                :meth:`daily_loss_pct`.
        """
        equity = self.portfolio.equity(marks)
        return max(0.0, 100.0 * safe_div(self.state.peak_equity - equity, self.state.peak_equity))

    def update_breakers(self, marks: dict[str, float] | None = None) -> HaltReason:
        """Refresh the high-water mark and evaluate every circuit breaker.

        Args:
            marks: latest prices, so open positions are marked to market. Using
                realised PnL alone would let an open loser grow past the limit
                unnoticed.

        Returns:
            The active :class:`HaltReason` (possibly ``NONE``).
        """
        equity = self.portfolio.equity(marks)
        self.state.peak_equity = max(self.state.peak_equity, equity)

        if self.state.halt.is_permanent:
            return self.state.halt

        # Both brakes must see the same marked-to-market equity used above,
        # otherwise an open loser is invisible to them.
        drawdown = self.drawdown_pct(marks)
        if drawdown >= self.cfg.max_drawdown_pct:
            self._halt(HaltReason.MAX_DRAWDOWN, f"drawdown {drawdown:.2f}%")
            return self.state.halt

        daily = self.daily_loss_pct(marks)
        if daily >= self.cfg.daily_loss_limit_pct:
            self._halt(HaltReason.DAILY_LOSS, f"daily loss {daily:.2f}%")
            return self.state.halt

        if self.state.consecutive_losses >= self.cfg.max_consecutive_losses:
            self._halt(
                HaltReason.CONSECUTIVE_LOSSES,
                f"{self.state.consecutive_losses} losses in a row",
            )
            return self.state.halt

        return HaltReason.NONE

    def _halt(self, reason: HaltReason, detail: str) -> None:
        if self.state.halt is reason:
            return
        self.state.halt = reason
        _log.warning(
            "TRADING HALTED (%s): %s. %s",
            reason.value,
            detail,
            "Manual review required." if reason.is_permanent else "Resumes at the next UTC day.",
        )

    # -- sizing -------------------------------------------------------------
    def approve(self, signal: Signal, *, marks: dict[str, float] | None = None) -> RiskDecision:
        """Decide whether and how large to take a signal.

        Args:
            signal: a signal that already passed every strategy gate.
            marks: latest prices, for equity valuation.

        Returns:
            A :class:`RiskDecision`.
        """
        cfg = self.cfg

        halt = self.update_breakers(marks)
        if halt is not HaltReason.NONE:
            return RiskDecision(False, reason=f"trading halted: {halt.value}")

        if self.portfolio.open_count >= cfg.max_open_positions:
            return RiskDecision(
                False,
                reason=f"already at max_open_positions ({cfg.max_open_positions})",
            )

        existing = self.portfolio.open_for(signal.symbol)
        if len(existing) >= cfg.max_positions_per_symbol:
            return RiskDecision(
                False,
                reason=f"already holding {len(existing)} position(s) in {signal.symbol}",
            )
        # Never hold both directions in the same instrument: the two positions
        # cancel economically while paying two sets of fees.
        if any(position.side is not signal.side for position in existing):
            return RiskDecision(False, reason="an opposite position is open in this symbol")

        same_direction = sum(
            1 for position in self.portfolio.positions.values() if position.side is signal.side
        )
        if same_direction >= cfg.max_correlated_positions:
            return RiskDecision(
                False,
                reason=(
                    f"{same_direction} positions already point {signal.side.label}; "
                    f"crypto pairs are highly correlated, so that is one bet, not "
                    f"{same_direction} (limit {cfg.max_correlated_positions})"
                ),
            )

        equity = self.portfolio.equity(marks)
        if equity <= 0:
            return RiskDecision(False, reason="equity is not positive")

        risk_per_unit = signal.risk_per_unit
        if not np.isfinite(risk_per_unit) or risk_per_unit <= 0:
            return RiskDecision(False, reason="stop distance is zero or undefined")

        risk_amount = equity * cfg.risk_per_trade_pct / 100.0
        quantity = risk_amount / risk_per_unit
        notional = quantity * signal.entry
        note = ""

        # Leverage cap. Without it, a very tight stop would produce a notional
        # many times the account, which no venue would accept and which would
        # make a gap through the stop catastrophic rather than merely painful.
        max_notional = equity * cfg.leverage
        if notional > max_notional:
            quantity = max_notional / signal.entry
            notional = max_notional
            risk_amount = quantity * risk_per_unit
            note = (
                f"size capped by leverage {cfg.leverage:g}x: effective risk "
                f"{100.0 * risk_amount / equity:.3f}% instead of {cfg.risk_per_trade_pct}%"
            )

        if quantity <= 0 or not np.isfinite(quantity):
            return RiskDecision(False, reason="computed quantity is not positive")

        # Reject dust: a position whose expected edge is smaller than its fees.
        round_trip_cost = notional * cfg.round_trip_cost_pct / 100.0
        if round_trip_cost >= risk_amount * 0.5:
            return RiskDecision(
                False,
                reason=(
                    f"round-trip cost {round_trip_cost:.4f} is more than half the "
                    f"{risk_amount:.4f} at risk; the stop is too tight to be worth trading"
                ),
            )

        return RiskDecision(
            approved=True,
            quantity=quantity,
            risk_amount=risk_amount,
            risk_pct=100.0 * risk_amount / equity,
            notional=notional,
            reason=note,
        )

    # -- in-trade management ------------------------------------------------
    def manage(self, position: Position, *, high: float, low: float, atr: float) -> str | None:
        """Advance break-even and trailing stops after a bar has been resolved.

        **Call order matters.** The caller must first test this bar's OHLC
        against the *existing* stop and targets, and only then call this method
        with the same bar's extremes. Trailing on the bar that is simultaneously
        being tested for a stop-out would let the stop escape a hit it should
        have taken.

        Args:
            position: the open position.
            high: bar high.
            low: bar low.
            atr: current ATR.

        Returns:
            A description of the stop change, or ``None`` if nothing moved.
        """
        position.observe(high, low)
        excursion = position.excursion_r
        cfg = self.cfg
        long_side = position.side is SignalSide.LONG
        old_stop = position.stop_loss
        action: str | None = None

        if (
            not position.breakeven_done
            and cfg.breakeven_at_rr > 0
            and excursion >= cfg.breakeven_at_rr
        ):
            # Break-even is set just beyond entry so the exit still covers the
            # round trip instead of scratching at a small loss.
            cushion = position.entry_price * cfg.round_trip_cost_pct / 100.0
            candidate = position.entry_price + position.side.value * cushion
            if (candidate > position.stop_loss) if long_side else (candidate < position.stop_loss):
                position.stop_loss = candidate
                position.breakeven_done = True
                action = f"stop to break-even ({candidate:.6g}) at {excursion:.2f}R"

        if (
            cfg.trailing_enabled
            and np.isfinite(atr)
            and atr > 0
            and excursion >= cfg.trailing_activate_rr
        ):
            candidate = position.best_price - position.side.value * cfg.trailing_atr_multiplier * atr
            improves = (candidate > position.stop_loss) if long_side else (candidate < position.stop_loss)
            if improves:
                position.stop_loss = candidate
                position.trailing_active = True
                action = (
                    f"trailing stop to {candidate:.6g} "
                    f"({cfg.trailing_atr_multiplier:g} ATR behind {position.best_price:.6g})"
                )

        if action is not None:
            _log.debug(
                "position %d %s: %s (was %.6g)", position.id, position.symbol, action, old_stop
            )
        return action

    # -- accounting ---------------------------------------------------------
    def on_trade_closed(self, trade: Trade) -> None:
        """Update streak and daily counters after a round trip."""
        if trade.is_win:
            self.state.consecutive_losses = 0
        else:
            self.state.consecutive_losses += 1
        self.state.trades_today += 1
        self.state.realised_today += trade.pnl
        self.update_breakers()

    def force_close_all(self, marks: dict[str, float], timestamp: datetime) -> list[Trade]:
        """Flatten every open position — used when a permanent halt fires.

        Args:
            marks: latest price per symbol.
            timestamp: exit time.

        Returns:
            The resulting trades.
        """
        closed: list[Trade] = []
        for position in list(self.portfolio.positions.values()):
            price = marks.get(position.symbol)
            if price is None:
                _log.error(
                    "cannot flatten position %d in %s: no current price available",
                    position.id, position.symbol,
                )
                continue
            trade = self.portfolio.close(
                position, price, ExitReason.RISK_HALT, timestamp=timestamp
            )
            if trade is not None:
                closed.append(trade)
                self.on_trade_closed(trade)
        return closed

    def snapshot(self) -> dict[str, object]:
        """Current risk picture, for logging and the daily summary."""
        return {
            "equity": round(self.portfolio.equity(), 4),
            "peak_equity": round(self.state.peak_equity, 4),
            "drawdown_pct": round(self.drawdown_pct(), 3),
            "daily_loss_pct": round(self.daily_loss_pct(), 3),
            "open_positions": self.portfolio.open_count,
            "halt": self.state.halt.value,
            "consecutive_losses": self.state.consecutive_losses,
            "trades_today": self.state.trades_today,
        }
