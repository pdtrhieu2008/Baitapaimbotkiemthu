"""Per-bar position resolution, shared by the backtester and the live loop.

This module exists to remove a specific class of bug. If the backtester decided
when a stop was hit and the live loop decided it separately, the two would drift
apart — and the drift would only ever be discovered with real money on the line.
Both call :func:`resolve_position_on_bar`.

The order of operations inside one bar is fixed and deliberate:

1. Test the bar's range against the stop and the next target **as they stand**.
2. Resolve at most one target per bar (a single OHLC record cannot justify two).
3. Only if the position survives, advance break-even and trailing stops using
   this bar's extremes — i.e. for the *next* bar.

Doing (3) before (1) would let the stop trail past a level the bar had already
traded through, silently converting losses into wins.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from risk.manager import RiskManager
from risk.portfolio import ExitReason, Portfolio, Position, Trade
from strategies.base import SignalSide
from utils.helpers import safe_div

__all__ = ["BarOutcome", "resolve_position_on_bar", "target_fraction_of_remaining"]


@dataclass(slots=True)
class BarOutcome:
    """What happened to a position during one bar.

    Attributes:
        trade: the completed round trip, if the position closed fully.
        exit_reason: why it closed or partially closed.
        partial: ``True`` when only part of the position was taken off.
        stop_action: description of a break-even/trailing move, if any.
    """

    trade: Trade | None = None
    exit_reason: ExitReason | None = None
    partial: bool = False
    stop_action: str | None = None

    @property
    def closed(self) -> bool:
        return self.trade is not None


def target_fraction_of_remaining(position: Position, planned_fraction: float) -> float:
    """Convert a share of the *original* size into a share of what remains.

    ``risk.tp_split`` is expressed against the original position, while
    :meth:`Portfolio.close` takes a fraction of the remainder. Without this
    conversion a configured 50/50 split would close 50% and then 25%.
    """
    share_of_original = planned_fraction * position.quantity
    return min(1.0, max(0.0, safe_div(share_of_original, position.remaining, 1.0)))


def resolve_position_on_bar(
    portfolio: Portfolio,
    risk: RiskManager,
    position: Position,
    *,
    timestamp: datetime,
    high: float,
    low: float,
    atr: float,
    pessimistic: bool = True,
    bars_held: int = 0,
) -> BarOutcome:
    """Apply one bar to one open position.

    Args:
        portfolio: the ledger holding the position.
        risk: risk manager, used for the stop advance and trade accounting.
        position: the open position.
        timestamp: the bar's timestamp, used as the fill time.
        high: bar high.
        low: bar low.
        atr: current ATR, for the trailing stop.
        pessimistic: when the bar contains both the stop and a target, assume the
            stop was hit first. Keep this ``True``.
        bars_held: bars the position has been open, for the trade record.

    Returns:
        A :class:`BarOutcome` describing what happened.
    """
    if not position.is_open:
        return BarOutcome()

    long_side = position.side is SignalSide.LONG
    stop_hit = (low <= position.stop_loss) if long_side else (high >= position.stop_loss)

    target = position.next_target()
    target_hit = False
    if target is not None:
        target_hit = (high >= target.price) if long_side else (low <= target.price)

    # --- 1. stop first, under the pessimistic assumption ---
    if stop_hit and (pessimistic or not target_hit):
        reason = (
            ExitReason.TRAILING_STOP
            if position.trailing_active
            else (ExitReason.BREAK_EVEN if position.breakeven_done else ExitReason.STOP_LOSS)
        )
        trade = portfolio.close(
            position, position.stop_loss, reason, timestamp=timestamp, bars_held=bars_held
        )
        if trade is not None:
            risk.on_trade_closed(trade)
        return BarOutcome(trade=trade, exit_reason=reason)

    # --- 2. at most one target per bar ---
    if target_hit and target is not None:
        position.targets_hit += 1
        is_last = position.targets_hit >= len(position.take_profits)
        fraction = 1.0 if is_last else target_fraction_of_remaining(position, target.fraction)
        trade = portfolio.close(
            position,
            target.price,
            ExitReason.TAKE_PROFIT,
            timestamp=timestamp,
            fraction=fraction,
            bars_held=bars_held,
        )
        if trade is not None:
            risk.on_trade_closed(trade)
            return BarOutcome(trade=trade, exit_reason=ExitReason.TAKE_PROFIT)

        # Partially filled: in optimistic mode the same bar may still stop out
        # the remainder, which is the only sequence a single OHLC record allows.
        if stop_hit and not pessimistic:
            trade = portfolio.close(
                position,
                position.stop_loss,
                ExitReason.STOP_LOSS,
                timestamp=timestamp,
                bars_held=bars_held,
            )
            if trade is not None:
                risk.on_trade_closed(trade)
            return BarOutcome(trade=trade, exit_reason=ExitReason.STOP_LOSS, partial=True)

        action = risk.manage(position, high=high, low=low, atr=atr)
        return BarOutcome(exit_reason=ExitReason.TAKE_PROFIT, partial=True, stop_action=action)

    # --- 3. survived: advance the stops for the next bar ---
    return BarOutcome(stop_action=risk.manage(position, high=high, low=low, atr=atr))
