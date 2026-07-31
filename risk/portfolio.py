"""Position and trade ledger, shared by paper trading and the backtester.

One ledger implementation for both paths is a deliberate constraint: if the
backtest used its own accounting, its equity curve would stop being evidence
about the live bot. Fees and slippage are applied on every fill here, so a
strategy that only works gross-of-costs cannot look profitable.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

import numpy as np

from strategies.base import Signal, SignalSide, TakeProfit
from utils.helpers import safe_div, utc_now

__all__ = ["ExitReason", "Fill", "Portfolio", "Position", "Trade"]

_position_ids = itertools.count(1)


class ExitReason(str, Enum):
    """Why a position (or part of it) was closed."""

    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"
    TRAILING_STOP = "trailing_stop"
    BREAK_EVEN = "break_even"
    SIGNAL_FLIP = "signal_flip"
    RISK_HALT = "risk_halt"
    END_OF_DATA = "end_of_data"
    MANUAL = "manual"


@dataclass(frozen=True, slots=True)
class Fill:
    """One execution against a position."""

    timestamp: datetime
    price: float
    quantity: float
    fee: float
    reason: ExitReason | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "price": self.price,
            "quantity": self.quantity,
            "fee": self.fee,
            "reason": self.reason.value if self.reason else None,
        }


@dataclass(slots=True)
class Position:
    """An open position with scaled exits and a movable stop.

    Attributes:
        initial_risk: cash at risk when the position was opened — the R unit
            every later result is measured against. Frozen at entry on purpose:
            if the stop moves to break-even, a subsequent win is still measured
            in the *original* R, which is the only way R multiples stay
            comparable across trades.
    """

    id: int
    symbol: str
    side: SignalSide
    entry_price: float
    quantity: float
    stop_loss: float
    take_profits: tuple[TakeProfit, ...]
    opened_at: datetime
    atr: float
    initial_stop: float
    initial_risk: float
    signal_score: float = 0.0
    signal_strength: str = ""
    trigger: str = ""
    timeframe: str = ""
    remaining: float = 0.0
    realised_pnl: float = 0.0
    fees_paid: float = 0.0
    fills: list[Fill] = field(default_factory=list)
    targets_hit: int = 0
    breakeven_done: bool = False
    trailing_active: bool = False
    best_price: float = float("nan")
    worst_price: float = float("nan")

    def __post_init__(self) -> None:
        if self.remaining == 0.0:
            self.remaining = self.quantity
        if not np.isfinite(self.best_price):
            self.best_price = self.entry_price
        if not np.isfinite(self.worst_price):
            self.worst_price = self.entry_price

    # -- state ---------------------------------------------------------------
    @property
    def is_open(self) -> bool:
        return self.remaining > 1e-12

    @property
    def risk_per_unit(self) -> float:
        """Original stop distance per unit."""
        return abs(self.entry_price - self.initial_stop)

    def unrealised_pnl(self, price: float) -> float:
        """Mark-to-market PnL of the remaining quantity, before exit costs."""
        return (price - self.entry_price) * self.side.value * self.remaining

    def r_multiple(self, price: float) -> float:
        """Open profit in R, using the original risk unit."""
        return safe_div((price - self.entry_price) * self.side.value, self.risk_per_unit)

    @property
    def excursion_r(self) -> float:
        """Best R reached so far (maximum favourable excursion)."""
        return self.r_multiple(self.best_price)

    @property
    def drawdown_r(self) -> float:
        """Worst R reached so far, as a negative number."""
        return self.r_multiple(self.worst_price)

    def observe(self, high: float, low: float) -> None:
        """Update the favourable/adverse extremes from a bar's range."""
        if self.side is SignalSide.LONG:
            self.best_price = max(self.best_price, high)
            self.worst_price = min(self.worst_price, low)
        else:
            self.best_price = min(self.best_price, low)
            self.worst_price = max(self.worst_price, high)

    def next_target(self) -> TakeProfit | None:
        """The nearest target not yet taken."""
        if self.targets_hit >= len(self.take_profits):
            return None
        return self.take_profits[self.targets_hit]

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "symbol": self.symbol,
            "side": self.side.name,
            "entry": self.entry_price,
            "quantity": self.quantity,
            "remaining": self.remaining,
            "stop_loss": self.stop_loss,
            "targets_hit": self.targets_hit,
            "realised_pnl": round(self.realised_pnl, 4),
            "opened_at": self.opened_at.isoformat(),
            "excursion_r": round(self.excursion_r, 2),
        }

    # -- persistence --------------------------------------------------------
    def to_state(self) -> dict[str, object]:
        """Full serialisable state, for surviving a restart.

        Distinct from :meth:`to_dict`, which is a lossy summary for logs. Every
        field needed to keep managing the position — the moved stop, which
        targets are already taken, the excursion extremes that drive trailing —
        is included, because a restart that forgot them would mismanage the exit.
        """
        return {
            "id": self.id,
            "symbol": self.symbol,
            "side": self.side.value,
            "entry_price": self.entry_price,
            "quantity": self.quantity,
            "stop_loss": self.stop_loss,
            "take_profits": [
                {"price": tp.price, "fraction": tp.fraction, "rr": tp.rr}
                for tp in self.take_profits
            ],
            "opened_at": self.opened_at.isoformat(),
            "atr": self.atr,
            "initial_stop": self.initial_stop,
            "initial_risk": self.initial_risk,
            "signal_score": self.signal_score,
            "signal_strength": self.signal_strength,
            "trigger": self.trigger,
            "timeframe": self.timeframe,
            "remaining": self.remaining,
            "realised_pnl": self.realised_pnl,
            "fees_paid": self.fees_paid,
            "targets_hit": self.targets_hit,
            "breakeven_done": self.breakeven_done,
            "trailing_active": self.trailing_active,
            "best_price": self.best_price,
            "worst_price": self.worst_price,
        }

    @classmethod
    def from_state(cls, data: dict[str, object]) -> Position:
        """Rebuild a position from :meth:`to_state` output."""
        targets = tuple(
            TakeProfit(price=float(t["price"]), fraction=float(t["fraction"]), rr=float(t["rr"]))
            for t in data.get("take_profits", [])  # type: ignore[union-attr]
        )
        return cls(
            id=int(data["id"]),  # type: ignore[arg-type]
            symbol=str(data["symbol"]),
            side=SignalSide(int(data["side"])),  # type: ignore[arg-type]
            entry_price=float(data["entry_price"]),  # type: ignore[arg-type]
            quantity=float(data["quantity"]),  # type: ignore[arg-type]
            stop_loss=float(data["stop_loss"]),  # type: ignore[arg-type]
            take_profits=targets,
            opened_at=datetime.fromisoformat(str(data["opened_at"])),
            atr=float(data["atr"]),  # type: ignore[arg-type]
            initial_stop=float(data["initial_stop"]),  # type: ignore[arg-type]
            initial_risk=float(data["initial_risk"]),  # type: ignore[arg-type]
            signal_score=float(data.get("signal_score", 0.0)),  # type: ignore[arg-type]
            signal_strength=str(data.get("signal_strength", "")),
            trigger=str(data.get("trigger", "")),
            timeframe=str(data.get("timeframe", "")),
            remaining=float(data.get("remaining", 0.0)),  # type: ignore[arg-type]
            realised_pnl=float(data.get("realised_pnl", 0.0)),  # type: ignore[arg-type]
            fees_paid=float(data.get("fees_paid", 0.0)),  # type: ignore[arg-type]
            targets_hit=int(data.get("targets_hit", 0)),  # type: ignore[arg-type]
            breakeven_done=bool(data.get("breakeven_done", False)),
            trailing_active=bool(data.get("trailing_active", False)),
            best_price=float(data.get("best_price", float("nan"))),  # type: ignore[arg-type]
            worst_price=float(data.get("worst_price", float("nan"))),  # type: ignore[arg-type]
        )


@dataclass(slots=True)
class Trade:
    """A completed round trip."""

    position_id: int
    symbol: str
    side: SignalSide
    timeframe: str
    entry_price: float
    exit_price: float
    quantity: float
    opened_at: datetime
    closed_at: datetime
    pnl: float
    fees: float
    exit_reason: ExitReason
    initial_risk: float
    r_multiple: float
    mae_r: float
    mfe_r: float
    bars_held: int = 0
    signal_score: float = 0.0
    signal_strength: str = ""
    trigger: str = ""

    @property
    def is_win(self) -> bool:
        return self.pnl > 0

    @property
    def duration_hours(self) -> float:
        return (self.closed_at - self.opened_at).total_seconds() / 3600.0

    @property
    def return_pct(self) -> float:
        """Return on the notional actually committed."""
        notional = self.entry_price * self.quantity
        return 100.0 * safe_div(self.pnl, notional)

    def to_dict(self) -> dict[str, object]:
        return {
            "position_id": self.position_id,
            "symbol": self.symbol,
            "side": self.side.name,
            "timeframe": self.timeframe,
            "entry": self.entry_price,
            "exit": self.exit_price,
            "quantity": self.quantity,
            "opened_at": self.opened_at.isoformat(),
            "closed_at": self.closed_at.isoformat(),
            "pnl": round(self.pnl, 4),
            "fees": round(self.fees, 4),
            "r_multiple": round(self.r_multiple, 3),
            "mae_r": round(self.mae_r, 2),
            "mfe_r": round(self.mfe_r, 2),
            "exit_reason": self.exit_reason.value,
            "return_pct": round(self.return_pct, 3),
            "bars_held": self.bars_held,
            "score": self.signal_score,
            "trigger": self.trigger,
        }


class Portfolio:
    """Cash, open positions, closed trades and the equity curve.

    Args:
        initial_equity: starting cash.
        fee_pct: taker fee per side, in percent.
        slippage_pct: assumed adverse slippage per side, in percent.
    """

    def __init__(
        self, initial_equity: float, fee_pct: float = 0.0, slippage_pct: float = 0.0
    ) -> None:
        if initial_equity <= 0:
            raise ValueError("initial_equity must be > 0")
        self.initial_equity = float(initial_equity)
        self.fee_pct = float(fee_pct)
        self.slippage_pct = float(slippage_pct)

        self.cash = float(initial_equity)
        self.positions: dict[int, Position] = {}
        self.trades: list[Trade] = []
        #: ``(timestamp, equity)`` samples, appended once per processed bar.
        self.equity_curve: list[tuple[datetime, float]] = []

    # -- pricing helpers ----------------------------------------------------
    def fill_price(self, price: float, side: SignalSide, *, entering: bool) -> float:
        """Apply slippage in the direction that hurts.

        Entering long and exiting short both pay the ask; the reverse pay the
        bid. Modelling it symmetrically-optimistically is the single easiest way
        to build a backtest that cannot be reproduced live.
        """
        direction = side.value if entering else -side.value
        return price * (1.0 + direction * self.slippage_pct / 100.0)

    def fee_for(self, price: float, quantity: float) -> float:
        """Commission on a notional amount."""
        return abs(price * quantity) * self.fee_pct / 100.0

    # -- lifecycle ----------------------------------------------------------
    def open_position(
        self,
        signal: Signal,
        quantity: float,
        *,
        timestamp: datetime | None = None,
        price: float | None = None,
    ) -> Position:
        """Open a position from a signal.

        Args:
            signal: the approved signal.
            quantity: size in base units, decided by the risk manager.
            timestamp: fill time; defaults to now.
            price: fill price before slippage; defaults to the signal's entry.

        Returns:
            The created :class:`Position`.

        Raises:
            ValueError: on a non-positive quantity.
        """
        if quantity <= 0:
            raise ValueError(f"quantity must be > 0, got {quantity}")

        raw_price = signal.entry if price is None else price
        entry = self.fill_price(raw_price, signal.side, entering=True)
        fee = self.fee_for(entry, quantity)
        moment = timestamp or utc_now()

        position = Position(
            id=next(_position_ids),
            symbol=signal.symbol,
            side=signal.side,
            entry_price=entry,
            quantity=quantity,
            stop_loss=signal.stop_loss,
            take_profits=signal.take_profits,
            opened_at=moment,
            atr=signal.atr,
            initial_stop=signal.stop_loss,
            initial_risk=abs(entry - signal.stop_loss) * quantity,
            signal_score=signal.score,
            signal_strength=signal.strength.value,
            trigger=signal.trigger,
            timeframe=signal.timeframe,
        )
        position.fills.append(Fill(moment, entry, quantity, fee))
        position.fees_paid += fee
        self.cash -= fee
        self.positions[position.id] = position
        return position

    def close(
        self,
        position: Position,
        price: float,
        reason: ExitReason,
        *,
        timestamp: datetime | None = None,
        fraction: float = 1.0,
        bars_held: int = 0,
    ) -> Trade | None:
        """Close all or part of a position.

        Args:
            position: the position to reduce.
            price: exit price before slippage.
            reason: why it is being closed.
            timestamp: fill time.
            fraction: share of the *remaining* quantity to close, ``(0, 1]``.
            bars_held: bars the position was open, for the trade record.

        Returns:
            A :class:`Trade` when the position is now fully closed, otherwise
            ``None``.

        Raises:
            ValueError: on an out-of-range fraction.
        """
        if not 0 < fraction <= 1:
            raise ValueError(f"fraction must be in (0, 1], got {fraction}")
        if not position.is_open:
            return None

        quantity = position.remaining * fraction
        exit_price = self.fill_price(price, position.side, entering=False)
        fee = self.fee_for(exit_price, quantity)
        gross = (exit_price - position.entry_price) * position.side.value * quantity
        moment = timestamp or utc_now()

        position.remaining -= quantity
        position.realised_pnl += gross - fee
        position.fees_paid += fee
        position.fills.append(Fill(moment, exit_price, -quantity, fee, reason))
        self.cash += gross - fee

        if position.remaining > 1e-12:
            return None

        position.remaining = 0.0
        self.positions.pop(position.id, None)
        trade = Trade(
            position_id=position.id,
            symbol=position.symbol,
            side=position.side,
            timeframe=position.timeframe,
            entry_price=position.entry_price,
            exit_price=exit_price,
            quantity=position.quantity,
            opened_at=position.opened_at,
            closed_at=moment,
            pnl=position.realised_pnl,
            fees=position.fees_paid,
            exit_reason=reason,
            initial_risk=position.initial_risk,
            r_multiple=safe_div(position.realised_pnl, position.initial_risk),
            mae_r=position.drawdown_r,
            mfe_r=position.excursion_r,
            bars_held=bars_held,
            signal_score=position.signal_score,
            signal_strength=position.signal_strength,
            trigger=position.trigger,
        )
        self.trades.append(trade)
        return trade

    # -- valuation ----------------------------------------------------------
    def equity(self, marks: dict[str, float] | None = None) -> float:
        """Cash plus the mark-to-market value of open positions.

        Args:
            marks: latest price per symbol. Positions without a mark are valued
                at their entry, i.e. flat — never at a guessed price.
        """
        marks = marks or {}
        open_value = sum(
            position.unrealised_pnl(marks.get(position.symbol, position.entry_price))
            for position in self.positions.values()
        )
        return self.cash + open_value

    def record_equity(self, timestamp: datetime, marks: dict[str, float] | None = None) -> float:
        """Append one sample to the equity curve and return it."""
        value = self.equity(marks)
        self.equity_curve.append((timestamp, value))
        return value

    def open_for(self, symbol: str) -> list[Position]:
        return [p for p in self.positions.values() if p.symbol == symbol]

    @property
    def open_count(self) -> int:
        return len(self.positions)

    @property
    def realised_pnl(self) -> float:
        return sum(trade.pnl for trade in self.trades)

    def to_dict(self) -> dict[str, object]:
        return {
            "initial_equity": self.initial_equity,
            "cash": round(self.cash, 4),
            "open_positions": [p.to_dict() for p in self.positions.values()],
            "closed_trades": len(self.trades),
            "realised_pnl": round(self.realised_pnl, 4),
        }

    # -- persistence --------------------------------------------------------
    def export_state(self) -> dict[str, object]:
        """Serialise cash and open positions for a restart.

        Closed trades are intentionally **not** restored into the ledger: they
        live in ``logs/trade.jsonl``, and re-loading them would double-count the
        realised PnL that is already reflected in ``cash``.
        """
        return {
            "initial_equity": self.initial_equity,
            "cash": self.cash,
            "positions": [p.to_state() for p in self.positions.values()],
        }

    def restore_state(self, data: dict[str, object]) -> int:
        """Restore cash and open positions.

        Args:
            data: output of :meth:`export_state`.

        Returns:
            Number of positions restored.
        """
        self.cash = float(data.get("cash", self.initial_equity))  # type: ignore[arg-type]
        self.positions.clear()
        restored = 0
        for entry in data.get("positions", []):  # type: ignore[union-attr]
            try:
                position = Position.from_state(entry)
            except (KeyError, TypeError, ValueError):
                continue
            self.positions[position.id] = position
            # Keep the id counter ahead of anything restored so a new position
            # cannot collide with a persisted one.
            while next(_position_ids) <= position.id:
                pass
            restored += 1
        return restored
