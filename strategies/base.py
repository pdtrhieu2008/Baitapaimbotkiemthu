"""Signal vocabulary and the strategy interface.

A :class:`Signal` is a fully-specified trade *proposal*: direction, entry, stop,
scaled targets, resulting reward-to-risk and the complete evidence trail that
produced it. It carries no position size — sizing is the risk manager's job, and
keeping the two separate means a signal can be logged, notified and backtested
without ever touching account state.

Every rejection is also a first-class object. Knowing *why* the bot did not
trade is as operationally important as knowing why it did, and it is the only
way to tell "no setup" apart from "a bug swallowed the setup".
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, IntEnum
from typing import TYPE_CHECKING

from utils.helpers import format_price, utc_now

if TYPE_CHECKING:  # avoid import cycles at runtime
    from analysis.context import MarketSnapshot
    from strategies.mtf import MTFVerdict
    from strategies.scoring import ScoreCard

__all__ = [
    "Rejection",
    "Signal",
    "SignalSide",
    "SignalStrength",
    "Strategy",
    "TakeProfit",
]


class SignalSide(IntEnum):
    """Trade direction. The integer value doubles as the PnL sign multiplier."""

    LONG = 1
    SHORT = -1

    @property
    def label(self) -> str:
        return "BUY" if self is SignalSide.LONG else "SELL"

    @property
    def opposite(self) -> SignalSide:
        return SignalSide.SHORT if self is SignalSide.LONG else SignalSide.LONG

    @classmethod
    def from_int(cls, value: int) -> SignalSide:
        if value > 0:
            return cls.LONG
        if value < 0:
            return cls.SHORT
        raise ValueError("0 is not a trade direction")


class SignalStrength(str, Enum):
    """Confidence tier derived from the total score."""

    WEAK = "weak"
    NORMAL = "normal"
    STRONG = "strong"

    @property
    def label(self) -> str:
        return {"weak": "WEAK", "normal": "", "strong": "STRONG"}[self.value]


@dataclass(frozen=True, slots=True)
class TakeProfit:
    """One scaled exit.

    Attributes:
        price: limit price of the partial exit.
        fraction: share of the position closed here (all fractions sum to 1).
        rr: reward-to-risk multiple this target represents.
    """

    price: float
    fraction: float
    rr: float

    def to_dict(self) -> dict[str, float]:
        return {"price": self.price, "fraction": self.fraction, "rr": self.rr}


@dataclass(slots=True)
class Signal:
    """A validated trade proposal.

    Attributes:
        symbol: unified symbol.
        timeframe: the timeframe the entry was triggered on.
        side: long or short.
        strength: confidence tier.
        score: total confluence score, 0-100.
        entry: reference entry price (the close of the signal bar; the actual
            fill is the next bar's open both live and in the backtest).
        stop_loss: protective stop.
        take_profits: scaled exits, nearest first.
        risk_reward: RR of the furthest target.
        atr: ATR at signal time — the risk unit for everything downstream.
        bar_time: open time of the closed bar that produced the signal.
        reasons: human-readable evidence, in scoring order.
        scorecard: per-component breakdown.
        mtf: multi-timeframe verdict.
        trigger: which entry trigger fired (``"breakout"``, ``"pullback"``, ...).
        notes: extra values for the notification (trend, volume, regime, ...).
    """

    symbol: str
    timeframe: str
    side: SignalSide
    strength: SignalStrength
    score: float
    entry: float
    stop_loss: float
    take_profits: tuple[TakeProfit, ...]
    risk_reward: float
    atr: float
    bar_time: datetime
    reasons: list[str] = field(default_factory=list)
    scorecard: ScoreCard | None = None
    mtf: MTFVerdict | None = None
    trigger: str = ""
    notes: dict[str, object] = field(default_factory=dict)
    created_at: datetime = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        # A stop on the wrong side of entry silently inverts the risk
        # calculation and would size the position off a negative denominator,
        # so it is rejected at construction rather than discovered later.
        if self.side is SignalSide.LONG and self.stop_loss >= self.entry:
            raise ValueError(
                f"long stop {self.stop_loss} must be below entry {self.entry}"
            )
        if self.side is SignalSide.SHORT and self.stop_loss <= self.entry:
            raise ValueError(
                f"short stop {self.stop_loss} must be above entry {self.entry}"
            )
        if not self.take_profits:
            raise ValueError("a signal must carry at least one take-profit")
        for target in self.take_profits:
            if self.side is SignalSide.LONG and target.price <= self.entry:
                raise ValueError(f"long target {target.price} must be above entry {self.entry}")
            if self.side is SignalSide.SHORT and target.price >= self.entry:
                raise ValueError(f"short target {target.price} must be below entry {self.entry}")

    @property
    def risk_per_unit(self) -> float:
        """Absolute stop distance — the denominator of every sizing formula."""
        return abs(self.entry - self.stop_loss)

    @property
    def stop_distance_pct(self) -> float:
        return 100.0 * self.risk_per_unit / self.entry if self.entry else 0.0

    @property
    def final_target(self) -> TakeProfit:
        return self.take_profits[-1]

    @property
    def key(self) -> str:
        """Stable identity for de-duplication across restarts."""
        return f"{self.symbol}|{self.timeframe}|{self.bar_time.isoformat()}|{self.side.name}"

    def summary(self) -> str:
        """One-line log form."""
        return (
            f"{self.side.label} {self.symbol} {self.timeframe} "
            f"score={self.score:.0f} ({self.strength.value}) "
            f"entry={format_price(self.entry)} sl={format_price(self.stop_loss)} "
            f"tp={format_price(self.final_target.price)} rr={self.risk_reward:.2f} "
            f"trigger={self.trigger}"
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "key": self.key,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "side": self.side.name,
            "strength": self.strength.value,
            "score": round(self.score, 1),
            "entry": self.entry,
            "stop_loss": self.stop_loss,
            "take_profits": [tp.to_dict() for tp in self.take_profits],
            "risk_reward": round(self.risk_reward, 2),
            "stop_distance_pct": round(self.stop_distance_pct, 3),
            "atr": self.atr,
            "bar_time": self.bar_time.isoformat(),
            "trigger": self.trigger,
            "reasons": self.reasons,
            "scorecard": self.scorecard.to_dict() if self.scorecard else None,
            "mtf": self.mtf.to_dict() if self.mtf else None,
            "notes": self.notes,
        }


@dataclass(slots=True)
class Rejection:
    """Why no signal was produced.

    Attributes:
        symbol: unified symbol.
        timeframe: entry timeframe under evaluation.
        gate: the gate or stage that blocked the trade.
        detail: specifics, including the numbers involved.
        side: the direction that was being considered, when known.
        score: the score reached before rejection, when computed.
    """

    symbol: str
    timeframe: str
    gate: str
    detail: str
    side: SignalSide | None = None
    score: float | None = None
    bar_time: datetime | None = None

    def summary(self) -> str:
        side = self.side.label if self.side else "-"
        score = f" score={self.score:.0f}" if self.score is not None else ""
        return f"{self.symbol} {self.timeframe} [{side}]{score} blocked by {self.gate}: {self.detail}"

    def to_dict(self) -> dict[str, object]:
        return {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "gate": self.gate,
            "detail": self.detail,
            "side": self.side.name if self.side else None,
            "score": self.score,
            "bar_time": self.bar_time.isoformat() if self.bar_time else None,
        }


class Strategy(ABC):
    """Interface every strategy implements.

    The contract is intentionally narrow — one method, one snapshot in, one
    signal or rejection out, no I/O and no mutable market state — because that
    is exactly what makes the same object usable by the live loop and by the
    backtester.
    """

    #: Human-readable name, surfaced in logs and notifications.
    name: str = "strategy"

    @abstractmethod
    def evaluate(self, snapshot: MarketSnapshot) -> Signal | Rejection:
        """Assess one symbol.

        Args:
            snapshot: all analysed timeframes plus external context.

        Returns:
            A :class:`Signal` when every gate passes and the score clears the
            threshold, otherwise a :class:`Rejection` explaining what blocked it.
        """

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<{type(self).__name__} name={self.name!r}>"
