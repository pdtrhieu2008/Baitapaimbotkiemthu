"""Stop-loss and take-profit placement — pure, stateless maths.

Separated from :mod:`risk.manager` on purpose: *where* the stop goes depends only
on price structure and configuration, while *how big* the position is depends on
account state. Keeping the first half pure makes it exhaustively testable and
identical in live and backtest.

The reward-to-risk figure this module reports is **net of costs**. A 2R target
measured gross is not 2R once two fees and two slippage estimates are paid, and
on a tight stop the difference decides whether a strategy has positive
expectancy at all.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from analysis.structure import StructureReport
from config.settings import RiskConfig
from strategies.base import SignalSide, TakeProfit
from utils.helpers import safe_div

__all__ = ["TradePlan", "TradePlanner"]


@dataclass(frozen=True, slots=True)
class TradePlan:
    """A proposed stop/target layout.

    Attributes:
        side: trade direction.
        entry: reference entry price.
        stop_loss: protective stop price.
        take_profits: scaled exits, nearest first.
        stop_distance: absolute distance from entry to stop.
        stop_source: ``"structure"``, ``"atr"``, ``"structure_capped"`` or
            ``"atr_floor"`` — recorded so a run can be audited later.
        risk_reward_gross: furthest target divided by stop distance.
        risk_reward_net: same, after subtracting round-trip costs from the
            reward and adding them to the risk.
        valid: whether the plan is usable.
        reason: why it is not, when ``valid`` is ``False``.
    """

    side: SignalSide
    entry: float
    stop_loss: float
    take_profits: tuple[TakeProfit, ...]
    stop_distance: float
    stop_source: str
    risk_reward_gross: float
    risk_reward_net: float
    valid: bool = True
    reason: str = ""

    @property
    def stop_distance_pct(self) -> float:
        return 100.0 * safe_div(self.stop_distance, self.entry)

    def to_dict(self) -> dict[str, object]:
        return {
            "side": self.side.name,
            "entry": self.entry,
            "stop_loss": self.stop_loss,
            "stop_distance_pct": round(self.stop_distance_pct, 3),
            "stop_source": self.stop_source,
            "take_profits": [tp.to_dict() for tp in self.take_profits],
            "rr_gross": round(self.risk_reward_gross, 2),
            "rr_net": round(self.risk_reward_net, 2),
            "valid": self.valid,
            "reason": self.reason,
        }


def _invalid(side: SignalSide, entry: float, reason: str) -> TradePlan:
    return TradePlan(
        side=side,
        entry=entry,
        stop_loss=float("nan"),
        take_profits=(),
        stop_distance=float("nan"),
        stop_source="none",
        risk_reward_gross=0.0,
        risk_reward_net=0.0,
        valid=False,
        reason=reason,
    )


class TradePlanner:
    """Compute stop and target levels from price structure and ATR.

    Args:
        cfg: the ``risk`` configuration section.
    """

    #: A structural stop wider than this multiple of the ATR stop is treated as
    #: unusable: it would either blow the RR budget or force a position so small
    #: the trade is not worth its costs.
    MAX_STRUCTURE_TO_ATR_RATIO = 2.5

    def __init__(self, cfg: RiskConfig) -> None:
        self.cfg = cfg

    def plan(
        self,
        side: SignalSide,
        entry: float,
        atr: float,
        structure: StructureReport | None = None,
    ) -> TradePlan:
        """Build a stop/target layout.

        Args:
            side: trade direction.
            entry: reference entry price (the signal bar's close).
            atr: ATR at signal time.
            structure: structure report, used for a swing-based stop when
                ``risk.use_structure_stop`` is enabled.

        Returns:
            A :class:`TradePlan`. Check :attr:`TradePlan.valid` before use — an
            invalid plan carries the reason instead of raising, because "this
            setup cannot be given a sane stop" is a normal outcome that belongs
            in the rejection log.
        """
        cfg = self.cfg
        if not np.isfinite(entry) or entry <= 0:
            return _invalid(side, entry, "entry price is not a positive finite number")
        if not np.isfinite(atr) or atr <= 0:
            return _invalid(side, entry, "ATR unavailable, cannot size the stop")

        atr_distance = cfg.sl_atr_multiplier * atr
        distance, source = atr_distance, "atr"

        structural = self._structure_distance(side, entry, atr, structure)
        if structural is not None:
            if structural > self.MAX_STRUCTURE_TO_ATR_RATIO * atr_distance:
                source = "atr_capped"  # swing too far away; fall back to ATR
            elif structural < 0.5 * atr_distance:
                # Swing sits inside the noise band: widen to the ATR floor so
                # ordinary volatility cannot take the trade out.
                distance, source = 0.5 * atr_distance, "atr_floor"
            else:
                distance, source = structural, "structure"

        # Hard ceiling on stop distance, independent of everything above.
        max_distance = entry * cfg.sl_max_pct / 100.0
        if distance > max_distance:
            distance, source = max_distance, f"{source}_pct_capped"

        if distance <= 0:
            return _invalid(side, entry, "computed stop distance is not positive")

        stop_loss = entry - side.value * distance
        if stop_loss <= 0:
            return _invalid(side, entry, "stop would fall at or below zero")

        targets = tuple(
            TakeProfit(
                price=entry + side.value * rr * distance,
                fraction=fraction,
                rr=rr,
            )
            for rr, fraction in zip(cfg.tp_rr_targets, cfg.tp_split, strict=True)
        )
        if not targets:
            return _invalid(side, entry, "no take-profit targets configured")

        furthest = max(cfg.tp_rr_targets)
        cost_per_unit = entry * cfg.round_trip_cost_pct / 100.0
        net_reward = furthest * distance - cost_per_unit
        net_risk = distance + cost_per_unit

        return TradePlan(
            side=side,
            entry=entry,
            stop_loss=stop_loss,
            take_profits=targets,
            stop_distance=distance,
            stop_source=source,
            risk_reward_gross=furthest,
            risk_reward_net=max(0.0, safe_div(net_reward, net_risk)),
        )

    def _structure_distance(
        self,
        side: SignalSide,
        entry: float,
        atr: float,
        structure: StructureReport | None,
    ) -> float | None:
        """Distance from entry to the protective swing, plus a buffer.

        The buffer exists because resting stops cluster exactly on the visible
        swing, which is precisely where a liquidity sweep reaches.
        """
        if not self.cfg.use_structure_stop or structure is None:
            return None

        anchor = (
            structure.last_swing_low if side is SignalSide.LONG else structure.last_swing_high
        )
        if anchor is None or not np.isfinite(anchor):
            return None

        buffer = self.cfg.sl_structure_buffer_atr * atr
        distance = (entry - anchor + buffer) if side is SignalSide.LONG else (anchor - entry + buffer)
        # A swing on the wrong side of entry (price already through it) gives a
        # negative distance and must be ignored rather than inverted.
        return distance if distance > 0 else None
