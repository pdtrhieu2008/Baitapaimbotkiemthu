"""Entry triggers.

Direction and confluence answer *whether* to trade; a trigger answers *now*.
Without one, a bot that likes the trend simply enters on every bar the trend
persists — which is how an account ends up long at the top of an extended move.

Four triggers are recognised, checked in this order:

1. ``breakout`` — close beyond the prior Donchian boundary, with participation.
2. ``bos_continuation`` — a recent break of structure that price has held.
3. ``pullback`` — an established trend retraces into the fast/slow EMA or a
   demand/supply zone and prints a rejection.
4. ``sweep_reversal`` — a liquidity sweep that closed back inside the range.

Each returns the *reason* it fired so the notification can state it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from analysis.context import TimeframeContext
from analysis.regime import MarketRegime
from analysis.structure import BreakKind
from config.settings import IndicatorConfig
from strategies.base import SignalSide

__all__ = ["Trigger", "TriggerDetector"]


@dataclass(frozen=True, slots=True)
class Trigger:
    """A fired entry condition.

    Attributes:
        name: trigger identifier.
        detail: human-readable description with the numbers involved.
        quality: ``0..1`` confidence in the trigger itself, used to break ties
            between two setups with the same score.
    """

    name: str
    detail: str
    quality: float = 0.5

    def to_dict(self) -> dict[str, object]:
        return {"name": self.name, "detail": self.detail, "quality": self.quality}


class TriggerDetector:
    """Find the entry trigger, if any, on the last bar of a context.

    Args:
        cfg: the ``indicators`` configuration section.
        max_break_age: how many bars a break of structure stays actionable.
    """

    def __init__(self, cfg: IndicatorConfig, max_break_age: int = 5) -> None:
        self.cfg = cfg
        self.max_break_age = max_break_age

    def detect(self, ctx: TimeframeContext, side: SignalSide) -> Trigger | None:
        """Return the highest-priority trigger for ``side``, or ``None``.

        Args:
            ctx: entry-timeframe context.
            side: direction under consideration.

        Returns:
            A :class:`Trigger`, or ``None`` when nothing justifies acting on
            this specific bar.
        """
        for check in (
            self._breakout,
            self._bos_continuation,
            self._pullback,
            self._sweep_reversal,
        ):
            trigger = check(ctx, side)
            if trigger is not None:
                return trigger
        return None

    # -- individual triggers ------------------------------------------------
    def _breakout(self, ctx: TimeframeContext, side: SignalSide) -> Trigger | None:
        """Close beyond the previous Donchian boundary, funded by volume.

        The boundary is taken from the *previous* bar, because the current bar's
        channel already includes the current bar's own high — comparing against
        it would make every new high a "breakout".
        """
        frame = ctx.frame
        if len(frame) < 2 or "dc_upper" not in frame.columns:
            return None

        long_side = side is SignalSide.LONG
        boundary = float(frame["dc_upper"].iloc[-2] if long_side else frame["dc_lower"].iloc[-2])
        close = ctx.value("close")
        if not np.isfinite(boundary) or not np.isfinite(close):
            return None

        broke = close > boundary if long_side else close < boundary
        if not broke:
            return None

        # Participation is what separates a breakout from a wick through thin
        # liquidity. Either a volume spike or a released squeeze qualifies.
        released = bool(frame["squeeze_released"].iloc[-1]) if "squeeze_released" in frame else False
        if not (ctx.volume.spike or released):
            return None

        quality = 0.8 if (ctx.volume.spike and released) else 0.65
        driver = "volume spike" if ctx.volume.spike else "squeeze release"
        return Trigger(
            "breakout",
            f"close {close:.6g} broke the {self.cfg.donchian_period}-bar "
            f"{'high' if long_side else 'low'} {boundary:.6g} on {driver}",
            quality,
        )

    def _bos_continuation(self, ctx: TimeframeContext, side: SignalSide) -> Trigger | None:
        """A recent break of structure that price has since held."""
        event = ctx.structure.break_event
        if event.kind is BreakKind.NONE or event.kind.bias != side.value:
            return None
        if event.bars_ago > self.max_break_age:
            return None
        if event.kind.is_choch:
            # A change of character is a reversal warning, not a continuation
            # entry; it is handled by the pullback/sweep triggers once the new
            # structure has actually formed.
            return None

        close = ctx.value("close")
        held = close > event.price if side is SignalSide.LONG else close < event.price
        if not held:
            return None

        ema_fast = ctx.value("ema_fast")
        if np.isfinite(ema_fast):
            on_side = close > ema_fast if side is SignalSide.LONG else close < ema_fast
            if not on_side:
                return None

        return Trigger(
            "bos_continuation",
            f"BOS at {event.price:.6g} held for {event.bars_ago} bars",
            0.7,
        )

    def _pullback(self, ctx: TimeframeContext, side: SignalSide) -> Trigger | None:
        """A retrace into dynamic support/resistance that got rejected.

        Requires four things together: an established trend, price actually
        reaching the moving average or the zone, a rejection candle, and momentum
        turning back. Any one of them alone is a coin flip.
        """
        regime = ctx.regime
        long_side = side is SignalSide.LONG
        wanted_regime = MarketRegime.TREND_UP if long_side else MarketRegime.TREND_DOWN
        if regime.regime is not wanted_regime:
            return None

        frame = ctx.frame
        if len(frame) < 4:
            return None

        atr = ctx.value("atr")
        ema_fast, ema_slow = ctx.value("ema_fast"), ctx.value("ema_slow")
        if not all(np.isfinite(x) for x in (atr, ema_fast, ema_slow)) or atr <= 0:
            return None

        # Did the last few bars actually reach the average?
        recent = frame.tail(3)
        tolerance = 0.35 * atr
        if long_side:
            touched = bool(
                ((recent["low"] - ema_fast).abs() <= tolerance).any()
                or ((recent["low"] - ema_slow).abs() <= tolerance).any()
                or (recent["low"] <= ema_fast).any()
            )
        else:
            touched = bool(
                ((recent["high"] - ema_fast).abs() <= tolerance).any()
                or ((recent["high"] - ema_slow).abs() <= tolerance).any()
                or (recent["high"] >= ema_fast).any()
            )
        in_zone = ctx.structure.in_demand_zone if long_side else ctx.structure.in_supply_zone
        if not (touched or in_zone):
            return None

        # Rejection: either a directional candle or a recognised pattern.
        close, open_ = ctx.value("close"), ctx.value("open")
        directional = close > open_ if long_side else close < open_
        if not (directional or ctx.pattern_bias == side.value):
            return None

        # Momentum turning back in our direction.
        k, d = ctx.value("stochrsi_k"), ctx.value("stochrsi_d")
        if np.isfinite(k) and np.isfinite(d):
            turning = k > d if long_side else k < d
            if not turning:
                return None

        where = "demand zone" if in_zone and long_side else (
            "supply zone" if in_zone else f"EMA{self.cfg.ema_fast}/EMA{self.cfg.ema_slow}"
        )
        return Trigger("pullback", f"rejection from {where} inside a {regime.regime.value}", 0.75)

    def _sweep_reversal(self, ctx: TimeframeContext, side: SignalSide) -> Trigger | None:
        """A stop-run beyond a swing extreme that closed back inside."""
        structure = ctx.structure
        if structure.sweep_direction != side.value:
            return None

        close, open_ = ctx.value("close"), ctx.value("open")
        if not (close > open_ if side is SignalSide.LONG else close < open_):
            return None

        return Trigger(
            "sweep_reversal",
            "liquidity swept "
            + ("below support then reclaimed" if side is SignalSide.LONG else "above resistance then rejected"),
            0.6,
        )


def bars_since_signal(frame: pd.DataFrame, last_signal_time: object) -> int:
    """How many closed bars have printed since ``last_signal_time``.

    Used to enforce ``strategy.cooldown_bars``, which stops the bot re-entering
    the same setup on consecutive bars while the conditions persist.

    Args:
        frame: the entry-timeframe frame.
        last_signal_time: timestamp of the previous signal, or ``None``.

    Returns:
        Number of bars, or a large sentinel when there is no previous signal.
    """
    if last_signal_time is None or frame.empty:
        return 10_000
    try:
        position = frame.index.get_indexer([pd.Timestamp(last_signal_time)], method="pad")[0]
    except (KeyError, ValueError, TypeError):
        return 10_000
    if position < 0:
        return 10_000
    return len(frame) - 1 - int(position)
