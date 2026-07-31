"""The confluence scoring system.

Eight independent components each own a slice of a 100-point budget, configured
in ``strategy.weights``. A component reports what fraction of its own weight the
current market earns *for a specific side*, plus the reasons behind it.

Two design decisions carry most of the weight here:

**Nothing scores on a single indicator.** Each component is itself a small vote
of several independent readings, so one distorted input cannot carry a component
to full marks.

**Unavailable inputs are removed, not scored as zero.** If no funding/sentiment
feed is configured, the sentiment component is marked unavailable and its 10
points leave the denominator entirely. Scoring it as 0 would silently cap every
score at 90 and shift the meaning of every threshold in the config.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from analysis.context import MarketSnapshot, TimeframeContext
from analysis.regime import TrendStrength
from analysis.structure import BreakKind
from config.settings import IndicatorConfig, StrategyConfig
from data.models import ExternalContext
from strategies.base import SignalSide, SignalStrength
from utils.helpers import safe_div

__all__ = ["ScoreCard", "ScoreComponent", "Scorer"]


@dataclass(slots=True)
class ScoreComponent:
    """One scored dimension.

    Attributes:
        name: component key, matching ``strategy.weights``.
        weight: points budgeted for it.
        fraction: how much of its budget the market earned, ``[0, 1]``.
        available: ``False`` when the inputs are missing; the weight is then
            redistributed rather than counted as a zero.
        reasons: evidence that earned the points.
        against: evidence that argued the other way.
    """

    name: str
    weight: float
    fraction: float = 0.0
    available: bool = True
    reasons: list[str] = field(default_factory=list)
    against: list[str] = field(default_factory=list)

    @property
    def earned(self) -> float:
        return self.weight * self.fraction if self.available else 0.0

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "weight": self.weight,
            "earned": round(self.earned, 1),
            "fraction": round(self.fraction, 3),
            "available": self.available,
            "reasons": self.reasons,
            "against": self.against,
        }


@dataclass(slots=True)
class ScoreCard:
    """Aggregated score for one side.

    Attributes:
        side: the direction that was scored.
        components: per-dimension breakdown.
    """

    side: SignalSide
    components: list[ScoreComponent]

    @property
    def available_weight(self) -> float:
        """Total weight of the components that had usable inputs."""
        return sum(c.weight for c in self.components if c.available)

    @property
    def total(self) -> float:
        """Score rescaled to 0-100 over the available weight."""
        return 100.0 * safe_div(
            sum(c.earned for c in self.components), self.available_weight
        )

    @property
    def reasons(self) -> list[str]:
        """Flat list of every positive reason, in component order."""
        return [reason for component in self.components for reason in component.reasons]

    @property
    def objections(self) -> list[str]:
        return [item for component in self.components for item in component.against]

    @property
    def unavailable(self) -> list[str]:
        return [c.name for c in self.components if not c.available]

    def strength(self, cfg: StrategyConfig) -> SignalStrength:
        """Map the total onto a confidence tier."""
        total = self.total
        if total >= cfg.min_score_strong:
            return SignalStrength.STRONG
        if total >= cfg.min_score_normal:
            return SignalStrength.NORMAL
        return SignalStrength.WEAK

    def to_dict(self) -> dict[str, object]:
        return {
            "side": self.side.name,
            "total": round(self.total, 1),
            "available_weight": self.available_weight,
            "unavailable": self.unavailable,
            "components": [c.to_dict() for c in self.components],
        }


def _vote_fraction(votes: list[bool]) -> float:
    """Fraction of satisfied conditions, ``0.0`` when the list is empty."""
    return safe_div(sum(1 for v in votes if v), len(votes))


class Scorer:
    """Compute a :class:`ScoreCard` for a candidate direction.

    Args:
        strategy_cfg: the ``strategy`` section (weights).
        indicator_cfg: the ``indicators`` section (thresholds).
    """

    def __init__(self, strategy_cfg: StrategyConfig, indicator_cfg: IndicatorConfig) -> None:
        self.cfg = strategy_cfg
        self.ind = indicator_cfg

    def score(self, snapshot: MarketSnapshot, side: SignalSide) -> ScoreCard:
        """Score ``side`` on the snapshot's primary timeframe.

        Args:
            snapshot: analysed market snapshot.
            side: direction under consideration.

        Returns:
            A populated :class:`ScoreCard`.
        """
        context = snapshot.primary
        components = [
            self._trend(context, side),
            self._macd(context, side),
            self._rsi(context, side),
            self._volume(context, side),
            self._adx(context, side),
            self._price_action(context, side),
            self._structure(context, side),
            self._sentiment(context, snapshot.external, side),
        ]
        return ScoreCard(side=side, components=components)

    # -- components ---------------------------------------------------------
    def _weight(self, name: str) -> float:
        return float(self.cfg.weights.get(name, 0.0))

    def _trend(self, ctx: TimeframeContext, side: SignalSide) -> ScoreComponent:
        """EMA stack, price vs the long EMA, Supertrend, Ichimoku, slope."""
        component = ScoreComponent("trend", self._weight("trend"))
        long_side = side is SignalSide.LONG

        close = ctx.value("close")
        ema_fast, ema_slow = ctx.value("ema_fast"), ctx.value("ema_slow")
        ema_trend = ctx.value("ema_trend")
        supertrend_dir = ctx.value("supertrend_dir", 0.0)
        cloud_top, cloud_bottom = ctx.value("cloud_top"), ctx.value("cloud_bottom")
        slope = ctx.value("ema_slow_slope", 0.0)

        if not np.isfinite(close) or not np.isfinite(ema_slow):
            component.available = False
            return component

        votes: list[bool] = []

        stacked = bool(ema_fast > ema_slow) if long_side else bool(ema_fast < ema_slow)
        votes.append(stacked)
        if stacked:
            component.reasons.append(
                f"EMA{self.ind.ema_fast} {'>' if long_side else '<'} EMA{self.ind.ema_slow}"
            )
        else:
            component.against.append(f"EMA stack opposes {side.label}")

        if np.isfinite(ema_trend):
            above = bool(close > ema_trend) if long_side else bool(close < ema_trend)
            votes.append(above)
            if above:
                component.reasons.append(
                    f"price {'above' if long_side else 'below'} EMA{self.ind.ema_trend}"
                )
            else:
                component.against.append(f"price on the wrong side of EMA{self.ind.ema_trend}")

        if supertrend_dir != 0:
            agrees = supertrend_dir > 0 if long_side else supertrend_dir < 0
            votes.append(bool(agrees))
            if agrees:
                component.reasons.append("Supertrend aligned")
            else:
                component.against.append("Supertrend opposes")

        if np.isfinite(cloud_top) and np.isfinite(cloud_bottom):
            outside = close > cloud_top if long_side else close < cloud_bottom
            votes.append(bool(outside))
            if outside:
                component.reasons.append("price clear of the Ichimoku cloud")
            else:
                component.against.append("price inside/against the cloud")

        if np.isfinite(slope) and slope != 0:
            rising = slope > 0 if long_side else slope < 0
            votes.append(bool(rising))
            if rising:
                component.reasons.append(f"EMA{self.ind.ema_slow} slope {slope:+.3f}%/bar")

        component.fraction = _vote_fraction(votes)
        return component

    def _macd(self, ctx: TimeframeContext, side: SignalSide) -> ScoreComponent:
        """Histogram sign, a recent cross, expansion, and the zero line."""
        component = ScoreComponent("macd", self._weight("macd"))
        long_side = side is SignalSide.LONG

        macd_line, signal_line = ctx.value("macd"), ctx.value("macd_signal")
        histogram = ctx.value("macd_hist")
        if not all(np.isfinite(x) for x in (macd_line, signal_line, histogram)):
            component.available = False
            return component

        votes: list[bool] = []

        aligned = histogram > 0 if long_side else histogram < 0
        votes.append(bool(aligned))
        if aligned:
            component.reasons.append("MACD histogram aligned")
        else:
            component.against.append("MACD histogram opposes")

        # A cross within the last few bars is what makes this a *trigger* rather
        # than a description of an already-mature move.
        crossed = self._recent_macd_cross(ctx, long_side, bars=4)
        votes.append(crossed)
        if crossed:
            component.reasons.append("MACD cross in the last 4 bars")

        expanding = self._histogram_expanding(ctx, long_side)
        votes.append(expanding)
        if expanding:
            component.reasons.append("histogram expanding")
        else:
            component.against.append("histogram contracting")

        zero_side = macd_line > 0 if long_side else macd_line < 0
        votes.append(bool(zero_side))
        if zero_side:
            component.reasons.append("MACD on the correct side of zero")

        component.fraction = _vote_fraction(votes)
        return component

    @staticmethod
    def _recent_macd_cross(ctx: TimeframeContext, long_side: bool, bars: int) -> bool:
        """Whether MACD crossed its signal line in the desired direction."""
        if ctx.frame.empty or "macd_hist" not in ctx.frame.columns:
            return False
        window = ctx.frame["macd_hist"].tail(bars + 1)
        if window.isna().any() or len(window) < 2:
            return False
        signs = np.sign(window.to_numpy(dtype=float))
        target = 1.0 if long_side else -1.0
        # Current sign is the target and at least one earlier bar in the window
        # had the opposite sign -> a cross happened inside the window.
        return bool(signs[-1] == target and (signs[:-1] == -target).any())

    @staticmethod
    def _histogram_expanding(ctx: TimeframeContext, long_side: bool) -> bool:
        if ctx.frame.empty or "macd_hist" not in ctx.frame.columns:
            return False
        window = ctx.frame["macd_hist"].tail(3)
        if len(window) < 3 or window.isna().any():
            return False
        values = window.to_numpy(dtype=float)
        signed = values if long_side else -values
        return bool(signed[-1] > signed[-2] > signed[-3])

    def _rsi(self, ctx: TimeframeContext, side: SignalSide) -> ScoreComponent:
        """RSI position, direction and room before the exhaustion zone."""
        component = ScoreComponent("rsi", self._weight("rsi"))
        rsi = ctx.value("rsi")
        if not np.isfinite(rsi):
            component.available = False
            return component

        long_side = side is SignalSide.LONG
        votes: list[bool] = []

        reclaimed = rsi > self.ind.rsi_bull_mid if long_side else rsi < self.ind.rsi_bull_mid
        votes.append(bool(reclaimed))
        if reclaimed:
            component.reasons.append(f"RSI {rsi:.0f} on the {side.label} side of the midline")
        else:
            component.against.append(f"RSI {rsi:.0f} against {side.label}")

        # Entering *into* an extreme is a worse risk/reward than entering with
        # room left, so an overbought long is penalised, not rewarded.
        room = rsi < self.ind.rsi_overbought if long_side else rsi > self.ind.rsi_oversold
        votes.append(bool(room))
        if not room:
            component.against.append(
                f"RSI {rsi:.0f} already {'overbought' if long_side else 'oversold'}"
            )

        rising = self._rsi_turning(ctx, long_side)
        votes.append(rising)
        if rising:
            component.reasons.append("RSI turning up" if long_side else "RSI turning down")

        component.fraction = _vote_fraction(votes)
        return component

    @staticmethod
    def _rsi_turning(ctx: TimeframeContext, long_side: bool) -> bool:
        if ctx.frame.empty or "rsi" not in ctx.frame.columns:
            return False
        window = ctx.frame["rsi"].tail(3)
        if len(window) < 3 or window.isna().any():
            return False
        values = window.to_numpy(dtype=float)
        return bool(values[-1] > values[-2]) if long_side else bool(values[-1] < values[-2])

    def _volume(self, ctx: TimeframeContext, side: SignalSide) -> ScoreComponent:
        """Participation and flow direction — the heaviest single component."""
        component = ScoreComponent("volume", self._weight("volume"))
        state = ctx.volume
        if not np.isfinite(state.relative):
            component.available = False
            return component

        long_side = side is SignalSide.LONG
        votes: list[bool] = []

        votes.append(state.spike)
        if state.spike:
            component.reasons.append(f"volume spike {state.relative:.1f}x average")
        elif state.dry:
            component.against.append(f"dry volume {state.relative:.1f}x average")

        flow_aligned = state.bias == side.value
        votes.append(flow_aligned)
        if flow_aligned:
            component.reasons.append(f"net flow {state.pressure:+.0f}% favours {side.label}")
        elif state.bias != 0:
            component.against.append("net flow opposes")

        obv_aligned = state.obv_rising if long_side else not state.obv_rising
        votes.append(obv_aligned)
        if obv_aligned:
            component.reasons.append("OBV trend aligned")

        cmf_aligned = state.cmf > 0 if long_side else state.cmf < 0
        votes.append(bool(cmf_aligned))
        if cmf_aligned:
            component.reasons.append(f"CMF {state.cmf:+.2f}")

        component.fraction = _vote_fraction(votes)
        return component

    def _adx(self, ctx: TimeframeContext, side: SignalSide) -> ScoreComponent:
        """Trend strength and directional-indicator agreement."""
        component = ScoreComponent("adx", self._weight("adx"))
        adx = ctx.value("adx")
        if not np.isfinite(adx):
            component.available = False
            return component

        plus_di, minus_di = ctx.value("plus_di", 0.0), ctx.value("minus_di", 0.0)
        long_side = side is SignalSide.LONG
        votes: list[bool] = []

        above_min = adx >= self.ind.adx_trend_min
        votes.append(bool(above_min))
        if above_min:
            component.reasons.append(f"ADX {adx:.0f} > {self.ind.adx_trend_min:.0f}")
        else:
            component.against.append(f"ADX {adx:.0f} below the trend threshold")

        votes.append(bool(adx >= self.ind.adx_strong))
        if adx >= self.ind.adx_strong:
            component.reasons.append("strong trend")

        di_aligned = plus_di > minus_di if long_side else minus_di > plus_di
        votes.append(bool(di_aligned))
        if di_aligned:
            component.reasons.append(f"DI+ {plus_di:.0f} / DI- {minus_di:.0f} aligned")
        else:
            component.against.append("DI opposes")

        rising = self._adx_rising(ctx)
        votes.append(rising)
        if rising:
            component.reasons.append("ADX rising")

        component.fraction = _vote_fraction(votes)
        return component

    @staticmethod
    def _adx_rising(ctx: TimeframeContext) -> bool:
        if ctx.frame.empty or "adx" not in ctx.frame.columns:
            return False
        window = ctx.frame["adx"].tail(3)
        if len(window) < 3 or window.isna().any():
            return False
        return bool(window.iloc[-1] > window.iloc[-3])

    def _price_action(self, ctx: TimeframeContext, side: SignalSide) -> ScoreComponent:
        """Candlestick evidence, sweeps and failed breaks."""
        component = ScoreComponent("price_action", self._weight("price_action"))
        votes: list[bool] = []
        structure = ctx.structure

        pattern_aligned = ctx.pattern_bias == side.value
        votes.append(pattern_aligned)
        if pattern_aligned:
            component.reasons.append(f"candles: {', '.join(ctx.pattern_names)}")
        elif ctx.pattern_bias != 0:
            component.against.append(f"opposing candles: {', '.join(ctx.pattern_names)}")

        sweep_aligned = structure.sweep_direction == side.value
        votes.append(sweep_aligned)
        if sweep_aligned:
            component.reasons.append(
                "liquidity sweep below support" if side is SignalSide.LONG
                else "liquidity sweep above resistance"
            )
        elif structure.sweep_direction != 0:
            component.against.append("sweep points the other way")

        false_break_aligned = structure.false_breakout_direction == side.value
        votes.append(false_break_aligned)
        if false_break_aligned:
            component.reasons.append("failed breakout in our favour")

        # A rejection candle only means something at a level. In open space it
        # is noise, so the zone check is a separate vote.
        at_zone = structure.in_demand_zone if side is SignalSide.LONG else structure.in_supply_zone
        votes.append(at_zone)
        if at_zone:
            component.reasons.append(
                "price in demand zone" if side is SignalSide.LONG else "price in supply zone"
            )

        component.fraction = _vote_fraction(votes)
        return component

    def _structure(self, ctx: TimeframeContext, side: SignalSide) -> ScoreComponent:
        """Swing structure, break classification, room and exhaustion."""
        component = ScoreComponent("structure", self._weight("structure"))
        structure = ctx.structure
        regime = ctx.regime
        votes: list[bool] = []

        trend_aligned = structure.trend.bias == side.value
        votes.append(trend_aligned)
        if trend_aligned:
            component.reasons.append(f"structure {structure.trend.value} ({'/'.join(structure.labels[-3:])})")
        elif structure.trend.bias != 0:
            component.against.append(f"structure is {structure.trend.value}")

        break_event = structure.break_event
        break_aligned = break_event.kind.bias == side.value
        votes.append(break_aligned)
        if break_aligned:
            label = "CHoCH" if break_event.kind.is_choch else "BOS"
            component.reasons.append(f"{label} confirmed {break_event.bars_ago} bars ago")
        elif break_event.kind is not BreakKind.NONE:
            component.against.append(f"last break was {break_event.kind.value}")

        # Room is scored here and *also* enforced as a hard gate: a marginal
        # amount of room should cost points, no room at all should veto.
        room = structure.room_for(side.value)
        has_room = room >= 2.0
        votes.append(bool(has_room))
        if has_room:
            component.reasons.append(f"{room:.1f} ATR of room to the next level")
        else:
            component.against.append(f"only {room:.1f} ATR to the next level")

        healthy = regime.strength is not TrendStrength.NONE and not regime.exhaustion
        votes.append(healthy)
        if regime.exhaustion:
            component.against.append("trend exhaustion detected")

        component.fraction = _vote_fraction(votes)
        return component

    def _sentiment(
        self, ctx: TimeframeContext, external: ExternalContext, side: SignalSide
    ) -> ScoreComponent:
        """Funding, open interest, Fear & Greed, dominance and news.

        Marked unavailable — weight redistributed — when no feed produced a
        usable number, which is the normal case on spot markets or when the bot
        runs without the optional sentiment sources.
        """
        component = ScoreComponent("sentiment", self._weight("sentiment"))
        if not external.has_any_sentiment:
            component.available = False
            if external.unavailable:
                component.against.append(
                    "sentiment unavailable: "
                    + "; ".join(f"{k} ({v})" for k, v in external.unavailable.items())
                )
            return component

        votes: list[bool] = []

        if external.funding is not None and np.isfinite(external.funding.rate_pct):
            funding = external.funding
            aligned = funding.bias == side.value or funding.bias == 0
            votes.append(bool(funding.bias == side.value))
            if funding.bias == side.value:
                component.reasons.append(
                    f"funding {funding.rate_pct:+.4f}% favours {side.label} (crowd on the other side)"
                )
            elif not aligned:
                component.against.append(f"funding {funding.rate_pct:+.4f}% is crowded with us")

        if external.open_interest is not None:
            price_change = ctx.value("close") - ctx.frame["close"].iloc[-2] if len(ctx.frame) > 1 else 0.0
            price_change_pct = 100.0 * safe_div(price_change, ctx.value("close", 1.0))
            oi_bias = external.open_interest.bias(price_change_pct)
            if oi_bias != 0:
                votes.append(oi_bias == side.value)
                if oi_bias == side.value:
                    component.reasons.append(
                        f"open interest {external.open_interest.change_pct:+.1f}% confirms the move"
                    )
                else:
                    component.against.append("open interest suggests the move is fading")

        sentiment = external.sentiment
        if sentiment is not None:
            if sentiment.fear_greed_bias != 0:
                votes.append(sentiment.fear_greed_bias == side.value)
                if sentiment.fear_greed_bias == side.value:
                    component.reasons.append(f"Fear & Greed {sentiment.fear_greed:.0f} (contrarian)")
                else:
                    component.against.append(f"Fear & Greed {sentiment.fear_greed:.0f} against us")
            if sentiment.news_bias != 0:
                votes.append(sentiment.news_bias == side.value)
                if sentiment.news_bias == side.value:
                    component.reasons.append(f"news sentiment {sentiment.news_score:+.2f}")
                else:
                    component.against.append("news sentiment against us")
            if np.isfinite(sentiment.btc_dominance_change_pct):
                # Rising dominance drains altcoins; on BTC itself it is neutral.
                is_btc = ctx.symbol.upper().startswith("BTC")
                if not is_btc and abs(sentiment.btc_dominance_change_pct) > 0.3:
                    favourable = (
                        sentiment.btc_dominance_change_pct < 0
                        if side is SignalSide.LONG
                        else sentiment.btc_dominance_change_pct > 0
                    )
                    votes.append(favourable)
                    if favourable:
                        component.reasons.append(
                            f"BTC dominance {sentiment.btc_dominance_change_pct:+.2f}% helps alts"
                        )
                    else:
                        component.against.append("BTC dominance flow against alts")

        if not votes:
            component.available = False
            return component

        component.fraction = _vote_fraction(votes)
        return component
