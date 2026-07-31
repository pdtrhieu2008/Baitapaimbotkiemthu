"""The confluence strategy: hard gates first, score second.

The evaluation order is the whole design:

1. **Data sanity** — are the indicators even seeded?
2. **Direction** — proposed by the higher timeframes, never by the entry chart.
3. **Multi-timeframe agreement** — context must not oppose, confirm must agree.
4. **Trigger** — something must justify acting on *this* bar.
5. **Hard gates** — ADX, volume, ATR range, room to the next level, structure.
   Any enabled gate that fails vetoes the trade *regardless of score*.
6. **Trade plan** — stop and targets; the net reward-to-risk gate applies here.
7. **Score threshold** — only now does the 0-100 confluence score decide.

That ordering is what implements "if conditions are missing, do NOT enter". A
score of 95 cannot buy its way past a failed gate: the gates are structural
statements about whether the setup is tradable at all, and the score only ranks
the setups that already are.
"""

from __future__ import annotations

from datetime import date, datetime

import numpy as np

from analysis.context import MarketSnapshot, TimeframeContext
from analysis.regime import VolatilityState
from config.settings import Settings
from risk.planner import TradePlan, TradePlanner
from strategies.base import Rejection, Signal, SignalSide, Strategy
from strategies.mtf import MTFVerdict, MultiTimeframeAnalyser
from strategies.scoring import ScoreCard, Scorer
from strategies.triggers import Trigger, TriggerDetector, bars_since_signal
from utils.logger import get_logger

__all__ = ["ConfluenceStrategy"]

_log = get_logger("strategies.confluence")


class ConfluenceStrategy(Strategy):
    """Multi-condition, multi-timeframe trend-continuation strategy.

    Args:
        settings: the full validated configuration.
    """

    name = "confluence"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.cfg = settings.strategy
        self.scorer = Scorer(settings.strategy, settings.indicators)
        self.mtf = MultiTimeframeAnalyser(settings.strategy.mtf)
        self.triggers = TriggerDetector(settings.indicators)
        self.planner = TradePlanner(settings.risk)

        # --- small amount of per-symbol state, needed for cooldown and the
        # daily signal cap. Both the live loop and the backtester drive it
        # through register_signal(), so the two behave identically.
        self._last_signal_bar: dict[str, datetime] = {}
        self._daily_count: dict[tuple[str, date], int] = {}

    # -- state --------------------------------------------------------------
    def register_signal(self, signal: Signal) -> None:
        """Record an accepted signal so cooldown and the daily cap apply."""
        self._last_signal_bar[signal.symbol] = signal.bar_time
        key = (signal.symbol, signal.bar_time.date())
        self._daily_count[key] = self._daily_count.get(key, 0) + 1

    def reset(self) -> None:
        """Clear per-symbol state (used between backtest runs)."""
        self._last_signal_bar.clear()
        self._daily_count.clear()

    # -- evaluation ---------------------------------------------------------
    def evaluate(self, snapshot: MarketSnapshot) -> Signal | Rejection:
        """Assess one symbol; see the module docstring for the gate order."""
        symbol = snapshot.symbol
        timeframe = snapshot.primary_timeframe

        def reject(gate: str, detail: str, **extra: object) -> Rejection:
            return Rejection(
                symbol=symbol,
                timeframe=timeframe,
                gate=gate,
                detail=detail,
                bar_time=context.bar_time if context is not None else None,
                **extra,  # type: ignore[arg-type]
            )

        # --- 1. data sanity ---
        context: TimeframeContext | None = None
        try:
            context = snapshot.primary
        except KeyError as exc:
            return Rejection(symbol, timeframe, "data", str(exc))
        if not context.is_complete:
            return reject("data", "core indicators not seeded yet (warm-up incomplete)")

        # --- 2. direction ---
        side = self.mtf.propose_side(snapshot)
        if side is None:
            return reject("direction", "higher timeframes are undecided")
        if side is SignalSide.LONG and not self.cfg.allow_long:
            return reject("direction", "long signals disabled by configuration", side=side)
        if side is SignalSide.SHORT and not self.cfg.allow_short:
            return reject("direction", "short signals disabled by configuration", side=side)

        # --- cooldown / daily cap (cheap, and independent of the setup) ---
        cooldown = self._cooldown_block(context, symbol)
        if cooldown is not None:
            return reject("cooldown", cooldown, side=side)

        # --- 3. multi-timeframe agreement ---
        mtf_verdict = self.mtf.evaluate(snapshot, side)
        if self.cfg.gate_enabled("mtf_alignment") and not mtf_verdict.passed:
            return reject("mtf_alignment", mtf_verdict.detail, side=side)

        # --- 4. trigger ---
        trigger, trigger_context = self._find_trigger(snapshot, side)
        if self.cfg.gate_enabled("trigger_required") and trigger is None:
            return reject(
                "trigger_required",
                "no breakout, pullback, BOS continuation or sweep reversal on this bar",
                side=side,
            )

        # --- score, so that every later rejection can report it ---
        scorecard = self.scorer.score(snapshot, side)
        score = scorecard.total

        # --- 5. hard gates ---
        blocked = self._check_gates(context, side)
        if blocked is not None:
            gate, detail = blocked
            return reject(gate, detail, side=side, score=score)

        # --- 6. trade plan ---
        entry_context = trigger_context or context
        plan = self.planner.plan(
            side=side,
            entry=entry_context.value("close"),
            atr=entry_context.value("atr"),
            structure=entry_context.structure,
        )
        if not plan.valid:
            return reject("trade_plan", plan.reason, side=side, score=score)
        if self.cfg.gate_enabled("min_rr") and plan.risk_reward_net < self.settings.risk.min_rr:
            return reject(
                "min_rr",
                f"net RR {plan.risk_reward_net:.2f} below the required "
                f"{self.settings.risk.min_rr:.2f} "
                f"(gross {plan.risk_reward_gross:.2f}, "
                f"costs {self.settings.risk.round_trip_cost_pct:.3f}% round trip)",
                side=side,
                score=score,
            )

        # --- 7. score threshold ---
        if score < self.cfg.min_score_to_emit:
            return reject(
                "score",
                f"score {score:.1f} below the {self.cfg.min_score_to_emit:.0f} threshold; "
                f"weakest components: {self._weakest(scorecard)}",
                side=side,
                score=score,
            )

        return self._build_signal(
            snapshot, entry_context, side, scorecard, mtf_verdict, trigger, plan
        )

    # -- helpers ------------------------------------------------------------
    def _cooldown_block(self, context: TimeframeContext, symbol: str) -> str | None:
        """Return a reason string when cooldown or the daily cap applies."""
        last = self._last_signal_bar.get(symbol)
        if last is not None:
            elapsed = bars_since_signal(context.frame, last)
            if elapsed < self.cfg.cooldown_bars:
                return (
                    f"cooldown active: {elapsed} of {self.cfg.cooldown_bars} bars since the "
                    f"last signal"
                )
        used = self._daily_count.get((symbol, context.bar_time.date()), 0)
        if used >= self.cfg.max_signals_per_symbol_per_day:
            return f"daily cap reached ({used}/{self.cfg.max_signals_per_symbol_per_day})"
        return None

    def _find_trigger(
        self, snapshot: MarketSnapshot, side: SignalSide
    ) -> tuple[Trigger | None, TimeframeContext | None]:
        """Look for a trigger on each entry timeframe, primary first.

        The primary timeframe is checked first so that, all else equal, the trade
        is planned on the timeframe the configuration designates as the decision
        chart rather than on the fastest one available.
        """
        ordered = [snapshot.primary_timeframe] + [
            tf for tf in self.cfg.mtf.entry if tf != snapshot.primary_timeframe
        ]
        for timeframe in ordered:
            context = snapshot.get(timeframe)
            if context is None or not context.is_complete:
                continue
            trigger = self.triggers.detect(context, side)
            if trigger is not None:
                return trigger, context
        return None, None

    def _check_gates(
        self, context: TimeframeContext, side: SignalSide
    ) -> tuple[str, str] | None:
        """Run every enabled hard gate; return the first failure."""
        cfg = self.cfg
        indicators = self.settings.indicators
        structure_cfg = self.settings.structure

        if cfg.gate_enabled("adx_min"):
            adx = context.value("adx")
            if not np.isfinite(adx) or adx < indicators.adx_trend_min:
                return "adx_min", f"ADX {adx:.1f} below {indicators.adx_trend_min:.0f}"

        if cfg.gate_enabled("volume_confirmation") and not context.volume.confirms(side.value):
            state = context.volume
            return (
                "volume_confirmation",
                f"volume does not confirm: {state.relative:.2f}x average, flow bias {state.bias}",
            )

        if cfg.gate_enabled("atr_in_range"):
            natr = context.value("natr")
            if not np.isfinite(natr):
                return "atr_in_range", "ATR unavailable"
            if natr < indicators.atr_min_pct:
                return (
                    "atr_in_range",
                    f"ATR {natr:.3f}% of price is below the {indicators.atr_min_pct}% floor: "
                    f"the move cannot cover costs",
                )
            if natr > indicators.atr_max_pct:
                return (
                    "atr_in_range",
                    f"ATR {natr:.3f}% of price exceeds the {indicators.atr_max_pct}% ceiling",
                )
            if context.regime.volatility is VolatilityState.EXTREME:
                return (
                    "atr_in_range",
                    f"volatility anomaly: ATR is {context.regime.atr_ratio:.1f}x its median",
                )

        if cfg.gate_enabled("room_to_level"):
            room = context.structure.room_for(side.value)
            if room < structure_cfg.room_to_level_atr:
                level = (
                    context.structure.nearest_resistance
                    if side is SignalSide.LONG
                    else context.structure.nearest_support
                )
                price = f"{level.price:.6g}" if level else "n/a"
                return (
                    "room_to_level",
                    f"only {room:.2f} ATR to {'resistance' if side is SignalSide.LONG else 'support'} "
                    f"at {price}, minimum {structure_cfg.room_to_level_atr} ATR",
                )

        if cfg.gate_enabled("no_conflicting_structure"):
            structure = context.structure
            trend_opposes = structure.trend.bias == -side.value
            break_opposes = structure.break_event.kind.bias == -side.value
            if trend_opposes and break_opposes:
                return (
                    "no_conflicting_structure",
                    f"structure is {structure.trend.value} and the last break was "
                    f"{structure.break_event.kind.value}",
                )

        return None

    @staticmethod
    def _weakest(scorecard: ScoreCard, limit: int = 3) -> str:
        """Name the components that contributed least, for the rejection log."""
        available = [c for c in scorecard.components if c.available]
        available.sort(key=lambda c: c.weight * (1.0 - c.fraction), reverse=True)
        return ", ".join(f"{c.name} {c.fraction:.0%}" for c in available[:limit]) or "n/a"

    def _build_signal(
        self,
        snapshot: MarketSnapshot,
        context: TimeframeContext,
        side: SignalSide,
        scorecard: ScoreCard,
        mtf_verdict: MTFVerdict,
        trigger: Trigger | None,
        plan: TradePlan,
    ) -> Signal:
        """Assemble the final :class:`Signal` with its full evidence trail."""
        reasons = list(scorecard.reasons)
        if trigger is not None:
            reasons.insert(0, f"trigger: {trigger.detail}")
        reasons.append(f"MTF: {mtf_verdict.detail}")
        if scorecard.unavailable:
            reasons.append(f"not scored (no data): {', '.join(scorecard.unavailable)}")

        external = snapshot.external
        notes: dict[str, object] = {
            "regime": context.regime.regime.value,
            "trend_strength": context.regime.strength.value,
            "volatility": context.regime.volatility.value,
            "structure": context.structure.trend.value,
            "last_break": context.structure.break_event.kind.value,
            "volume": f"{context.volume.relative:.2f}x average",
            "volume_flow": f"{context.volume.pressure:+.0f}%",
            "adx": round(context.value("adx"), 1),
            "rsi": round(context.value("rsi"), 1),
            "natr_pct": round(context.value("natr"), 3),
            "stop_source": plan.stop_source,
            "rr_gross": round(plan.risk_reward_gross, 2),
            "objections": scorecard.objections,
            "mtf_scores": {tf: round(v, 2) for tf, v in mtf_verdict.scores.items()},
        }
        if external.funding is not None:
            notes["funding_pct"] = external.funding.rate_pct
        if external.sentiment is not None and np.isfinite(external.sentiment.fear_greed):
            notes["fear_greed"] = external.sentiment.fear_greed
        if external.unavailable:
            notes["data_gaps"] = external.unavailable

        signal = Signal(
            symbol=snapshot.symbol,
            timeframe=context.timeframe,
            side=side,
            strength=scorecard.strength(self.cfg),
            score=scorecard.total,
            entry=plan.entry,
            stop_loss=plan.stop_loss,
            take_profits=plan.take_profits,
            risk_reward=plan.risk_reward_net,
            atr=context.value("atr"),
            bar_time=context.bar_time,
            reasons=reasons,
            scorecard=scorecard,
            mtf=mtf_verdict,
            trigger=trigger.name if trigger else "none",
            notes=notes,
        )
        _log.debug("signal built: %s", signal.summary())
        return signal
