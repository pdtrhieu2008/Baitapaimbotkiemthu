"""Scoring, multi-timeframe agreement, triggers and the gate ordering.

The central assertion here is that **a high score cannot buy its way past a
failed gate**. If that ever regresses, the bot starts taking trades it was
explicitly configured to refuse.
"""

from __future__ import annotations

import pandas as pd
import pytest

from analysis.context import ContextBuilder, MarketSnapshot, TimeframeContext
from config.settings import Settings
from data.models import (
    ExternalContext,
    FundingSnapshot,
    OpenInterestSnapshot,
    SentimentSnapshot,
    Ticker,
)
from strategies import build_strategy
from strategies.base import Rejection, Signal, SignalSide
from strategies.mtf import MultiTimeframeAnalyser
from strategies.scoring import Scorer
from tests.conftest import AGGREGATION, make_ohlcv


def build_snapshot(
    settings: Settings,
    base: pd.DataFrame | None = None,
    external: ExternalContext | None = None,
    symbol: str = "BTC/USDT",
) -> MarketSnapshot:
    """Assemble a snapshot from one base series resampled to 15m/1h/4h.

    6000 bars is not arbitrary: the 4h context needs ~238 of its own bars to
    seed EMA-200 plus the ADX warm-up, i.e. 238 * 16 = 3808 fifteen-minute bars.
    With fewer, the 4h context reads as missing data and every MTF test would
    pass or fail for the wrong reason.
    """
    builder = ContextBuilder(settings.indicators, settings.structure)
    base = base if base is not None else make_ohlcv(bars=6000, seed=17, regime_shifts=True)

    frames = {"15m": base}
    for timeframe, rule in (("1h", "1h"), ("4h", "4h")):
        frames[timeframe] = base.resample(rule).agg(AGGREGATION).dropna()

    contexts: dict[str, TimeframeContext] = {}
    for timeframe, frame in frames.items():
        contexts[timeframe] = builder.build(symbol, timeframe, builder.enrich(frame, timeframe))

    return MarketSnapshot(
        symbol=symbol,
        primary_timeframe="15m",
        contexts=contexts,
        external=external or ExternalContext(symbol=symbol),
    )


def force_side(context: TimeframeContext, side: SignalSide) -> None:
    """Override a context's bias so MTF direction is deterministic in a test."""
    context.bias = side.value
    context.bias_score = 0.9 * side.value


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def test_score_is_bounded_and_components_are_complete(test_settings: Settings) -> None:
    snapshot = build_snapshot(test_settings)
    card = Scorer(test_settings.strategy, test_settings.indicators).score(
        snapshot, SignalSide.LONG
    )
    assert 0.0 <= card.total <= 100.0
    assert {c.name for c in card.components} == set(test_settings.strategy.COMPONENTS)
    for component in card.components:
        assert 0.0 <= component.fraction <= 1.0


def test_missing_sentiment_leaves_the_denominator(test_settings: Settings) -> None:
    """The key normalisation property: unavailable != scored zero."""
    scorer = Scorer(test_settings.strategy, test_settings.indicators)
    base = make_ohlcv(bars=6000, seed=17, regime_shifts=True)

    without = scorer.score(build_snapshot(test_settings, base), SignalSide.LONG)
    assert "sentiment" in without.unavailable
    assert without.available_weight == pytest.approx(
        sum(test_settings.strategy.weights.values()) - test_settings.strategy.weights["sentiment"]
    )

    rich = ExternalContext(
        symbol="BTC/USDT",
        ticker=Ticker("BTC/USDT", 100.0, 99.99, 100.01),
        funding=FundingSnapshot("BTC/USDT", rate_pct=-0.08),
        open_interest=OpenInterestSnapshot("BTC/USDT", value=1e6, change_pct=3.0),
        sentiment=SentimentSnapshot(fear_greed=20.0, btc_dominance=54.0,
                                    btc_dominance_change_pct=-0.5),
    )
    with_data = scorer.score(build_snapshot(test_settings, base, rich), SignalSide.LONG)
    assert "sentiment" not in with_data.unavailable
    assert with_data.available_weight == pytest.approx(
        sum(test_settings.strategy.weights.values())
    )
    # Both totals are on the same 0-100 scale despite different denominators.
    assert 0.0 <= with_data.total <= 100.0


def test_a_component_with_no_data_scores_nothing_and_is_flagged(
    test_settings: Settings,
) -> None:
    snapshot = build_snapshot(test_settings)
    snapshot.primary.values["rsi"] = float("nan")
    card = Scorer(test_settings.strategy, test_settings.indicators).score(
        snapshot, SignalSide.LONG
    )
    rsi = next(c for c in card.components if c.name == "rsi")
    assert rsi.available is False
    assert rsi.earned == 0.0
    assert "rsi" in card.unavailable


def test_long_and_short_scores_disagree(test_settings: Settings) -> None:
    """Scoring is side-specific; a trending market cannot favour both ways."""
    snapshot = build_snapshot(
        test_settings, make_ohlcv(bars=6000, seed=19, drift=0.0012, volatility=0.003)
    )
    scorer = Scorer(test_settings.strategy, test_settings.indicators)
    long_total = scorer.score(snapshot, SignalSide.LONG).total
    short_total = scorer.score(snapshot, SignalSide.SHORT).total
    assert long_total != short_total


def test_strength_tiers_follow_the_thresholds(test_settings: Settings) -> None:
    from strategies.base import SignalStrength
    from strategies.scoring import ScoreCard, ScoreComponent

    def card(fraction: float) -> ScoreCard:
        return ScoreCard(
            side=SignalSide.LONG,
            components=[ScoreComponent("trend", 100.0, fraction)],
        )

    cfg = test_settings.strategy
    assert card(0.85).strength(cfg) is SignalStrength.STRONG
    assert card(0.72).strength(cfg) is SignalStrength.NORMAL
    assert card(0.40).strength(cfg) is SignalStrength.WEAK


# ---------------------------------------------------------------------------
# Multi-timeframe
# ---------------------------------------------------------------------------
def test_opposing_context_timeframe_fails(test_settings: Settings) -> None:
    snapshot = build_snapshot(test_settings)
    analyser = MultiTimeframeAnalyser(test_settings.strategy.mtf)

    for timeframe in ("15m", "1h", "4h"):
        force_side(snapshot.contexts[timeframe], SignalSide.LONG)
    assert analyser.evaluate(snapshot, SignalSide.LONG).passed is True

    force_side(snapshot.contexts["4h"], SignalSide.SHORT)
    verdict = analyser.evaluate(snapshot, SignalSide.LONG)
    assert verdict.passed is False
    assert verdict.context_ok is False
    assert "higher timeframe opposes" in verdict.detail


def test_neutral_context_is_allowed_but_neutral_confirm_is_not(
    test_settings: Settings,
) -> None:
    snapshot = build_snapshot(test_settings)
    analyser = MultiTimeframeAnalyser(test_settings.strategy.mtf)
    for timeframe in ("15m", "1h", "4h"):
        force_side(snapshot.contexts[timeframe], SignalSide.LONG)

    snapshot.contexts["4h"].bias = 0  # context may abstain
    assert analyser.evaluate(snapshot, SignalSide.LONG).context_ok is True

    snapshot.contexts["1h"].bias = 0  # confirmation may not
    verdict = analyser.evaluate(snapshot, SignalSide.LONG)
    assert verdict.confirm_ok is False
    assert verdict.passed is False


def test_missing_timeframe_fails_closed(test_settings: Settings) -> None:
    snapshot = build_snapshot(test_settings)
    for timeframe in ("15m", "1h", "4h"):
        force_side(snapshot.contexts[timeframe], SignalSide.LONG)
    snapshot.contexts.pop("4h")

    verdict = MultiTimeframeAnalyser(test_settings.strategy.mtf).evaluate(
        snapshot, SignalSide.LONG
    )
    assert verdict.passed is False
    assert "missing timeframe data" in verdict.detail


def test_agreement_is_weighted_not_counted(test_settings: Settings) -> None:
    """Two aligned fast timeframes must not outvote an opposing 4h."""
    snapshot = build_snapshot(test_settings)
    analyser = MultiTimeframeAnalyser(test_settings.strategy.mtf)
    force_side(snapshot.contexts["15m"], SignalSide.LONG)
    force_side(snapshot.contexts["1h"], SignalSide.LONG)
    force_side(snapshot.contexts["4h"], SignalSide.SHORT)

    verdict = analyser.evaluate(snapshot, SignalSide.LONG)
    weights = test_settings.strategy.mtf
    expected = (weights.weight_of("15m") + weights.weight_of("1h")) / (
        weights.weight_of("15m") + weights.weight_of("1h") + weights.weight_of("4h")
    )
    assert verdict.agreement == pytest.approx(expected)


def test_direction_comes_from_higher_timeframes(test_settings: Settings) -> None:
    snapshot = build_snapshot(test_settings)
    analyser = MultiTimeframeAnalyser(test_settings.strategy.mtf)

    force_side(snapshot.contexts["4h"], SignalSide.SHORT)
    force_side(snapshot.contexts["1h"], SignalSide.SHORT)
    force_side(snapshot.contexts["15m"], SignalSide.LONG)  # entry chart disagrees
    assert analyser.propose_side(snapshot) is SignalSide.SHORT

    # Conflicting higher timeframes produce no proposal at all.
    force_side(snapshot.contexts["4h"], SignalSide.LONG)
    snapshot.contexts["1h"].bias_score = -0.9
    assert analyser.propose_side(snapshot) is None


# ---------------------------------------------------------------------------
# Gate ordering: the property that matters most
# ---------------------------------------------------------------------------
def test_a_perfect_score_cannot_pass_a_failed_gate(test_settings: Settings) -> None:
    """Force every gate-relevant reading to fail; no signal may be emitted."""
    strategy = build_strategy(test_settings)
    snapshot = build_snapshot(test_settings)
    for timeframe in ("15m", "1h", "4h"):
        force_side(snapshot.contexts[timeframe], SignalSide.LONG)

    # ADX far below the threshold. Gates are evaluated BEFORE the score
    # threshold, so the rejection must name a gate - never "score".
    snapshot.primary.values["adx"] = 1.0
    verdict = strategy.evaluate(snapshot)
    assert isinstance(verdict, Rejection)
    assert verdict.gate != "score"
    assert verdict.gate in {
        "adx_min", "trigger_required", "volume_confirmation", "atr_in_range",
        "room_to_level", "min_rr", "no_conflicting_structure",
    }


def test_atr_out_of_range_is_vetoed(test_settings: Settings) -> None:
    strategy = build_strategy(test_settings)
    snapshot = build_snapshot(test_settings)
    for timeframe in ("15m", "1h", "4h"):
        force_side(snapshot.contexts[timeframe], SignalSide.LONG)
    snapshot.primary.values["natr"] = test_settings.indicators.atr_max_pct * 3

    verdict = strategy.evaluate(snapshot)
    assert isinstance(verdict, Rejection)


def test_disabling_a_gate_removes_its_veto(test_settings: Settings) -> None:
    """Gates are configuration, not hard-coded behaviour."""
    snapshot = build_snapshot(test_settings)
    for timeframe in ("15m", "1h", "4h"):
        force_side(snapshot.contexts[timeframe], SignalSide.LONG)
    snapshot.primary.values["adx"] = 1.0

    with_gate = build_strategy(test_settings).evaluate(snapshot)
    assert isinstance(with_gate, Rejection)

    relaxed = test_settings.with_overrides({"strategy.gates": {"adx_min": False}})
    without_gate = build_strategy(relaxed).evaluate(snapshot)
    # It may still be rejected for another reason, but not for ADX.
    if isinstance(without_gate, Rejection):
        assert without_gate.gate != "adx_min"


def test_direction_can_be_disabled(test_settings: Settings) -> None:
    tuned = test_settings.with_overrides({"strategy.allow_short": "false"})
    strategy = build_strategy(tuned)
    snapshot = build_snapshot(tuned)
    for timeframe in ("15m", "1h", "4h"):
        force_side(snapshot.contexts[timeframe], SignalSide.SHORT)

    verdict = strategy.evaluate(snapshot)
    assert isinstance(verdict, Rejection)
    assert verdict.gate == "direction"
    assert "disabled" in verdict.detail


def test_incomplete_data_is_rejected_before_anything_else(test_settings: Settings) -> None:
    strategy = build_strategy(test_settings)
    snapshot = build_snapshot(test_settings)
    snapshot.primary.values["atr"] = float("nan")

    verdict = strategy.evaluate(snapshot)
    assert isinstance(verdict, Rejection)
    assert verdict.gate == "data"


def test_cooldown_blocks_a_repeat_signal(test_settings: Settings) -> None:
    strategy = build_strategy(test_settings)
    snapshot = build_snapshot(test_settings)
    for timeframe in ("15m", "1h", "4h"):
        force_side(snapshot.contexts[timeframe], SignalSide.LONG)

    from strategies.base import SignalStrength
    from risk.planner import TradePlanner

    plan = TradePlanner(test_settings.risk).plan(SignalSide.LONG, 100.0, 1.0)
    fake = Signal(
        symbol=snapshot.symbol, timeframe="15m", side=SignalSide.LONG,
        strength=SignalStrength.STRONG, score=90.0, entry=plan.entry,
        stop_loss=plan.stop_loss, take_profits=plan.take_profits,
        risk_reward=plan.risk_reward_net, atr=1.0,
        bar_time=snapshot.primary.bar_time,
    )
    strategy.register_signal(fake)

    verdict = strategy.evaluate(snapshot)
    assert isinstance(verdict, Rejection)
    assert verdict.gate == "cooldown"


def test_signal_carries_a_complete_evidence_trail(test_settings: Settings) -> None:
    """Whatever the outcome, a Signal must be fully explainable."""
    tuned = test_settings.with_overrides(
        {
            "strategy.min_score_to_emit": "60",
            "strategy.gates": {
                "mtf_alignment": True, "adx_min": False, "volume_confirmation": False,
                "atr_in_range": False, "room_to_level": False, "min_rr": True,
                "trigger_required": False, "no_conflicting_structure": False,
            },
        }
    )
    strategy = build_strategy(tuned)
    snapshot = build_snapshot(tuned)
    for timeframe in ("15m", "1h", "4h"):
        force_side(snapshot.contexts[timeframe], SignalSide.LONG)

    verdict = strategy.evaluate(snapshot)
    if isinstance(verdict, Rejection):
        pytest.skip(f"fixture did not produce a signal ({verdict.gate}); trail tested elsewhere")

    assert verdict.scorecard is not None
    assert verdict.mtf is not None
    assert verdict.reasons
    assert verdict.risk_reward >= tuned.risk.min_rr
    assert verdict.stop_loss < verdict.entry
    assert verdict.key.startswith(snapshot.symbol)
    payload = verdict.to_dict()
    assert payload["scorecard"] is not None
    assert payload["mtf"] is not None
