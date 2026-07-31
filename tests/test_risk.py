"""Risk: stop placement, sizing, circuit breakers, ledger arithmetic.

These are the tests that protect capital. Every one of them pins a behaviour
whose failure mode is losing more money than intended.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from analysis.structure import StructureReport
from config.settings import Settings
from risk.execution import resolve_position_on_bar, target_fraction_of_remaining
from risk.guards import TradingGuards
from risk.manager import HaltReason, RiskManager
from risk.planner import TradePlanner
from risk.portfolio import ExitReason, Portfolio
from strategies.base import Signal, SignalSide, SignalStrength

BAR = datetime(2024, 1, 1, tzinfo=UTC)


def make_signal(
    settings: Settings,
    side: SignalSide = SignalSide.LONG,
    entry: float = 100.0,
    atr: float = 1.0,
    structure: StructureReport | None = None,
) -> Signal:
    """Build a signal through the real planner, so levels are consistent."""
    plan = TradePlanner(settings.risk).plan(side, entry, atr, structure)
    assert plan.valid, plan.reason
    return Signal(
        symbol="BTC/USDT",
        timeframe="15m",
        side=side,
        strength=SignalStrength.STRONG,
        score=85.0,
        entry=plan.entry,
        stop_loss=plan.stop_loss,
        take_profits=plan.take_profits,
        risk_reward=plan.risk_reward_net,
        atr=atr,
        bar_time=BAR,
    )


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------
def test_atr_stop_is_used_without_structure(settings: Settings) -> None:
    plan = TradePlanner(settings.risk).plan(SignalSide.LONG, 100.0, 2.0)
    assert plan.stop_source == "atr"
    assert plan.stop_distance == pytest.approx(settings.risk.sl_atr_multiplier * 2.0)
    assert plan.stop_loss == pytest.approx(100.0 - plan.stop_distance)


def test_targets_sit_at_the_configured_r_multiples(settings: Settings) -> None:
    plan = TradePlanner(settings.risk).plan(SignalSide.LONG, 100.0, 1.0)
    for target, rr in zip(plan.take_profits, settings.risk.tp_rr_targets, strict=True):
        assert target.rr == rr
        assert target.price == pytest.approx(100.0 + rr * plan.stop_distance)


def test_short_plan_is_mirrored(settings: Settings) -> None:
    plan = TradePlanner(settings.risk).plan(SignalSide.SHORT, 100.0, 1.0)
    assert plan.stop_loss > 100.0
    assert all(target.price < 100.0 for target in plan.take_profits)


def test_net_rr_is_below_gross(settings: Settings) -> None:
    """Costs must reduce the reported reward-to-risk, never be ignored."""
    plan = TradePlanner(settings.risk).plan(SignalSide.LONG, 100.0, 1.0)
    assert plan.risk_reward_net < plan.risk_reward_gross
    assert plan.risk_reward_gross == pytest.approx(max(settings.risk.tp_rr_targets))

    cost = 100.0 * settings.risk.round_trip_cost_pct / 100.0
    expected = (plan.risk_reward_gross * plan.stop_distance - cost) / (plan.stop_distance + cost)
    assert plan.risk_reward_net == pytest.approx(expected)


def test_tight_stop_makes_costs_dominate(settings: Settings) -> None:
    """With a very tight stop, the net RR must collapse - that is the point."""
    wide = TradePlanner(settings.risk).plan(SignalSide.LONG, 100.0, atr=2.0)
    tight = TradePlanner(settings.risk).plan(SignalSide.LONG, 100.0, atr=0.02)
    assert tight.risk_reward_net < wide.risk_reward_net


def test_structure_stop_is_preferred_when_sensible(settings: Settings) -> None:
    structure = StructureReport(last_swing_low=98.0)
    plan = TradePlanner(settings.risk).plan(SignalSide.LONG, 100.0, 1.0, structure)
    assert plan.stop_source == "structure"
    # Swing plus the configured ATR buffer.
    expected = 100.0 - 98.0 + settings.risk.sl_structure_buffer_atr * 1.0
    assert plan.stop_distance == pytest.approx(expected)


def test_far_structure_stop_falls_back_to_atr(settings: Settings) -> None:
    """A swing far below price would destroy the RR budget, so ATR is used."""
    structure = StructureReport(last_swing_low=50.0)
    plan = TradePlanner(settings.risk).plan(SignalSide.LONG, 100.0, 1.0, structure)
    assert plan.stop_source == "atr_capped"
    assert plan.stop_distance == pytest.approx(settings.risk.sl_atr_multiplier * 1.0)


def test_stop_inside_the_noise_band_is_widened(settings: Settings) -> None:
    structure = StructureReport(last_swing_low=99.95)
    plan = TradePlanner(settings.risk).plan(SignalSide.LONG, 100.0, 1.0, structure)
    assert plan.stop_source == "atr_floor"
    assert plan.stop_distance == pytest.approx(0.5 * settings.risk.sl_atr_multiplier)


def test_stop_distance_is_capped_by_percent(settings: Settings) -> None:
    plan = TradePlanner(settings.risk).plan(SignalSide.LONG, 100.0, atr=50.0)
    assert plan.stop_distance == pytest.approx(100.0 * settings.risk.sl_max_pct / 100.0)
    assert "pct_capped" in plan.stop_source


def test_plan_is_invalid_without_atr(settings: Settings) -> None:
    plan = TradePlanner(settings.risk).plan(SignalSide.LONG, 100.0, float("nan"))
    assert plan.valid is False
    assert "ATR" in plan.reason


def test_swing_on_the_wrong_side_is_ignored(settings: Settings) -> None:
    """Price already below the swing low must not invert the stop distance."""
    structure = StructureReport(last_swing_low=105.0)
    plan = TradePlanner(settings.risk).plan(SignalSide.LONG, 100.0, 1.0, structure)
    assert plan.valid is True
    assert plan.stop_loss < 100.0
    assert plan.stop_source == "atr"


# ---------------------------------------------------------------------------
# Signal invariants
# ---------------------------------------------------------------------------
def test_signal_rejects_a_stop_on_the_wrong_side(settings: Settings) -> None:
    plan = TradePlanner(settings.risk).plan(SignalSide.LONG, 100.0, 1.0)
    with pytest.raises(ValueError, match="must be below entry"):
        Signal(
            symbol="X", timeframe="15m", side=SignalSide.LONG,
            strength=SignalStrength.NORMAL, score=70.0,
            entry=100.0, stop_loss=101.0, take_profits=plan.take_profits,
            risk_reward=2.0, atr=1.0, bar_time=BAR,
        )


def test_signal_requires_a_target() -> None:
    with pytest.raises(ValueError, match="at least one take-profit"):
        Signal(
            symbol="X", timeframe="15m", side=SignalSide.LONG,
            strength=SignalStrength.NORMAL, score=70.0,
            entry=100.0, stop_loss=99.0, take_profits=(),
            risk_reward=2.0, atr=1.0, bar_time=BAR,
        )


# ---------------------------------------------------------------------------
# Sizing
# ---------------------------------------------------------------------------
def test_risk_per_trade_is_honoured_exactly(settings: Settings) -> None:
    portfolio = Portfolio(1000.0)
    manager = RiskManager(settings.risk, portfolio)
    signal = make_signal(settings)

    decision = manager.approve(signal)
    assert decision.approved
    assert decision.risk_pct == pytest.approx(settings.risk.risk_per_trade_pct)
    assert decision.quantity == pytest.approx(
        1000.0 * settings.risk.risk_per_trade_pct / 100.0 / signal.risk_per_unit
    )


def test_a_wider_stop_gives_a_smaller_position(settings: Settings) -> None:
    """The defining property of risk-based sizing."""
    manager = RiskManager(settings.risk, Portfolio(1000.0))
    tight = manager.approve(make_signal(settings, atr=1.0))
    wide = manager.approve(make_signal(settings, atr=3.0))

    assert tight.quantity > wide.quantity
    assert tight.risk_amount == pytest.approx(wide.risk_amount)


def test_leverage_caps_the_notional(settings: Settings) -> None:
    tuned = settings.with_overrides({"risk.leverage": "1"})
    manager = RiskManager(tuned.risk, Portfolio(1000.0))
    # A 0.05% stop would otherwise imply a notional 20x the account.
    decision = manager.approve(make_signal(tuned, atr=0.02))
    assert decision.notional <= 1000.0 * tuned.risk.leverage + 1e-6
    assert "capped by leverage" in decision.reason or not decision.approved


def test_dust_trades_are_refused(settings: Settings) -> None:
    """A stop so tight that fees eat half the risk budget is not worth taking."""
    tuned = settings.with_overrides({"risk.fee_pct": "0.5", "risk.slippage_pct": "0.3"})
    manager = RiskManager(tuned.risk, Portfolio(1000.0))
    decision = manager.approve(make_signal(tuned, atr=0.01))
    assert decision.approved is False
    assert "round-trip cost" in decision.reason or "leverage" in decision.reason


def test_position_limits_are_enforced(settings: Settings) -> None:
    tuned = settings.with_overrides({"risk.max_open_positions": "1"})
    portfolio = Portfolio(1000.0)
    manager = RiskManager(tuned.risk, portfolio)

    first = make_signal(tuned)
    decision = manager.approve(first)
    portfolio.open_position(first, decision.quantity, timestamp=BAR)

    blocked = manager.approve(make_signal(tuned))
    assert blocked.approved is False
    assert "max_open_positions" in blocked.reason


def test_opposite_position_in_same_symbol_is_refused(settings: Settings) -> None:
    portfolio = Portfolio(1000.0)
    manager = RiskManager(settings.risk, portfolio)
    long_signal = make_signal(settings, SignalSide.LONG)
    portfolio.open_position(long_signal, manager.approve(long_signal).quantity, timestamp=BAR)

    decision = manager.approve(make_signal(settings, SignalSide.SHORT))
    assert decision.approved is False
    assert "opposite position" in decision.reason or "already holding" in decision.reason


def test_correlated_position_limit_is_enforced(settings: Settings) -> None:
    tuned = settings.with_overrides(
        {"risk.max_correlated_positions": "1", "risk.max_open_positions": "3"}
    )
    portfolio = Portfolio(1000.0)
    manager = RiskManager(tuned.risk, portfolio)

    first = make_signal(tuned)
    portfolio.open_position(first, manager.approve(first).quantity, timestamp=BAR)

    second = make_signal(tuned)
    second.symbol = "ETH/USDT"
    decision = manager.approve(second)
    assert decision.approved is False
    assert "point" in decision.reason  # "...positions already point BUY"


# ---------------------------------------------------------------------------
# Circuit breakers
# ---------------------------------------------------------------------------
def test_daily_loss_limit_halts_and_lifts_next_day(settings: Settings) -> None:
    tuned = settings.with_overrides({"risk.daily_loss_limit_pct": "3.0"})
    portfolio = Portfolio(1000.0)
    manager = RiskManager(tuned.risk, portfolio)

    portfolio.cash = 960.0  # -4% on the day
    assert manager.update_breakers() is HaltReason.DAILY_LOSS
    assert manager.is_halted
    assert manager.approve(make_signal(tuned)).approved is False

    manager.roll_day(BAR + timedelta(days=1))
    assert manager.is_halted is False


def test_max_drawdown_halt_is_permanent(settings: Settings) -> None:
    tuned = settings.with_overrides({"risk.max_drawdown_pct": "10.0"})
    portfolio = Portfolio(1000.0)
    manager = RiskManager(tuned.risk, portfolio)

    portfolio.cash = 850.0  # -15% from the peak
    assert manager.update_breakers() is HaltReason.MAX_DRAWDOWN

    manager.roll_day(BAR + timedelta(days=5))
    assert manager.is_halted is True, "a drawdown halt must survive the day boundary"


def test_open_losses_count_towards_the_breakers(settings: Settings) -> None:
    """Realised PnL alone would let an open loser sail past the limit."""
    tuned = settings.with_overrides({"risk.daily_loss_limit_pct": "3.0"})
    portfolio = Portfolio(1000.0)
    manager = RiskManager(tuned.risk, portfolio)

    signal = make_signal(tuned, entry=100.0, atr=1.0)
    quantity = manager.approve(signal).quantity
    portfolio.open_position(signal, quantity, timestamp=BAR)

    # Mark the position far below entry: unrealised, but real.
    halt = manager.update_breakers({"BTC/USDT": 60.0})
    assert halt is not HaltReason.NONE


def test_consecutive_losses_halt(settings: Settings) -> None:
    tuned = settings.with_overrides({"risk.max_consecutive_losses": "2"})
    manager = RiskManager(tuned.risk, Portfolio(1000.0))
    manager.state.consecutive_losses = 2
    assert manager.update_breakers() is HaltReason.CONSECUTIVE_LOSSES


# ---------------------------------------------------------------------------
# In-trade management
# ---------------------------------------------------------------------------
def test_breakeven_then_trailing(settings: Settings) -> None:
    portfolio = Portfolio(1000.0)
    manager = RiskManager(settings.risk, portfolio)
    signal = make_signal(settings, entry=100.0, atr=1.0)
    position = portfolio.open_position(signal, 1.0, timestamp=BAR)
    original_stop = position.stop_loss

    manager.manage(position, high=100.5, low=100.0, atr=1.0)
    assert position.stop_loss == original_stop, "nothing should move below 1R"

    manager.manage(position, high=102.5, low=101.0, atr=1.0)
    assert position.breakeven_done is True
    assert position.stop_loss > position.entry_price

    before = position.stop_loss
    manager.manage(position, high=106.0, low=104.0, atr=1.0)
    assert position.trailing_active is True
    assert position.stop_loss > before


def test_stop_never_moves_against_the_position(settings: Settings) -> None:
    portfolio = Portfolio(1000.0)
    manager = RiskManager(settings.risk, portfolio)
    signal = make_signal(settings, entry=100.0, atr=1.0)
    position = portfolio.open_position(signal, 1.0, timestamp=BAR)

    manager.manage(position, high=110.0, low=100.0, atr=1.0)
    high_water_stop = position.stop_loss
    # Price falls back: the stop must hold, not follow it down.
    manager.manage(position, high=101.0, low=100.5, atr=1.0)
    assert position.stop_loss == high_water_stop


def test_r_multiple_uses_the_original_risk_after_breakeven(settings: Settings) -> None:
    portfolio = Portfolio(1000.0)
    manager = RiskManager(settings.risk, portfolio)
    signal = make_signal(settings, entry=100.0, atr=1.0)
    position = portfolio.open_position(signal, 1.0, timestamp=BAR)
    original_risk_per_unit = position.risk_per_unit

    manager.manage(position, high=103.0, low=100.0, atr=1.0)
    assert position.risk_per_unit == pytest.approx(original_risk_per_unit), (
        "moving the stop must not rescale R, or trades stop being comparable"
    )


# ---------------------------------------------------------------------------
# Ledger
# ---------------------------------------------------------------------------
def test_slippage_always_hurts() -> None:
    portfolio = Portfolio(1000.0, fee_pct=0.0, slippage_pct=0.1)
    # Entering long pays up; exiting long sells lower.
    assert portfolio.fill_price(100.0, SignalSide.LONG, entering=True) > 100.0
    assert portfolio.fill_price(100.0, SignalSide.LONG, entering=False) < 100.0
    # Entering short sells lower; exiting short buys higher.
    assert portfolio.fill_price(100.0, SignalSide.SHORT, entering=True) < 100.0
    assert portfolio.fill_price(100.0, SignalSide.SHORT, entering=False) > 100.0


def test_a_stopped_out_trade_loses_slightly_more_than_1r(settings: Settings) -> None:
    """Costs push the realised loss beyond the nominal stop distance."""
    portfolio = Portfolio(1000.0, settings.risk.fee_pct, settings.risk.slippage_pct)
    manager = RiskManager(settings.risk, portfolio)
    signal = make_signal(settings, entry=100.0, atr=1.0)
    position = portfolio.open_position(signal, manager.approve(signal).quantity, timestamp=BAR)

    trade = portfolio.close(position, signal.stop_loss, ExitReason.STOP_LOSS, timestamp=BAR)
    assert trade is not None
    assert trade.r_multiple < -1.0
    assert trade.r_multiple > -1.3, "costs should be a small penalty, not a doubling"


def test_partial_exit_then_full_close(settings: Settings) -> None:
    portfolio = Portfolio(1000.0, 0.0, 0.0)
    signal = make_signal(settings, entry=100.0, atr=1.0)
    position = portfolio.open_position(signal, 2.0, timestamp=BAR)

    assert portfolio.close(position, 103.0, ExitReason.TAKE_PROFIT, timestamp=BAR, fraction=0.5) is None
    assert position.remaining == pytest.approx(1.0)

    trade = portfolio.close(position, 104.0, ExitReason.TAKE_PROFIT, timestamp=BAR)
    assert trade is not None
    assert trade.quantity == pytest.approx(2.0), "the record reports the original size"
    assert trade.pnl == pytest.approx(0.5 * 2.0 * 3.0 + 0.5 * 2.0 * 4.0)


def test_target_fraction_conversion(settings: Settings) -> None:
    """tp_split is a share of the original size, close() takes a share of the rest."""
    portfolio = Portfolio(1000.0, 0.0, 0.0)
    signal = make_signal(settings, entry=100.0, atr=1.0)
    position = portfolio.open_position(signal, 4.0, timestamp=BAR)

    assert target_fraction_of_remaining(position, 0.5) == pytest.approx(0.5)
    portfolio.close(position, 103.0, ExitReason.TAKE_PROFIT, timestamp=BAR, fraction=0.5)
    # 2 of the original 4 remain; taking another "50% of original" is now all of it.
    assert target_fraction_of_remaining(position, 0.5) == pytest.approx(1.0)


def test_equity_marks_open_positions(settings: Settings) -> None:
    portfolio = Portfolio(1000.0, 0.0, 0.0)
    signal = make_signal(settings, entry=100.0, atr=1.0)
    portfolio.open_position(signal, 1.0, timestamp=BAR)

    assert portfolio.equity({"BTC/USDT": 110.0}) == pytest.approx(1010.0)
    # Without a mark, the position is valued flat rather than guessed.
    assert portfolio.equity() == pytest.approx(1000.0)


def test_state_round_trip_preserves_management(settings: Settings) -> None:
    """A restart must not forget a moved stop or a taken target."""
    portfolio = Portfolio(1000.0, 0.0, 0.0)
    manager = RiskManager(settings.risk, portfolio)
    signal = make_signal(settings, entry=100.0, atr=1.0)
    position = portfolio.open_position(signal, 2.0, timestamp=BAR)
    manager.manage(position, high=104.0, low=100.0, atr=1.0)
    position.targets_hit = 1

    state = portfolio.export_state()
    restored = Portfolio(1000.0, 0.0, 0.0)
    assert restored.restore_state(state) == 1

    revived = next(iter(restored.positions.values()))
    assert revived.stop_loss == pytest.approx(position.stop_loss)
    assert revived.targets_hit == 1
    assert revived.breakeven_done == position.breakeven_done
    assert revived.best_price == pytest.approx(position.best_price)
    assert revived.take_profits == position.take_profits


# ---------------------------------------------------------------------------
# Bar resolution
# ---------------------------------------------------------------------------
def test_pessimistic_resolution_takes_the_stop_first(settings: Settings) -> None:
    """When a bar spans both levels, the loss is assumed."""
    portfolio = Portfolio(1000.0, 0.0, 0.0)
    manager = RiskManager(settings.risk, portfolio)
    signal = make_signal(settings, entry=100.0, atr=1.0)
    position = portfolio.open_position(signal, 1.0, timestamp=BAR)

    outcome = resolve_position_on_bar(
        portfolio, manager, position,
        timestamp=BAR, high=110.0, low=signal.stop_loss - 1.0, atr=1.0, pessimistic=True,
    )
    assert outcome.closed
    assert outcome.exit_reason is ExitReason.STOP_LOSS


def test_optimistic_resolution_takes_the_target_first(settings: Settings) -> None:
    portfolio = Portfolio(1000.0, 0.0, 0.0)
    manager = RiskManager(settings.risk, portfolio)
    signal = make_signal(settings, entry=100.0, atr=1.0)
    position = portfolio.open_position(signal, 1.0, timestamp=BAR)

    outcome = resolve_position_on_bar(
        portfolio, manager, position,
        timestamp=BAR, high=signal.take_profits[0].price + 1.0,
        low=signal.stop_loss - 1.0, atr=1.0, pessimistic=False,
    )
    assert outcome.exit_reason in {ExitReason.TAKE_PROFIT, ExitReason.STOP_LOSS}


def test_only_one_target_is_taken_per_bar(settings: Settings) -> None:
    portfolio = Portfolio(1000.0, 0.0, 0.0)
    manager = RiskManager(settings.risk, portfolio)
    signal = make_signal(settings, entry=100.0, atr=1.0)
    position = portfolio.open_position(signal, 2.0, timestamp=BAR)

    # A bar reaching past BOTH targets must still only fill the first.
    outcome = resolve_position_on_bar(
        portfolio, manager, position,
        timestamp=BAR, high=signal.take_profits[-1].price + 5.0, low=100.0, atr=1.0,
    )
    assert outcome.partial is True
    assert position.targets_hit == 1
    assert position.is_open


# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------
def test_wide_spread_blocks_trading(settings: Settings, builder, ohlcv) -> None:
    from data.models import ExternalContext, Ticker

    context = builder.build("BTC/USDT", "15m", builder.enrich(ohlcv, "15m"))
    guards = TradingGuards(settings.risk)

    wide = ExternalContext(
        symbol="BTC/USDT", ticker=Ticker("BTC/USDT", last=100.0, bid=99.0, ask=101.0)
    )
    result = guards.check("BTC/USDT", context, wide)
    assert result.ok is False
    assert result.blocked_by == "spread"


def test_unmeasurable_spread_warns_but_does_not_block(
    settings: Settings, builder, ohlcv
) -> None:
    from data.models import ExternalContext, Ticker

    context = builder.build("BTC/USDT", "15m", builder.enrich(ohlcv, "15m"))
    guards = TradingGuards(settings.risk)
    result = guards.check(
        "BTC/USDT", context, ExternalContext("BTC/USDT", ticker=Ticker("BTC/USDT", last=100.0))
    )
    assert result.ok is True
    assert any("unmeasurable" in warning for warning in result.warnings)


def test_extreme_funding_blocks_trading(settings: Settings, builder, ohlcv) -> None:
    from data.models import ExternalContext, FundingSnapshot, Ticker

    context = builder.build("BTC/USDT", "15m", builder.enrich(ohlcv, "15m"))
    guards = TradingGuards(settings.risk)
    external = ExternalContext(
        symbol="BTC/USDT",
        ticker=Ticker("BTC/USDT", last=100.0, bid=99.99, ask=100.01),
        funding=FundingSnapshot("BTC/USDT", rate_pct=5.0),
    )
    result = guards.check("BTC/USDT", context, external)
    assert result.ok is False
    assert result.blocked_by == "funding_extreme"


def test_blocked_hours_are_respected(settings: Settings, builder, ohlcv) -> None:
    from data.models import ExternalContext, Ticker

    tuned = settings.with_overrides({"risk.blocked_hours_utc": [3]})
    context = builder.build("BTC/USDT", "15m", builder.enrich(ohlcv, "15m"))
    guards = TradingGuards(tuned.risk)
    result = guards.check(
        "BTC/USDT",
        context,
        ExternalContext("BTC/USDT", ticker=Ticker("BTC/USDT", 100.0, 99.99, 100.01)),
        now=datetime(2024, 5, 1, 3, 30, tzinfo=UTC),
    )
    assert result.ok is False
    assert result.blocked_by == "trading_hours"
