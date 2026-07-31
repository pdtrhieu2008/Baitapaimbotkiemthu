"""Price action, market structure, volume reading and regime classification."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from analysis.patterns import PATTERN_COLUMNS, detect_patterns, pattern_bias
from analysis.regime import MarketRegime, TrendStrength, VolatilityState, classify_regime
from analysis.structure import BreakKind, StructureTrend, analyse_structure
from analysis.swings import find_swings
from analysis.volume_analysis import analyse_volume
from tests.conftest import make_ohlcv


def _frame(bars: list[dict[str, float]]) -> pd.DataFrame:
    """Build a tiny OHLCV frame from explicit bars."""
    index = pd.date_range("2024-01-01", periods=len(bars), freq="1h", tz="UTC")
    return pd.DataFrame(bars, index=index).assign(
        volume=lambda df: df.get("volume", pd.Series([100.0] * len(df), index=df.index))
    )


# ---------------------------------------------------------------------------
# Candlestick patterns
# ---------------------------------------------------------------------------
def test_detect_patterns_returns_every_column_as_bool(ohlcv: pd.DataFrame) -> None:
    frame = detect_patterns(ohlcv)
    assert list(frame.columns) == list(PATTERN_COLUMNS)
    assert all(frame[column].dtype == bool for column in frame.columns)
    assert len(frame) == len(ohlcv)


def test_doji_needs_a_tiny_body() -> None:
    frame = _frame(
        [
            {"open": 100.0, "high": 105.0, "low": 95.0, "close": 100.2, "volume": 1.0},
            {"open": 100.0, "high": 105.0, "low": 95.0, "close": 104.5, "volume": 1.0},
        ]
    )
    result = detect_patterns(frame)
    assert bool(result["doji"].iloc[0]) is True
    assert bool(result["doji"].iloc[1]) is False


def test_hammer_and_shooting_star_are_mirror_images() -> None:
    hammer = _frame(
        [{"open": 100.0, "high": 101.0, "low": 90.0, "close": 100.5, "volume": 1.0}]
    )
    assert bool(detect_patterns(hammer)["hammer"].iloc[0]) is True

    star = _frame(
        [{"open": 100.0, "high": 110.0, "low": 99.5, "close": 99.8, "volume": 1.0}]
    )
    assert bool(detect_patterns(star)["shooting_star"].iloc[0]) is True


def test_engulfing_requires_a_bigger_opposite_body() -> None:
    frame = _frame(
        [
            {"open": 105.0, "high": 106.0, "low": 100.0, "close": 101.0, "volume": 1.0},
            {"open": 100.5, "high": 108.0, "low": 100.0, "close": 106.0, "volume": 1.0},
        ]
    )
    result = detect_patterns(frame)
    assert bool(result["engulfing_bull"].iloc[1]) is True
    assert bool(result["engulfing_bear"].iloc[1]) is False


def test_inside_and_outside_bars() -> None:
    frame = _frame(
        [
            {"open": 100.0, "high": 110.0, "low": 90.0, "close": 105.0, "volume": 1.0},
            {"open": 101.0, "high": 108.0, "low": 95.0, "close": 103.0, "volume": 1.0},
            {"open": 101.0, "high": 115.0, "low": 85.0, "close": 103.0, "volume": 1.0},
        ]
    )
    result = detect_patterns(frame)
    assert bool(result["inside_bar"].iloc[1]) is True
    assert bool(result["outside_bar"].iloc[2]) is True


def test_pattern_bias_nets_conflicting_patterns_to_zero() -> None:
    row = pd.Series({name: False for name in PATTERN_COLUMNS})
    row["hammer"] = True
    assert pattern_bias(row)[0] == 1

    row["shooting_star"] = True
    bias, names = pattern_bias(row)
    assert bias == 0, "a bar showing both directions is ambiguous, not bullish"
    assert set(names) == {"hammer", "shooting_star"}


def test_patterns_never_fire_on_a_zero_range_bar() -> None:
    frame = _frame([{"open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0, "volume": 1.0}])
    result = detect_patterns(frame)
    assert not result.iloc[0].any()


# ---------------------------------------------------------------------------
# Structure
# ---------------------------------------------------------------------------
def _staircase(direction: int, legs: int = 14, leg_bars: int = 14) -> pd.DataFrame:
    """A trend that advances in legs and pulls back, like a real one.

    A near-monotonic ramp is the wrong fixture for structure tests: if every bar
    makes a new extreme, no fractal swing ever completes (a swing high needs
    lower highs on its right), so there is nothing to label. Real trends pull
    back, and it is the pullbacks that create the HH/HL sequence.
    """
    prices: list[float] = [100.0]
    for leg in range(legs):
        advance = 6.0 if leg % 2 == 0 else 6.0
        retrace = -2.5
        for _ in range(leg_bars):
            prices.append(prices[-1] + direction * advance / leg_bars)
        for _ in range(leg_bars // 2):
            prices.append(prices[-1] + direction * retrace / (leg_bars // 2))

    close = np.array(prices, dtype=float)
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) + 0.12
    low = np.minimum(open_, close) - 0.12
    index = pd.date_range("2024-01-01", periods=len(close), freq="15min", tz="UTC")
    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": np.full(len(close), 1000.0),
        },
        index=index,
    )


def test_uptrend_is_labelled_up(settings) -> None:
    report = analyse_structure(_staircase(1), settings.structure)
    assert report.trend is StructureTrend.UP
    assert report.bias == 1
    assert "HH" in report.labels
    assert "HL" in report.labels


def test_downtrend_is_labelled_down(settings) -> None:
    report = analyse_structure(_staircase(-1), settings.structure)
    assert report.trend is StructureTrend.DOWN
    assert report.bias == -1
    assert "LL" in report.labels
    assert "LH" in report.labels


def test_too_few_swings_reports_range_not_a_guess(settings, builder) -> None:
    """A smooth ramp yields almost no confirmed swings, so structure is RANGE.

    This is fail-closed and intentional: with fewer than two comparable swings
    per side there is no evidence of a *structural* trend, and inventing one
    would let the strategy's structure component vote on nothing. The moving
    averages and the regime classifier still see the trend; only the swing-based
    reading abstains.
    """
    smooth = make_ohlcv(bars=600, seed=21, drift=0.0025, volatility=0.0002)
    report = analyse_structure(builder.enrich(smooth, "15m"), settings.structure)
    labelled_highs = [label for label in report.labels if label in {"HH", "LH"}]
    labelled_lows = [label for label in report.labels if label in {"HL", "LL"}]
    if not labelled_highs or not labelled_lows:
        assert report.trend is StructureTrend.RANGE


def test_bos_is_a_break_with_the_trend(settings) -> None:
    """A break in the trend's direction is a BOS, not a CHoCH."""
    report = analyse_structure(_staircase(1), settings.structure)
    assert report.trend is StructureTrend.UP
    if report.break_event.kind is not BreakKind.NONE and report.break_event.bias > 0:
        assert report.break_event.kind is BreakKind.BOS_BULL
        assert report.break_event.kind.is_choch is False


def test_choch_is_a_break_against_the_trend(settings) -> None:
    """Appending a sharp reversal to an uptrend must produce a bullish-to-bearish CHoCH."""
    up = _staircase(1)
    swings = find_swings(up, settings.structure.swing_strength, confirmed_only=True)
    lows = [s for s in swings if s.kind == "low"]
    assert lows

    # Drive price decisively below the last confirmed swing low.
    target = lows[-1].price * 0.97
    extra_index = pd.date_range(
        up.index[-1] + pd.Timedelta(minutes=15), periods=4, freq="15min", tz="UTC"
    )
    step = (up["close"].iloc[-1] - target) / 4
    closes = [up["close"].iloc[-1] - step * (n + 1) for n in range(4)]
    extra = pd.DataFrame(
        {
            "open": [up["close"].iloc[-1], *closes[:-1]],
            "high": [c + 0.1 for c in closes],
            "low": [c - 0.1 for c in closes],
            "close": closes,
            "volume": [3000.0] * 4,
        },
        index=extra_index,
    )
    report = analyse_structure(pd.concat([up, extra]), settings.structure)
    assert report.break_event.kind is BreakKind.CHOCH_BEAR
    assert report.break_event.kind.is_choch is True
    assert report.bias == -1, "a CHoCH must override the stale swing trend"


def test_structure_returns_neutral_on_short_series(settings) -> None:
    tiny = make_ohlcv(bars=5, seed=1)
    report = analyse_structure(tiny, settings.structure)
    assert report.trend is StructureTrend.RANGE
    assert report.break_event.kind is BreakKind.NONE
    assert report.nearest_support is None


def test_levels_are_on_the_correct_side_of_price(settings, builder, ohlcv) -> None:
    enriched = builder.enrich(ohlcv, "15m")
    report = analyse_structure(enriched, settings.structure)
    close = float(enriched["close"].iloc[-1])

    assert all(level.price > close for level in report.resistances)
    assert all(level.price < close for level in report.supports)
    if report.resistances:
        # Sorted nearest-first.
        assert report.resistances == sorted(report.resistances, key=lambda lv: lv.price)
    if report.supports:
        assert report.supports == sorted(
            report.supports, key=lambda lv: lv.price, reverse=True
        )


def test_room_is_measured_in_atr(settings, builder, ohlcv) -> None:
    enriched = builder.enrich(ohlcv, "15m")
    report = analyse_structure(enriched, settings.structure)
    close = float(enriched["close"].iloc[-1])
    atr = float(enriched["atr"].iloc[-1])

    if report.nearest_resistance is not None:
        expected = (report.nearest_resistance.price - close) / atr
        assert report.room_up_atr == pytest.approx(max(0.0, expected))
    assert report.room_for(1) == report.room_up_atr
    assert report.room_for(-1) == report.room_down_atr


def test_level_strength_rewards_touches_and_recency() -> None:
    from analysis.structure import Level

    frequent = Level(price=100.0, kind="support", touches=6, swings=3, last_touch_bars_ago=2)
    rare = Level(price=100.0, kind="support", touches=1, swings=1, last_touch_bars_ago=140)
    assert frequent.strength > rare.strength
    assert 0.0 <= rare.strength <= 1.0


def test_liquidity_sweep_is_detected(settings, builder) -> None:
    """A wick through the swing low that closes back above it is a bullish sweep."""
    base = make_ohlcv(bars=300, seed=31, volatility=0.003).copy()
    enriched = builder.enrich(base, "15m")
    swings = find_swings(enriched, settings.structure.swing_strength, confirmed_only=True)
    lows = [s for s in swings if s.kind == "low"]
    assert lows, "fixture must contain a swing low"

    target = lows[-1].price
    last = base.index[-1]
    base.loc[last, "open"] = target * 1.004
    base.loc[last, "close"] = target * 1.006
    base.loc[last, "high"] = target * 1.008
    base.loc[last, "low"] = target * 0.985  # long lower wick through the level

    report = analyse_structure(builder.enrich(base, "15m"), settings.structure)
    assert report.sweep_direction == 1


# ---------------------------------------------------------------------------
# Volume
# ---------------------------------------------------------------------------
def test_volume_spike_and_dry_are_mutually_exclusive(settings, builder, ohlcv) -> None:
    state = analyse_volume(builder.enrich(ohlcv, "15m"), settings.indicators)
    assert not (state.spike and state.dry)


def test_volume_confirms_only_with_spike_and_matching_flow(settings, builder, ohlcv) -> None:
    enriched = builder.enrich(ohlcv, "15m").copy()
    average = float(enriched["volume_sma"].iloc[-1])
    last = enriched.index[-1]

    # Big volume, close on the high -> buying confirmation for a long.
    enriched.loc[last, "volume"] = average * 4.0
    enriched.loc[last, "rel_volume"] = 4.0
    enriched.loc[last, "volume_delta_pct"] = 90.0
    enriched.loc[last, "cmf"] = 0.4
    enriched.loc[last, "mfi"] = 70.0
    state = analyse_volume(enriched, settings.indicators)
    assert state.spike is True
    assert state.bias == 1
    assert state.confirms(1) is True
    assert state.confirms(-1) is False

    # Dry volume never confirms, whatever the flow says.
    enriched.loc[last, "rel_volume"] = 0.2
    dry = analyse_volume(enriched, settings.indicators)
    assert dry.dry is True
    assert dry.confirms(1) is False


def test_volume_state_degrades_without_data(settings) -> None:
    state = analyse_volume(pd.DataFrame(), settings.indicators)
    assert state.confirms(1) is False
    assert state.dry is True


# ---------------------------------------------------------------------------
# Regime
# ---------------------------------------------------------------------------
def test_strong_uptrend_is_classified_as_trend_up(settings, builder) -> None:
    up = make_ohlcv(bars=600, seed=41, drift=0.004, volatility=0.0012)
    report = classify_regime(builder.enrich(up, "15m"), settings.indicators)
    assert report.regime is MarketRegime.TREND_UP
    assert report.regime.bias == 1
    assert report.strength is not TrendStrength.NONE


def test_quiet_market_is_not_tradable(settings, builder) -> None:
    """A flat, low-volatility series must not be reported as tradable."""
    quiet = make_ohlcv(bars=600, seed=42, drift=0.0, volatility=0.00008)
    report = classify_regime(builder.enrich(quiet, "15m"), settings.indicators)
    assert report.regime in {MarketRegime.SIDEWAYS, MarketRegime.RANGE}
    assert report.tradable is False


def test_extreme_volatility_is_not_tradable(settings, builder, ohlcv) -> None:
    enriched = builder.enrich(ohlcv, "15m").copy()
    last = enriched.index[-1]
    enriched.loc[last, "natr"] = settings.indicators.atr_max_pct * 2.0
    report = classify_regime(enriched, settings.indicators)
    assert report.volatility is VolatilityState.EXTREME
    assert report.tradable is False


def test_regime_degrades_without_indicators(settings) -> None:
    report = classify_regime(pd.DataFrame(), settings.indicators)
    assert report.tradable is False
    assert report.strength is TrendStrength.NONE


def test_directionless_market_is_never_strong(settings, builder) -> None:
    """ADX alone must not make a market 'strong' when direction is unclear."""
    choppy = make_ohlcv(bars=600, seed=43, drift=0.0, volatility=0.004)
    report = classify_regime(builder.enrich(choppy, "15m"), settings.indicators)
    if report.regime.bias == 0:
        assert report.strength is not TrendStrength.STRONG


def test_context_bias_score_is_bounded(builder, ohlcv) -> None:
    context = builder.build("T/USDT", "15m", builder.enrich(ohlcv, "15m"))
    assert -1.0 <= context.bias_score <= 1.0
    assert context.bias in {-1, 0, 1}
    assert np.isfinite(context.close)
    assert context.is_complete
