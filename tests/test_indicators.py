"""Indicator correctness.

Where possible, values are checked against hand-computed results rather than
against another library — otherwise the test only proves the two agree, not that
either is right.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from indicators import momentum, trend, volatility, volume
from indicators.base import validate_ohlcv, wilder_smooth
from indicators.engine import IndicatorEngine
from tests.conftest import make_ohlcv


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------
def test_wilder_smooth_matches_the_recursion() -> None:
    """Wilder smoothing must equal prev*(n-1)/n + new/n, not an SMA."""
    values = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    period = 3
    result = wilder_smooth(values, period)

    # Seeded at the period-th value by the EWMA recursion with alpha = 1/n.
    manual = values.iloc[0]
    for value in values.iloc[1:]:
        manual = manual + (value - manual) / period
    assert float(result.iloc[-1]) == pytest.approx(manual)
    # Under-seeded values stay NaN.
    assert result.iloc[: period - 1].isna().all()


def test_sma_is_the_arithmetic_mean() -> None:
    values = pd.Series([2.0, 4.0, 6.0, 8.0])
    assert float(trend.sma(values, 4).iloc[-1]) == pytest.approx(5.0)
    assert trend.sma(values, 4).iloc[:3].isna().all()


def test_ema_uses_2_over_n_plus_1() -> None:
    values = pd.Series([10.0, 11.0, 12.0, 13.0, 14.0])
    period = 3
    alpha = 2.0 / (period + 1)
    result = trend.ema(values, period)
    seed = values.iloc[:period].mean()  # not how pandas seeds it; compute directly
    manual = values.iloc[0]
    for value in values.iloc[1:]:
        manual = alpha * value + (1 - alpha) * manual
    assert float(result.iloc[-1]) == pytest.approx(manual)
    assert seed  # keep the intent visible: pandas seeds recursively, not on an SMA


def test_validate_ohlcv_rejects_corrupt_frames(ohlcv: pd.DataFrame) -> None:
    validate_ohlcv(ohlcv)

    with pytest.raises(ValueError, match="missing column"):
        validate_ohlcv(ohlcv.drop(columns=["volume"]))

    unsorted = pd.concat([ohlcv.iloc[5:10], ohlcv.iloc[0:5]])
    with pytest.raises(ValueError, match="sorted ascending"):
        validate_ohlcv(unsorted)

    duplicated = pd.concat([ohlcv.iloc[:10], ohlcv.iloc[:10]]).sort_index()
    with pytest.raises(ValueError, match="duplicate timestamps"):
        validate_ohlcv(duplicated)

    broken = ohlcv.iloc[:10].copy()
    broken.loc[broken.index[3], "high"] = broken["low"].iloc[3] - 1.0
    with pytest.raises(ValueError, match="high < low"):
        validate_ohlcv(broken)

    with pytest.raises(TypeError, match="DatetimeIndex"):
        validate_ohlcv(ohlcv.reset_index(drop=True))


# ---------------------------------------------------------------------------
# Momentum
# ---------------------------------------------------------------------------
def test_rsi_is_bounded_and_handles_the_extremes() -> None:
    rising = pd.Series(np.arange(1, 60, dtype=float))
    assert float(momentum.rsi(rising, 14).iloc[-1]) == pytest.approx(100.0)

    falling = pd.Series(np.arange(60, 1, -1, dtype=float))
    assert float(momentum.rsi(falling, 14).iloc[-1]) == pytest.approx(0.0, abs=1e-9)

    flat = pd.Series([50.0] * 40)
    assert float(momentum.rsi(flat, 14).iloc[-1]) == pytest.approx(50.0)


def test_rsi_stays_within_bounds_on_noise(ohlcv: pd.DataFrame) -> None:
    values = momentum.rsi(ohlcv["close"], 14).dropna()
    assert values.between(0.0, 100.0).all()


def test_macd_histogram_is_line_minus_signal(ohlcv: pd.DataFrame) -> None:
    frame = momentum.macd(ohlcv["close"], 12, 26, 9).dropna()
    assert np.allclose(frame["macd_hist"], frame["macd"] - frame["macd_signal"])


def test_macd_rejects_inverted_periods() -> None:
    with pytest.raises(ValueError, match="must be <"):
        momentum.macd(pd.Series([1.0, 2.0]), fast=26, slow=12)


def test_stoch_rsi_is_bounded(ohlcv: pd.DataFrame) -> None:
    frame = momentum.stoch_rsi(ohlcv["close"]).dropna()
    assert frame["stochrsi_k"].between(0.0, 100.0).all()
    assert frame["stochrsi_d"].between(0.0, 100.0).all()


# ---------------------------------------------------------------------------
# Volatility
# ---------------------------------------------------------------------------
def test_true_range_includes_the_gap() -> None:
    """A gap up must produce a TR larger than the bar's own range."""
    frame = pd.DataFrame(
        {
            "open": [100.0, 120.0],
            "high": [101.0, 121.0],
            "low": [99.0, 119.0],
            "close": [100.0, 120.0],
            "volume": [1.0, 1.0],
        },
        index=pd.date_range("2024-01-01", periods=2, freq="1h", tz="UTC"),
    )
    from indicators.base import true_range

    tr = true_range(frame["high"], frame["low"], frame["close"])
    assert float(tr.iloc[1]) == pytest.approx(21.0)  # 121 - 100, not 121 - 119


def test_atr_is_positive_and_natr_is_relative(ohlcv: pd.DataFrame) -> None:
    atr = volatility.atr(ohlcv, 14).dropna()
    assert (atr > 0).all()

    natr = volatility.natr(ohlcv, 14).dropna()
    expected = 100.0 * atr / ohlcv["close"].loc[atr.index]
    assert np.allclose(natr, expected)


def test_bollinger_bands_are_ordered_and_pct_locates_price(ohlcv: pd.DataFrame) -> None:
    frame = volatility.bollinger_bands(ohlcv["close"], 20, 2.0).dropna()
    assert (frame["bb_upper"] >= frame["bb_middle"]).all()
    assert (frame["bb_middle"] >= frame["bb_lower"]).all()

    # bb_pct 0 = lower band, 1 = upper band.
    close = ohlcv["close"].loc[frame.index]
    manual = (close - frame["bb_lower"]) / (frame["bb_upper"] - frame["bb_lower"])
    assert np.allclose(frame["bb_pct"], manual)


def test_donchian_includes_the_current_bar(ohlcv: pd.DataFrame) -> None:
    """Documented behaviour: a new high has high == dc_upper."""
    frame = volatility.donchian_channel(ohlcv, 20).dropna()
    highs = ohlcv["high"].loc[frame.index]
    assert (frame["dc_upper"] >= highs).all()
    # At least one bar must be its own channel high, or the note is wrong.
    assert bool(np.isclose(frame["dc_upper"], highs).any())


def test_keltner_width_scales_with_atr(ohlcv: pd.DataFrame) -> None:
    frame = volatility.keltner_channel(ohlcv, 20, 20, 1.5).dropna()
    atr = volatility.atr(ohlcv, 20).loc[frame.index]
    assert np.allclose(frame["kc_upper"] - frame["kc_middle"], 1.5 * atr)


# ---------------------------------------------------------------------------
# Trend
# ---------------------------------------------------------------------------
def test_adx_is_bounded_and_di_reacts_to_direction() -> None:
    up = make_ohlcv(bars=300, seed=1, drift=0.004, volatility=0.001)
    frame = trend.adx(up, 14).dropna()
    assert frame["adx"].between(0.0, 100.0).all()
    assert float(frame["plus_di"].iloc[-1]) > float(frame["minus_di"].iloc[-1])

    down = make_ohlcv(bars=300, seed=2, drift=-0.004, volatility=0.001)
    frame = trend.adx(down, 14).dropna()
    assert float(frame["minus_di"].iloc[-1]) > float(frame["plus_di"].iloc[-1])


def test_supertrend_flips_and_sits_on_the_right_side() -> None:
    up = make_ohlcv(bars=300, seed=5, drift=0.004, volatility=0.0012)
    frame = trend.supertrend(up, 10, 3.0).dropna()
    assert set(frame["supertrend_dir"].unique()) <= {1.0, -1.0}
    assert float(frame["supertrend_dir"].iloc[-1]) == 1.0
    # In an uptrend the band is the lower one, i.e. below price.
    assert float(frame["supertrend"].iloc[-1]) < float(up["close"].iloc[-1])


def test_supertrend_rejects_bad_parameters(ohlcv: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="multiplier"):
        trend.supertrend(ohlcv, 10, 0.0)


def test_vwap_resets_each_session(ohlcv: pd.DataFrame) -> None:
    """A daily-anchored VWAP must start over at each UTC midnight."""
    result = trend.vwap(ohlcv, anchor="D").dropna()
    first_of_day = result.groupby(result.index.date).head(1)
    typical = ((ohlcv["high"] + ohlcv["low"] + ohlcv["close"]) / 3.0).loc[first_of_day.index]
    # The first bar of a session has VWAP equal to its own typical price.
    assert np.allclose(first_of_day, typical)


def test_ichimoku_cloud_bounds_are_ordered(ohlcv: pd.DataFrame) -> None:
    frame = trend.ichimoku(ohlcv).dropna()
    assert (frame["cloud_top"] >= frame["cloud_bottom"]).all()


# ---------------------------------------------------------------------------
# Volume
# ---------------------------------------------------------------------------
def test_obv_follows_the_signed_close_change() -> None:
    frame = pd.DataFrame(
        {
            "open": [1.0, 1.0, 1.0],
            "high": [1.0, 1.0, 1.0],
            "low": [1.0, 1.0, 1.0],
            "close": [10.0, 11.0, 9.0],
            "volume": [100.0, 200.0, 300.0],
        },
        index=pd.date_range("2024-01-01", periods=3, freq="1h", tz="UTC"),
    )
    result = volume.obv(frame)
    assert list(result) == [0.0, 200.0, -100.0]


def test_mfi_and_cmf_are_bounded(ohlcv: pd.DataFrame) -> None:
    assert volume.mfi(ohlcv, 14).dropna().between(0.0, 100.0).all()
    assert volume.cmf(ohlcv, 20).dropna().between(-1.0, 1.0).all()


def test_effort_split_conserves_volume(ohlcv: pd.DataFrame) -> None:
    frame = volume.effort_split(ohlcv)
    assert np.allclose(frame["buy_volume"] + frame["sell_volume"], ohlcv["volume"])
    assert (frame["buy_volume"] >= 0).all()
    assert frame["volume_delta_pct"].dropna().between(-100.0, 100.0).all()


def test_effort_split_reads_a_close_on_the_high_as_buying() -> None:
    frame = pd.DataFrame(
        {
            "open": [10.0],
            "high": [12.0],
            "low": [10.0],
            "close": [12.0],
            "volume": [500.0],
        },
        index=pd.date_range("2024-01-01", periods=1, freq="1h", tz="UTC"),
    )
    result = volume.effort_split(frame)
    assert float(result["buy_volume"].iloc[0]) == pytest.approx(500.0)
    assert float(result["volume_delta_pct"].iloc[0]) == pytest.approx(100.0)


def test_volume_profile_locates_the_value_area(ohlcv: pd.DataFrame) -> None:
    profile = volume.volume_profile(ohlcv, bins=24, lookback=200)
    assert profile is not None
    assert profile.val <= profile.poc <= profile.vah
    window = ohlcv.iloc[-200:]
    assert window["low"].min() <= profile.poc <= window["high"].max()
    assert sum(profile.volumes) == pytest.approx(window["volume"].sum(), rel=1e-6)


def test_volume_profile_returns_none_without_volume(ohlcv: pd.DataFrame) -> None:
    silent = ohlcv.copy()
    silent["volume"] = 0.0
    assert volume.volume_profile(silent, lookback=100) is None


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------
def test_engine_produces_every_documented_column(settings, ohlcv: pd.DataFrame) -> None:
    engine = IndicatorEngine(settings.indicators)
    out = engine.compute(ohlcv)

    expected = {
        "ema_fast", "ema_slow", "ema_trend", "sma", "vwap", "adx", "plus_di", "minus_di",
        "supertrend", "supertrend_dir", "tenkan", "kijun", "senkou_a", "senkou_b",
        "chikou_above", "rsi", "macd", "macd_signal", "macd_hist", "stoch_k", "stoch_d",
        "stochrsi_k", "stochrsi_d", "atr", "natr", "atr_median", "bb_upper", "bb_lower",
        "bb_width", "bb_pct", "kc_upper", "kc_lower", "dc_upper", "dc_lower", "squeeze_on",
        "squeeze_released", "obv", "obv_ema", "cmf", "mfi", "volume_sma", "rel_volume",
        "buy_volume", "sell_volume", "volume_delta_pct", "pivot", "pivot_r1", "pivot_s1",
    }
    missing = expected - set(out.columns)
    assert not missing, f"engine is missing columns: {sorted(missing)}"
    assert not out.iloc[-1].isna().any(), "the last row should be fully seeded"


def test_engine_leaves_the_input_untouched(settings, ohlcv: pd.DataFrame) -> None:
    """compute() must not mutate the caller's frame."""
    engine = IndicatorEngine(settings.indicators)
    before = ohlcv.copy()
    engine.compute(ohlcv)
    pd.testing.assert_frame_equal(ohlcv, before)


def test_engine_skips_pivots_on_the_pivot_timeframe(settings, builder) -> None:
    """Daily pivots on a daily chart would be degenerate, so they are omitted."""
    daily = make_ohlcv(bars=400, seed=9, freq="1D")
    enriched = builder.enrich(daily, "1d")
    assert "pivot" not in enriched.columns
