"""The most important tests in the suite: causality.

A backtest is only evidence if every value the strategy reads at bar *i* could
have been known at bar *i*. These tests assert that directly, by computing the
same bar two ways:

* once from the **full** series (as the backtester does, enriching once);
* once from the series **truncated** at that bar (as the live bot does, having
  never seen the future).

Any indicator, pattern or structural reading that leaked a future bar would
produce different numbers and fail here. This catches the whole class of bug —
including in code added later — rather than one instance of it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from analysis.context import ContextBuilder, MarketSnapshot
from analysis.swings import find_swings
from config.settings import Settings
from data.models import ExternalContext
from strategies import build_strategy
from strategies.base import Rejection, Signal
from tests.conftest import make_ohlcv

#: Bars to probe. Spread out, and all well past the warm-up.
PROBE_POSITIONS = (400, 600, 800, 999, 1150)


def test_indicator_columns_are_causal(builder: ContextBuilder, ohlcv: pd.DataFrame) -> None:
    """Every enriched column at bar i must not change when the future is removed."""
    full = builder.enrich(ohlcv, "15m")

    for position in PROBE_POSITIONS:
        truncated = builder.enrich(ohlcv.iloc[: position + 1], "15m")
        assert truncated.index[-1] == full.index[position]

        full_row = full.iloc[position]
        truncated_row = truncated.iloc[-1]

        differing: list[str] = []
        for column in full.columns:
            a, b = full_row[column], truncated_row[column]
            if isinstance(a, (bool, np.bool_)) or isinstance(b, (bool, np.bool_)):
                if bool(a) != bool(b):
                    differing.append(f"{column}: {a} vs {b}")
                continue
            a, b = float(a), float(b)
            if np.isnan(a) and np.isnan(b):
                continue
            if np.isnan(a) != np.isnan(b) or not np.isclose(a, b, rtol=1e-9, atol=1e-9):
                differing.append(f"{column}: {a} vs {b}")

        assert not differing, (
            f"bar {position} changed when future bars were removed, which means these columns "
            f"read the future: {differing}"
        )


def test_swings_are_only_confirmed_ones(ohlcv: pd.DataFrame) -> None:
    """A swing must not be reported until it has `strength` bars to its right."""
    strength = 3
    window = ohlcv.iloc[:500]
    swings = find_swings(window, strength, confirmed_only=True)

    assert swings, "the fixture should contain swings"
    last_allowed = len(window) - 1 - strength
    offenders = [s for s in swings if s.position > last_allowed]
    assert not offenders, (
        f"swings at {[s.position for s in offenders]} are within {strength} bars of the end and "
        f"cannot be confirmed yet"
    )


def test_swing_set_is_stable_as_data_arrives(ohlcv: pd.DataFrame) -> None:
    """Swings confirmed at bar i stay confirmed, at the same price, later on."""
    strength = 3
    early = find_swings(ohlcv.iloc[:600], strength, confirmed_only=True)
    later = find_swings(ohlcv.iloc[:700], strength, confirmed_only=True)
    # Keyed by (position, kind): an outside bar can legitimately be both a
    # fractal high and a fractal low, so position alone is not unique.
    later_by_key = {(s.position, s.kind): s for s in later}

    for swing in early:
        key = (swing.position, swing.kind)
        assert key in later_by_key, (
            f"{swing.kind} swing at {swing.position} disappeared once more data arrived"
        )
        assert later_by_key[key].price == pytest.approx(swing.price)


def test_structure_and_context_are_causal(builder: ContextBuilder, ohlcv: pd.DataFrame) -> None:
    """The whole TimeframeContext, not just the columns, must be reproducible."""
    full = builder.enrich(ohlcv, "15m")

    for position in PROBE_POSITIONS:
        from_full = builder.build("T/USDT", "15m", full.iloc[: position + 1])
        from_truncated = builder.build(
            "T/USDT", "15m", builder.enrich(ohlcv.iloc[: position + 1], "15m")
        )

        assert from_full.bias == from_truncated.bias
        assert from_full.bias_score == pytest.approx(from_truncated.bias_score)
        assert from_full.structure.trend == from_truncated.structure.trend
        assert from_full.structure.break_event.kind == from_truncated.structure.break_event.kind
        assert from_full.regime.regime == from_truncated.regime.regime
        assert from_full.volume.bias == from_truncated.volume.bias
        assert from_full.pattern_names == from_truncated.pattern_names


def test_strategy_verdict_is_causal(test_settings: Settings, ohlcv: pd.DataFrame) -> None:
    """The strategy's decision at a bar must not depend on later bars."""
    builder = ContextBuilder(test_settings.indicators, test_settings.structure)
    base = make_ohlcv(bars=1400, seed=11, regime_shifts=True)

    def verdict_at(cut: int) -> Signal | Rejection:
        history = base.iloc[:cut]
        frames = {
            "15m": history,
            "1h": history.resample("1h").agg(
                {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
            ).dropna(),
            "4h": history.resample("4h").agg(
                {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
            ).dropna(),
        }
        contexts = {}
        for timeframe, frame in frames.items():
            if len(frame) < 60:
                continue
            contexts[timeframe] = builder.build(
                "T/USDT", timeframe, builder.enrich(frame, timeframe)
            )
        snapshot = MarketSnapshot(
            symbol="T/USDT",
            primary_timeframe="15m",
            contexts=contexts,
            external=ExternalContext(symbol="T/USDT"),
        )
        # A fresh strategy each time: cooldown state must not leak between calls.
        return build_strategy(test_settings).evaluate(snapshot)

    for cut in (1000, 1200, 1400):
        first = verdict_at(cut)
        second = verdict_at(cut)
        assert type(first) is type(second)
        if isinstance(first, Rejection):
            assert isinstance(second, Rejection)
            assert (first.gate, first.detail) == (second.gate, second.detail)
        else:
            assert isinstance(second, Signal)
            assert first.side == second.side
            assert first.score == pytest.approx(second.score)
            assert first.stop_loss == pytest.approx(second.stop_loss)


def test_pivots_use_the_previous_period(builder: ContextBuilder) -> None:
    """Intraday bars must see yesterday's pivot, never today's own aggregate."""
    base = make_ohlcv(bars=1000, seed=3, freq="15min")
    enriched = builder.enrich(base, "15m")

    daily = base.resample("1D").agg({"high": "max", "low": "min", "close": "last"}).dropna()
    assert len(daily) >= 3

    # Take a bar in the middle of the third day and check its pivot equals the
    # value derived from the SECOND day.
    third_day = daily.index[2]
    mask = (enriched.index >= third_day) & (enriched.index < third_day + pd.Timedelta(days=1))
    sample = enriched[mask]
    assert not sample.empty

    previous = daily.iloc[1]
    expected = (previous["high"] + previous["low"] + previous["close"]) / 3.0
    assert float(sample["pivot"].iloc[0]) == pytest.approx(expected)
    # And crucially, not the pivot of its own day.
    today = daily.iloc[2]
    own_day = (today["high"] + today["low"] + today["close"]) / 3.0
    assert float(sample["pivot"].iloc[0]) != pytest.approx(own_day)


def test_ichimoku_exposes_no_future_chikou(builder: ContextBuilder, ohlcv: pd.DataFrame) -> None:
    """The frame must not carry a backward-shifted (future-reading) Chikou span."""
    enriched = builder.enrich(ohlcv, "15m")
    assert "chikou" not in enriched.columns, (
        "a forward-looking chikou column would inject lookahead into any rule reading it"
    )
    assert "chikou_above" in enriched.columns

    displacement = builder.indicator_cfg.ichimoku_displacement
    position = 800
    expected = bool(
        enriched["close"].iloc[position] > enriched["close"].iloc[position - displacement]
    )
    assert bool(enriched["chikou_above"].iloc[position]) is expected
