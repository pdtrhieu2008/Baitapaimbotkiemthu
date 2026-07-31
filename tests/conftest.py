"""Shared fixtures.

Most fixtures are session-scoped: they are read-only (generated data, validated
settings, a stateless ContextBuilder) and regenerating them per test dominated
the suite's wall time. Tests that need to mutate a frame take a ``.copy()``.

Synthetic series are used **only** in tests, and only because a unit test needs
determinism: a seeded generator gives byte-identical bars on every run, which a
live exchange cannot. No production code path fabricates data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from analysis.context import ContextBuilder
from config.settings import Settings, load_settings

AGGREGATION = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}


def make_ohlcv(
    bars: int = 1200,
    seed: int = 7,
    freq: str = "15min",
    start: str = "2024-01-01",
    drift: float = 0.0,
    regime_shifts: bool = False,
    volatility: float = 0.0035,
    start_price: float = 30_000.0,
) -> pd.DataFrame:
    """Generate a deterministic OHLCV frame.

    Bars are internally consistent — ``low <= open, close <= high`` always holds —
    so a test failure means the code under test is wrong, not the fixture.
    """
    rng = np.random.default_rng(seed)
    index = pd.date_range(start, periods=bars, freq=freq, tz="UTC")

    if regime_shifts:
        block = 80
        shifts = rng.choice([0.0011, -0.0009, 0.0], size=bars // block + 1)
        trend = np.repeat(shifts, block)[:bars]
    else:
        trend = np.full(bars, drift)

    returns = rng.normal(0.0, volatility, bars) + trend
    close = start_price * np.exp(np.cumsum(returns))
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) * (1.0 + np.abs(rng.normal(0, 0.0018, bars)))
    low = np.minimum(open_, close) * (1.0 - np.abs(rng.normal(0, 0.0018, bars)))
    volume = rng.lognormal(5.0, 0.85, bars)

    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=index,
    )


def resample_ladder(base: pd.DataFrame, rules: dict[str, str]) -> dict[str, pd.DataFrame]:
    """Build higher timeframes from one base series.

    Resampling from a single source is what makes the multi-timeframe
    relationship real; independently generated series can never align and would
    make every MTF test vacuous.
    """
    frames = {}
    for timeframe, rule in rules.items():
        frames[timeframe] = base.resample(rule).agg(AGGREGATION).dropna()
    return frames


@pytest.fixture(scope="session")
def ohlcv() -> pd.DataFrame:
    """A 1200-bar 15-minute series with regime shifts."""
    return make_ohlcv(bars=1200, seed=7, regime_shifts=True)


@pytest.fixture(scope="session")
def settings() -> Settings:
    """The shipped configuration, with the environment ignored."""
    return load_settings("config/config.yaml", env_file=None)


@pytest.fixture(scope="session")
def test_settings(settings: Settings) -> Settings:
    """Configuration tuned for a short, fast test run.

    Uses a 4h/1h/15m ladder because a 1d timeframe would need ~240 daily bars
    (EMA-200 plus ADX seeding) to produce a usable context.
    """
    return settings.with_overrides(
        {
            "backtest.timeframe": "15m",
            "data.primary_timeframe": "15m",
            "strategy.mtf.context": ["4h"],
            "strategy.mtf.confirm": ["1h"],
            "strategy.mtf.entry": ["15m"],
            "indicators.adx_trend_min": 18,
            "strategy.min_score_to_emit": 60,
        }
    )


@pytest.fixture(scope="session")
def builder(settings: Settings) -> ContextBuilder:
    """A context builder on the default indicator/structure configuration."""
    return ContextBuilder(settings.indicators, settings.structure)


@pytest.fixture(scope="session")
def enriched(builder: ContextBuilder, ohlcv: pd.DataFrame) -> pd.DataFrame:
    """The fixture series with all indicator and pattern columns."""
    return builder.enrich(ohlcv, "15m")


@pytest.fixture(scope="session")
def ladder() -> dict[str, pd.DataFrame]:
    """A coherent 15m/1h/4h set derived from one 6000-bar base series.

    Session-scoped: generating and, more importantly, back-testing over it is
    the slowest thing in the suite, and the data is read-only.
    """
    base = make_ohlcv(bars=6000, seed=42, regime_shifts=True)
    frames = {"15m": base}
    frames.update(resample_ladder(base, {"1h": "1h", "4h": "4h"}))
    return frames
