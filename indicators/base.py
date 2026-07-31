"""Shared primitives for the indicator library.

Every indicator in this package is implemented directly on pandas/numpy rather
than delegating to ``ta``/``pandas-ta``. Reasons:

* the formulas stay auditable and unit-tested inside this repo;
* no silent behaviour change when an upstream release retunes a default;
* no dependency conflicts on numpy>=2 / pandas>=2.2, which currently break
  several releases of ``pandas-ta``.

All functions are **causal**: the value at bar *i* is computed only from bars
``<= i``. Anything that would need a future bar is either omitted or exposed as
an explicit backward comparison (see :func:`indicators.trend.ichimoku`).
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

__all__ = [
    "OHLCV_COLUMNS",
    "pandas_freq",
    "rolling_max",
    "rolling_min",
    "validate_ohlcv",
    "wilder_smooth",
]

#: Column names every OHLCV frame in this project uses (lower case).
OHLCV_COLUMNS: tuple[str, ...] = ("open", "high", "low", "close", "volume")

_TF_PATTERN = re.compile(r"^(\d+)([smhdwM])$")

#: ccxt timeframe unit -> pandas offset alias. ``m`` maps to ``min`` because
#: pandas reserves ``M`` for month-end.
_FREQ_UNITS: dict[str, str] = {
    "s": "s",
    "m": "min",
    "h": "h",
    "d": "D",
    "w": "W",
    "M": "ME",
}


def pandas_freq(timeframe: str) -> str:
    """Translate a ccxt timeframe into a pandas offset alias.

    Args:
        timeframe: e.g. ``"15m"``, ``"4h"``, ``"1d"``.

    Returns:
        A pandas alias such as ``"15min"``, ``"4h"``, ``"1D"``.

    Raises:
        ValueError: on an unparsable timeframe.
    """
    match = _TF_PATTERN.match(timeframe.strip())
    if not match:
        raise ValueError(f"invalid timeframe {timeframe!r}")
    return f"{int(match.group(1))}{_FREQ_UNITS[match.group(2)]}"


def validate_ohlcv(df: pd.DataFrame, *, min_rows: int = 1) -> None:
    """Assert that ``df`` is a usable OHLCV frame.

    Catching a malformed frame here produces one clear error instead of a
    cascade of NaN columns and a nonsensical signal downstream.

    Args:
        df: candidate frame, indexed by bar-open timestamp.
        min_rows: minimum number of rows required.

    Raises:
        TypeError: if the index is not a ``DatetimeIndex``.
        ValueError: on missing columns, too few rows, a non-monotonic or
            duplicated index, or impossible bars (``high < low``).
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"expected a DataFrame, got {type(df).__name__}")
    missing = [c for c in OHLCV_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"OHLCV frame is missing column(s): {missing}")
    if len(df) < min_rows:
        raise ValueError(f"OHLCV frame has {len(df)} rows, need at least {min_rows}")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("OHLCV frame must be indexed by a DatetimeIndex of bar-open times")
    if not df.index.is_monotonic_increasing:
        raise ValueError("OHLCV index must be sorted ascending")
    if df.index.has_duplicates:
        dupes = df.index[df.index.duplicated()][:3].tolist()
        raise ValueError(f"OHLCV index contains duplicate timestamps, e.g. {dupes}")
    if len(df) and bool((df["high"] < df["low"]).any()):
        bad = df.index[df["high"] < df["low"]][:3].tolist()
        raise ValueError(f"corrupt bars with high < low at {bad}")


def wilder_smooth(series: pd.Series, period: int) -> pd.Series:
    """Wilder's smoothing, the recursive average behind RSI/ATR/ADX.

    Equivalent to an EWMA with ``alpha = 1 / period`` and no bias correction,
    which is what Wilder's original ``prev * (n-1)/n + new/n`` recursion is.
    Using a simple moving average here instead is the single most common cause
    of indicator values that disagree with TradingView.

    Args:
        series: input values.
        period: smoothing length.

    Returns:
        The smoothed series, same index.
    """
    if period < 1:
        raise ValueError(f"period must be >= 1, got {period}")
    return series.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def rolling_max(series: pd.Series, period: int) -> pd.Series:
    """Rolling maximum over ``period`` bars, inclusive of the current bar."""
    return series.rolling(window=period, min_periods=period).max()


def rolling_min(series: pd.Series, period: int) -> pd.Series:
    """Rolling minimum over ``period`` bars, inclusive of the current bar."""
    return series.rolling(window=period, min_periods=period).min()


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """Wilder's True Range: ``max(h-l, |h-c_prev|, |l-c_prev|)``.

    The gap terms matter: on an instrument that gaps (stocks at the open, or
    crypto after a liquidation cascade) ``high - low`` alone understates risk,
    which would in turn understate the ATR-based stop distance.
    """
    prev_close = close.shift(1)
    ranges = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    )
    return ranges.max(axis=1, skipna=False)


def series_slope(series: pd.Series, period: int) -> pd.Series:
    """Least-squares slope over a rolling window, normalised by price.

    Expressed in percent of the series value per bar so it is comparable
    between BTC at 67 000 and an altcoin at 0.4.

    Args:
        series: typically a moving average.
        period: window length.

    Returns:
        Slope in percent-per-bar.
    """
    if period < 2:
        raise ValueError("slope period must be >= 2")
    x = np.arange(period, dtype=float)
    x_centred = x - x.mean()
    denominator = float((x_centred**2).sum())

    def _slope(window: np.ndarray) -> float:
        return float((x_centred * (window - window.mean())).sum() / denominator)

    raw = series.rolling(window=period, min_periods=period).apply(_slope, raw=True)
    return (raw / series.abs().replace(0.0, np.nan)) * 100.0
