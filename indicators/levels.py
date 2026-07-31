"""Horizontal reference levels: classic and Fibonacci pivot points."""

from __future__ import annotations

import pandas as pd

from indicators.base import pandas_freq

__all__ = ["PIVOT_COLUMNS", "pivot_points"]

#: Columns produced by :func:`pivot_points`, in reading order.
PIVOT_COLUMNS: tuple[str, ...] = (
    "pivot", "pivot_r1", "pivot_r2", "pivot_r3", "pivot_s1", "pivot_s2", "pivot_s3",
)


def pivot_points(
    df: pd.DataFrame, method: str = "classic", period: str = "1d"
) -> pd.DataFrame:
    """Pivot levels derived from the **previous completed** period.

    The ``shift(1)`` on the resampled frame is what makes this causal: bars
    inside today's session see yesterday's high/low/close, never their own
    session's unfinished aggregate. Getting this wrong is a classic backtest
    inflator, because today's pivot computed from today's full range "knows"
    where the day topped out.

    Args:
        df: OHLCV frame with a ``DatetimeIndex``.
        method: ``"classic"`` or ``"fibonacci"``.
        period: aggregation period, e.g. ``"1d"`` or ``"1w"``.

    Returns:
        Frame indexed like ``df`` with the columns in :data:`PIVOT_COLUMNS`.
        Rows before the first completed period are NaN.

    Raises:
        ValueError: on an unknown method.
    """
    if method not in {"classic", "fibonacci"}:
        raise ValueError(f"unknown pivot method {method!r}, expected classic|fibonacci")

    freq = pandas_freq(period)
    aggregated = df.resample(freq).agg({"high": "max", "low": "min", "close": "last"})
    previous = aggregated.shift(1).dropna(how="all")
    if previous.empty:
        return pd.DataFrame(
            {name: pd.Series(index=df.index, dtype=float) for name in PIVOT_COLUMNS}
        )

    high, low, close = previous["high"], previous["low"], previous["close"]
    pivot = (high + low + close) / 3.0
    span = high - low

    if method == "classic":
        levels = {
            "pivot": pivot,
            "pivot_r1": 2.0 * pivot - low,
            "pivot_r2": pivot + span,
            "pivot_r3": high + 2.0 * (pivot - low),
            "pivot_s1": 2.0 * pivot - high,
            "pivot_s2": pivot - span,
            "pivot_s3": low - 2.0 * (high - pivot),
        }
    else:  # fibonacci
        levels = {
            "pivot": pivot,
            "pivot_r1": pivot + 0.382 * span,
            "pivot_r2": pivot + 0.618 * span,
            "pivot_r3": pivot + 1.000 * span,
            "pivot_s1": pivot - 0.382 * span,
            "pivot_s2": pivot - 0.618 * span,
            "pivot_s3": pivot - 1.000 * span,
        }

    frame = pd.DataFrame(levels)
    # Forward-fill the period's levels across every intraday bar that follows.
    return frame.reindex(df.index, method="ffill")[list(PIVOT_COLUMNS)]
