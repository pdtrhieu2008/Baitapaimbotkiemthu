"""Trend indicators: SMA, EMA, WMA, VWAP, ADX/DI, Supertrend, Ichimoku."""

from __future__ import annotations

import numpy as np
import pandas as pd

from indicators.base import (
    rolling_max,
    rolling_min,
    true_range,
    wilder_smooth,
)

__all__ = [
    "adx",
    "ema",
    "ichimoku",
    "sma",
    "supertrend",
    "vwap",
    "wma",
]


def sma(series: pd.Series, period: int) -> pd.Series:
    """Simple moving average."""
    if period < 1:
        raise ValueError("period must be >= 1")
    return series.rolling(window=period, min_periods=period).mean()


def ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential moving average with the standard ``2/(n+1)`` smoothing.

    ``adjust=False`` reproduces the recursive definition used by charting
    platforms; ``min_periods=period`` keeps the first, badly-seeded values as
    NaN instead of letting them leak into signals.
    """
    if period < 1:
        raise ValueError("period must be >= 1")
    return series.ewm(span=period, adjust=False, min_periods=period).mean()


def wma(series: pd.Series, period: int) -> pd.Series:
    """Linearly weighted moving average (most recent bar carries most weight)."""
    if period < 1:
        raise ValueError("period must be >= 1")
    weights = np.arange(1, period + 1, dtype=float)
    weights /= weights.sum()
    return series.rolling(window=period, min_periods=period).apply(
        lambda w: float(np.dot(w, weights)), raw=True
    )


def vwap(df: pd.DataFrame, *, anchor: str = "D") -> pd.Series:
    """Session-anchored Volume Weighted Average Price.

    VWAP is only meaningful relative to an anchor: a cumulative-since-inception
    VWAP on a 2-year series is a flat, useless line. The anchor resets the
    accumulation, matching how desks actually read it.

    Args:
        df: OHLCV frame with a ``DatetimeIndex``.
        anchor: pandas period alias to reset on — ``"D"`` (daily, the default),
            ``"W"`` or ``"ME"``.

    Returns:
        VWAP series aligned to ``df.index``.
    """
    typical = (df["high"] + df["low"] + df["close"]) / 3.0
    volume = df["volume"].fillna(0.0)
    # Periods carry no timezone, so drop it explicitly rather than letting
    # pandas warn about it. The index is already UTC, so the session boundaries
    # are UTC midnights either way.
    index = df.index
    naive = index.tz_localize(None) if index.tz is not None else index
    groups = naive.to_period(anchor)
    cum_pv = (typical * volume).groupby(groups).cumsum()
    cum_vol = volume.groupby(groups).cumsum()
    # A zero-volume session start would divide by zero; NaN is the honest answer.
    return cum_pv / cum_vol.replace(0.0, np.nan)


def adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Average Directional Index with the directional indicators.

    ADX measures *how strongly* price trends, not in which direction — the sign
    comes from ``plus_di`` vs ``minus_di``. The strategy uses ADX purely as a
    gate (``ADX > adx_trend_min``) to avoid taking breakout setups inside a
    range, which is where breakout systems bleed.

    Args:
        df: OHLCV frame.
        period: Wilder period.

    Returns:
        Frame with ``adx``, ``plus_di``, ``minus_di``.
    """
    high, low, close = df["high"], df["low"], df["close"]
    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=df.index
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=df.index
    )

    atr_ = wilder_smooth(true_range(high, low, close), period)
    safe_atr = atr_.replace(0.0, np.nan)
    plus_di = 100.0 * wilder_smooth(plus_dm, period) / safe_atr
    minus_di = 100.0 * wilder_smooth(minus_dm, period) / safe_atr

    di_sum = (plus_di + minus_di).replace(0.0, np.nan)
    dx = 100.0 * (plus_di - minus_di).abs() / di_sum
    return pd.DataFrame(
        {"adx": wilder_smooth(dx, period), "plus_di": plus_di, "minus_di": minus_di}
    )


def supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 3.0) -> pd.DataFrame:
    """Supertrend: an ATR band that flips side when price closes through it.

    The recursion is inherently sequential (each band ratchets off the previous
    one), so this runs a numpy loop rather than a vectorised expression. At a
    few hundred bars per symbol per cycle that cost is irrelevant, and the
    explicit loop is far easier to verify against a chart.

    Args:
        df: OHLCV frame.
        period: ATR period.
        multiplier: ATR multiple for the band offset.

    Returns:
        Frame with ``supertrend`` (the active band) and ``supertrend_dir``
        (``1`` uptrend, ``-1`` downtrend).
    """
    if period < 1:
        raise ValueError("period must be >= 1")
    if multiplier <= 0:
        raise ValueError("multiplier must be > 0")

    atr_ = wilder_smooth(true_range(df["high"], df["low"], df["close"]), period)
    hl2 = (df["high"] + df["low"]) / 2.0
    upper_raw = (hl2 + multiplier * atr_).to_numpy(dtype=float)
    lower_raw = (hl2 - multiplier * atr_).to_numpy(dtype=float)
    close = df["close"].to_numpy(dtype=float)
    n = len(df)

    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    direction = np.zeros(n, dtype=float)
    trend = np.full(n, np.nan)

    # Start once ATR is defined; everything before stays NaN.
    valid = np.flatnonzero(np.isfinite(upper_raw) & np.isfinite(lower_raw))
    if valid.size == 0:
        return pd.DataFrame({"supertrend": trend, "supertrend_dir": np.nan}, index=df.index)

    start = int(valid[0])
    upper[start], lower[start] = upper_raw[start], lower_raw[start]
    direction[start] = 1.0
    trend[start] = lower[start]

    for i in range(start + 1, n):
        # The band only tightens; it resets when price closes beyond it.
        upper[i] = (
            upper_raw[i]
            if (upper_raw[i] < upper[i - 1] or close[i - 1] > upper[i - 1])
            else upper[i - 1]
        )
        lower[i] = (
            lower_raw[i]
            if (lower_raw[i] > lower[i - 1] or close[i - 1] < lower[i - 1])
            else lower[i - 1]
        )
        if close[i] > upper[i - 1]:
            direction[i] = 1.0
        elif close[i] < lower[i - 1]:
            direction[i] = -1.0
        else:
            direction[i] = direction[i - 1]
        trend[i] = lower[i] if direction[i] > 0 else upper[i]

    direction[:start] = np.nan
    return pd.DataFrame(
        {"supertrend": trend, "supertrend_dir": direction}, index=df.index
    )


def ichimoku(
    df: pd.DataFrame,
    tenkan: int = 9,
    kijun: int = 26,
    senkou_b: int = 52,
    displacement: int = 26,
) -> pd.DataFrame:
    """Ichimoku Kinko Hyo.

    Note on the two shifted components — this is where most implementations
    leak the future:

    * ``senkou_a``/``senkou_b`` are shifted **forward** by ``displacement``, so
      the cloud value sitting under bar *i* was computed from bar
      ``i - displacement``. That is causal and safe to compare against price.
    * The Chikou span is conventionally plotted **backwards**, i.e. the value
      drawn at bar *i* is ``close[i + displacement]`` — a future price. Exposing
      it as a column would silently inject lookahead into any rule that reads
      it, so this function instead returns ``chikou_above``, the equivalent
      causal statement "the current close is above the close ``displacement``
      bars ago".

    Args:
        df: OHLCV frame.
        tenkan: conversion-line period.
        kijun: base-line period.
        senkou_b: leading-span-B period.
        displacement: cloud projection, in bars.

    Returns:
        Frame with ``tenkan``, ``kijun``, ``senkou_a``, ``senkou_b``,
        ``chikou_above`` and ``cloud_top`` / ``cloud_bottom``.
    """
    for name, value in (
        ("tenkan", tenkan), ("kijun", kijun),
        ("senkou_b", senkou_b), ("displacement", displacement),
    ):
        if value < 1:
            raise ValueError(f"ichimoku {name} must be >= 1")

    high, low, close = df["high"], df["low"], df["close"]

    def _mid(period: int) -> pd.Series:
        return (rolling_max(high, period) + rolling_min(low, period)) / 2.0

    tenkan_sen = _mid(tenkan)
    kijun_sen = _mid(kijun)
    span_a = ((tenkan_sen + kijun_sen) / 2.0).shift(displacement)
    span_b = _mid(senkou_b).shift(displacement)

    return pd.DataFrame(
        {
            "tenkan": tenkan_sen,
            "kijun": kijun_sen,
            "senkou_a": span_a,
            "senkou_b": span_b,
            "cloud_top": pd.concat([span_a, span_b], axis=1).max(axis=1),
            "cloud_bottom": pd.concat([span_a, span_b], axis=1).min(axis=1),
            "chikou_above": close > close.shift(displacement),
        }
    )
