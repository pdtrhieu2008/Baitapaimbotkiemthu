"""Momentum indicators: RSI, MACD, Stochastic, Stochastic RSI, ROC."""

from __future__ import annotations

import numpy as np
import pandas as pd

from indicators.base import rolling_max, rolling_min, wilder_smooth

__all__ = ["macd", "roc", "rsi", "stochastic", "stoch_rsi"]


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's Relative Strength Index, bounded ``[0, 100]``.

    Args:
        series: usually the close.
        period: lookback.

    Returns:
        RSI series. A window with no losses yields 100 (not NaN), which is the
        mathematically correct limit of ``100 - 100/(1+inf)``.
    """
    if period < 1:
        raise ValueError("period must be >= 1")
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    avg_gain = wilder_smooth(gain, period)
    avg_loss = wilder_smooth(loss, period)

    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100.0 - (100.0 / (1.0 + rs))
    # avg_loss == 0 -> pure uptrend -> RSI 100; both zero -> flat -> neutral 50.
    flat = (avg_gain == 0.0) & (avg_loss == 0.0)
    out = out.where(~(avg_loss == 0.0), 100.0)
    return out.where(~flat, 50.0)


def macd(
    series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> pd.DataFrame:
    """Moving Average Convergence Divergence.

    Args:
        series: usually the close.
        fast: fast EMA period.
        slow: slow EMA period.
        signal: EMA period of the MACD line.

    Returns:
        Frame with ``macd``, ``macd_signal`` and ``macd_hist``.
    """
    if fast >= slow:
        raise ValueError(f"fast ({fast}) must be < slow ({slow})")
    if signal < 1:
        raise ValueError("signal must be >= 1")

    ema_fast = series.ewm(span=fast, adjust=False, min_periods=fast).mean()
    ema_slow = series.ewm(span=slow, adjust=False, min_periods=slow).mean()
    line = ema_fast - ema_slow
    # The signal line is seeded from the MACD line, so it needs slow+signal
    # bars before it is trustworthy; min_periods enforces that.
    signal_line = line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return pd.DataFrame(
        {"macd": line, "macd_signal": signal_line, "macd_hist": line - signal_line}
    )


def stochastic(
    df: pd.DataFrame, period: int = 14, k_smooth: int = 3, d_smooth: int = 3
) -> pd.DataFrame:
    """Stochastic oscillator (%K / %D) over the high-low range.

    Args:
        df: OHLCV frame.
        period: lookback for the range.
        k_smooth: SMA applied to raw %K.
        d_smooth: SMA applied to %K to obtain %D.

    Returns:
        Frame with ``stoch_k`` and ``stoch_d``.
    """
    highest = rolling_max(df["high"], period)
    lowest = rolling_min(df["low"], period)
    span = (highest - lowest).replace(0.0, np.nan)
    raw_k = 100.0 * (df["close"] - lowest) / span
    k = raw_k.rolling(k_smooth, min_periods=k_smooth).mean()
    d = k.rolling(d_smooth, min_periods=d_smooth).mean()
    return pd.DataFrame({"stoch_k": k, "stoch_d": d})


def stoch_rsi(
    series: pd.Series,
    rsi_period: int = 14,
    stoch_period: int = 14,
    k_smooth: int = 3,
    d_smooth: int = 3,
) -> pd.DataFrame:
    """Stochastic RSI: the stochastic of the RSI, not of price.

    More responsive than raw RSI, which makes it useful for timing a pullback
    entry inside an established trend — and useless as a standalone signal,
    which is why the strategy only ever reads it as a confirmation.

    Args:
        series: usually the close.
        rsi_period: RSI lookback.
        stoch_period: lookback of the stochastic applied to the RSI.
        k_smooth: SMA on raw %K.
        d_smooth: SMA on %K to obtain %D.

    Returns:
        Frame with ``stochrsi_k`` and ``stochrsi_d``, both ``[0, 100]``.
    """
    rsi_values = rsi(series, rsi_period)
    lowest = rolling_min(rsi_values, stoch_period)
    highest = rolling_max(rsi_values, stoch_period)
    span = (highest - lowest).replace(0.0, np.nan)
    raw_k = 100.0 * (rsi_values - lowest) / span
    k = raw_k.rolling(k_smooth, min_periods=k_smooth).mean()
    d = k.rolling(d_smooth, min_periods=d_smooth).mean()
    return pd.DataFrame({"stochrsi_k": k, "stochrsi_d": d})


def roc(series: pd.Series, period: int = 12) -> pd.Series:
    """Rate of change, in percent, over ``period`` bars."""
    if period < 1:
        raise ValueError("period must be >= 1")
    previous = series.shift(period).replace(0.0, np.nan)
    return (series / previous - 1.0) * 100.0
