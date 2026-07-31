"""Volatility indicators: ATR/NATR, Bollinger, Keltner, Donchian, squeeze."""

from __future__ import annotations

import numpy as np
import pandas as pd

from indicators.base import rolling_max, rolling_min, true_range, wilder_smooth

__all__ = [
    "atr",
    "bollinger_bands",
    "donchian_channel",
    "keltner_channel",
    "natr",
    "squeeze_on",
    "true_range",
]


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range (Wilder).

    This is the risk unit of the whole system: stop distance, take-profit
    distance, trailing step, the volatility-anomaly guard and the "room to the
    next level" gate are all expressed in ATR multiples so that one set of
    parameters behaves the same on BTC and on a low-priced altcoin.
    """
    return wilder_smooth(true_range(df["high"], df["low"], df["close"]), period)


def natr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Normalised ATR: ATR as a percentage of close.

    Used by the ``atr_in_range`` gate — an absolute ATR cannot be compared
    across instruments, a percentage can.
    """
    return 100.0 * atr(df, period) / df["close"].replace(0.0, np.nan)


def bollinger_bands(
    series: pd.Series, period: int = 20, std_mult: float = 2.0
) -> pd.DataFrame:
    """Bollinger Bands around an SMA.

    Args:
        series: usually the close.
        period: SMA and standard-deviation window.
        std_mult: band width in standard deviations.

    Returns:
        Frame with ``bb_upper``, ``bb_middle``, ``bb_lower``, ``bb_width``
        (band span as % of the middle band) and ``bb_pct`` (position of price
        inside the bands, 0 = lower, 1 = upper).
    """
    if period < 2:
        raise ValueError("bollinger period must be >= 2")
    middle = series.rolling(period, min_periods=period).mean()
    # ddof=0: the population deviation, matching every charting platform.
    deviation = series.rolling(period, min_periods=period).std(ddof=0)
    upper = middle + std_mult * deviation
    lower = middle - std_mult * deviation
    span = (upper - lower).replace(0.0, np.nan)
    return pd.DataFrame(
        {
            "bb_upper": upper,
            "bb_middle": middle,
            "bb_lower": lower,
            "bb_width": 100.0 * (upper - lower) / middle.replace(0.0, np.nan),
            "bb_pct": (series - lower) / span,
        }
    )


def keltner_channel(
    df: pd.DataFrame, period: int = 20, atr_period: int = 20, multiplier: float = 1.5
) -> pd.DataFrame:
    """Keltner Channel: an EMA with ATR-scaled envelopes.

    Args:
        df: OHLCV frame.
        period: EMA period of the centre line.
        atr_period: ATR period for the envelope width.
        multiplier: ATR multiple.

    Returns:
        Frame with ``kc_upper``, ``kc_middle``, ``kc_lower``.
    """
    middle = df["close"].ewm(span=period, adjust=False, min_periods=period).mean()
    width = multiplier * atr(df, atr_period)
    return pd.DataFrame(
        {"kc_upper": middle + width, "kc_middle": middle, "kc_lower": middle - width}
    )


def donchian_channel(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """Donchian Channel: the rolling high/low envelope.

    ``dc_upper``/``dc_lower`` include the current bar, so a bar that prints a
    new high has ``high == dc_upper``. Breakout rules therefore compare against
    the channel of the **previous** bar (``dc_upper.shift(1)``) — see
    :mod:`analysis.structure`.
    """
    upper = rolling_max(df["high"], period)
    lower = rolling_min(df["low"], period)
    return pd.DataFrame(
        {"dc_upper": upper, "dc_middle": (upper + lower) / 2.0, "dc_lower": lower}
    )


def squeeze_on(bb: pd.DataFrame, kc: pd.DataFrame) -> pd.Series:
    """TTM-style squeeze flag: Bollinger Bands inside the Keltner Channel.

    A squeeze marks compressed volatility — the condition that precedes an
    expansion. The strategy treats a *released* squeeze (squeeze on, then off)
    as a valid breakout trigger, and an *active* squeeze as a reason to wait.

    Args:
        bb: output of :func:`bollinger_bands`.
        kc: output of :func:`keltner_channel`.

    Returns:
        Boolean series, ``True`` while compressed.
    """
    return (bb["bb_upper"] < kc["kc_upper"]) & (bb["bb_lower"] > kc["kc_lower"])
