"""Volume indicators: OBV, CMF, MFI, relative volume, effort split, profile."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

__all__ = [
    "VolumeProfile",
    "cmf",
    "effort_split",
    "mfi",
    "obv",
    "relative_volume",
    "volume_profile",
]


def obv(df: pd.DataFrame) -> pd.Series:
    """On-Balance Volume: cumulative volume signed by the close-to-close move.

    The absolute level is arbitrary (it depends on where the series starts), so
    only its *slope* and its divergence from price carry information. The engine
    therefore also publishes an EMA of OBV for slope comparison.
    """
    direction = np.sign(df["close"].diff().fillna(0.0))
    return (direction * df["volume"].fillna(0.0)).cumsum()


def cmf(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Chaikin Money Flow, bounded ``[-1, 1]``.

    Weights each bar's volume by where the close sits within the bar's range,
    then averages over ``period``. Positive means accumulation.
    """
    if period < 1:
        raise ValueError("period must be >= 1")
    high, low, close = df["high"], df["low"], df["close"]
    span = (high - low).replace(0.0, np.nan)
    multiplier = ((close - low) - (high - close)) / span
    money_flow_volume = (multiplier * df["volume"]).fillna(0.0)
    volume_sum = df["volume"].rolling(period, min_periods=period).sum()
    return money_flow_volume.rolling(period, min_periods=period).sum() / volume_sum.replace(
        0.0, np.nan
    )


def mfi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Money Flow Index: a volume-weighted RSI on the typical price, ``[0, 100]``."""
    if period < 1:
        raise ValueError("period must be >= 1")
    typical = (df["high"] + df["low"] + df["close"]) / 3.0
    raw_flow = typical * df["volume"].fillna(0.0)
    delta = typical.diff()

    positive = raw_flow.where(delta > 0, 0.0).rolling(period, min_periods=period).sum()
    negative = raw_flow.where(delta < 0, 0.0).rolling(period, min_periods=period).sum()

    ratio = positive / negative.replace(0.0, np.nan)
    out = 100.0 - (100.0 / (1.0 + ratio))
    # No down-flow in the window is a genuine 100, not a NaN.
    return out.where(~(negative == 0.0), 100.0)


def relative_volume(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """Volume against its own moving average.

    Returns:
        Frame with ``volume_sma`` and ``rel_volume`` (current volume divided by
        that average). ``rel_volume >= volume_spike_multiplier`` is the "volume
        spike" condition; ``<= volume_dry_multiplier`` is "dry".
    """
    volume = df["volume"].fillna(0.0)
    average = volume.rolling(period, min_periods=period).mean()
    return pd.DataFrame(
        {"volume_sma": average, "rel_volume": volume / average.replace(0.0, np.nan)}
    )


def effort_split(df: pd.DataFrame) -> pd.DataFrame:
    """Split each bar's volume into an estimated buying and selling share.

    **This is an approximation, not real order-flow.** True buy/sell volume
    requires trade-level data (Binance ``aggTrades`` / ``takerBuyBaseVolume``),
    which the OHLCV endpoint does not carry. The estimate here distributes
    volume by where the close sits inside the bar's range: a close on the high
    is read as fully buyer-driven, a close on the low as fully seller-driven.

    It is good enough for the *relative* comparison the strategy makes
    ("was the volume behind this breakout mostly buying?") and it must not be
    presented as measured delta. If you need the real thing, extend
    :mod:`data.exchange` with a trades fetcher and replace this function.

    Returns:
        Frame with ``buy_volume``, ``sell_volume`` and ``volume_delta_pct``
        (net buying pressure as % of the bar's volume, ``[-100, 100]``).
    """
    high, low, close = df["high"], df["low"], df["close"]
    volume = df["volume"].fillna(0.0)
    span = (high - low).replace(0.0, np.nan)
    buy_fraction = ((close - low) / span).clip(0.0, 1.0)
    # A zero-range bar carries no directional information: split it evenly.
    buy_fraction = buy_fraction.fillna(0.5)
    buy = volume * buy_fraction
    sell = volume - buy
    return pd.DataFrame(
        {
            "buy_volume": buy,
            "sell_volume": sell,
            "volume_delta_pct": 100.0 * (buy - sell) / volume.replace(0.0, np.nan),
        }
    )


@dataclass(frozen=True, slots=True)
class VolumeProfile:
    """Result of :func:`volume_profile`.

    Attributes:
        poc: Point of Control — the price level with the most traded volume.
        vah: Value Area High — upper bound of the 70% volume band.
        val: Value Area Low — lower bound of the 70% volume band.
        bins: Bin centre prices.
        volumes: Volume in each bin.
        lookback: Number of bars aggregated.
    """

    poc: float
    vah: float
    val: float
    bins: tuple[float, ...]
    volumes: tuple[float, ...]
    lookback: int

    def to_dict(self) -> dict[str, float | int]:
        """Compact form for logging (drops the histogram)."""
        return {"poc": self.poc, "vah": self.vah, "val": self.val, "lookback": self.lookback}


def volume_profile(
    df: pd.DataFrame, bins: int = 24, lookback: int = 120, value_area: float = 0.70
) -> VolumeProfile | None:
    """Approximate a horizontal volume profile over the last ``lookback`` bars.

    Each bar's volume is spread uniformly across the price bins its high-low
    range overlaps. That is an approximation of the true intrabar distribution
    (which needs tick data), but it locates the POC and the value area well
    enough to answer the question the strategy asks: *is price entering a
    high-volume shelf that is likely to stall it, or a low-volume pocket it can
    travel through?*

    Args:
        df: OHLCV frame.
        bins: number of price buckets.
        lookback: bars to aggregate.
        value_area: fraction of total volume defining the value area.

    Returns:
        A :class:`VolumeProfile`, or ``None`` when there is not enough data or
        no volume at all (both are normal for a fresh listing).
    """
    if bins < 2:
        raise ValueError("bins must be >= 2")
    if not 0 < value_area < 1:
        raise ValueError("value_area must be in (0, 1)")

    window = df.iloc[-lookback:]
    if len(window) < 2:
        return None

    low = float(window["low"].min())
    high = float(window["high"].max())
    total_volume = float(window["volume"].fillna(0.0).sum())
    if not np.isfinite(low) or not np.isfinite(high) or high <= low or total_volume <= 0:
        return None

    edges = np.linspace(low, high, bins + 1)
    centres = (edges[:-1] + edges[1:]) / 2.0
    histogram = np.zeros(bins, dtype=float)

    lows = window["low"].to_numpy(dtype=float)
    highs = window["high"].to_numpy(dtype=float)
    volumes = window["volume"].fillna(0.0).to_numpy(dtype=float)

    for bar_low, bar_high, bar_volume in zip(lows, highs, volumes, strict=True):
        if bar_volume <= 0 or not np.isfinite(bar_low) or not np.isfinite(bar_high):
            continue
        first = int(np.searchsorted(edges, bar_low, side="right") - 1)
        last = int(np.searchsorted(edges, bar_high, side="left") - 1)
        first = max(0, min(first, bins - 1))
        last = max(0, min(last, bins - 1))
        touched = last - first + 1
        histogram[first : last + 1] += bar_volume / touched

    poc_index = int(np.argmax(histogram))

    # Grow the value area outward from the POC, always taking the heavier side.
    included = {poc_index}
    accumulated = histogram[poc_index]
    target = value_area * histogram.sum()
    low_index = high_index = poc_index
    while accumulated < target and (low_index > 0 or high_index < bins - 1):
        below = histogram[low_index - 1] if low_index > 0 else -1.0
        above = histogram[high_index + 1] if high_index < bins - 1 else -1.0
        if above >= below:
            high_index += 1
            included.add(high_index)
            accumulated += histogram[high_index]
        else:
            low_index -= 1
            included.add(low_index)
            accumulated += histogram[low_index]

    return VolumeProfile(
        poc=float(centres[poc_index]),
        vah=float(edges[max(included) + 1]),
        val=float(edges[min(included)]),
        bins=tuple(float(x) for x in centres),
        volumes=tuple(float(x) for x in histogram),
        lookback=len(window),
    )
