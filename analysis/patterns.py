"""Candlestick pattern recognition, fully vectorised.

Patterns are produced as boolean **columns** rather than as a verdict on the
last bar. That is deliberate: the backtester walks bar by bar and reads row *i*
of the same frame the live bot reads the last row of, so a pattern can never be
defined differently in the two paths.

Every rule below uses only bars ``<= i``.

A caveat worth stating plainly: candlestick patterns in isolation have weak and
regime-dependent edge. They are used here as *one* scored component out of
eight, and only ever near a structural level — never as a standalone trigger.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "BEARISH_PATTERNS",
    "BULLISH_PATTERNS",
    "NEUTRAL_PATTERNS",
    "PATTERN_COLUMNS",
    "detect_patterns",
    "pattern_bias",
]

#: Patterns that argue for higher prices.
BULLISH_PATTERNS: tuple[str, ...] = (
    "hammer", "pin_bar_bull", "engulfing_bull", "morning_star",
)
#: Patterns that argue for lower prices.
BEARISH_PATTERNS: tuple[str, ...] = (
    "shooting_star", "pin_bar_bear", "engulfing_bear", "evening_star",
)
#: Patterns that describe compression / indecision rather than direction.
NEUTRAL_PATTERNS: tuple[str, ...] = ("doji", "inside_bar", "outside_bar")

#: All columns added by :func:`detect_patterns`.
PATTERN_COLUMNS: tuple[str, ...] = (
    *BULLISH_PATTERNS, *BEARISH_PATTERNS, *NEUTRAL_PATTERNS,
)


def detect_patterns(
    df: pd.DataFrame,
    *,
    doji_body_ratio: float = 0.10,
    pin_wick_ratio: float = 0.60,
    star_body_ratio: float = 0.30,
) -> pd.DataFrame:
    """Return one boolean column per candlestick pattern.

    Args:
        df: OHLCV frame.
        doji_body_ratio: body/range ceiling for a doji.
        pin_wick_ratio: wick/range floor for a pin bar.
        star_body_ratio: body/average-body ceiling for the middle candle of a
            morning/evening star.

    Returns:
        Boolean frame indexed like ``df`` with the columns in
        :data:`PATTERN_COLUMNS`.
    """
    open_, high, low, close = df["open"], df["high"], df["low"], df["close"]

    body = (close - open_).abs()
    span = (high - low).replace(0.0, np.nan)
    upper_wick = high - pd.concat([open_, close], axis=1).max(axis=1)
    lower_wick = pd.concat([open_, close], axis=1).min(axis=1) - low
    bullish = close > open_
    bearish = close < open_
    # Reference body size, so "small body" adapts to the instrument's volatility
    # instead of relying on an absolute threshold.
    avg_body = body.rolling(14, min_periods=5).mean()

    prev_open, prev_close = open_.shift(1), close.shift(1)
    prev_high, prev_low = high.shift(1), low.shift(1)
    prev_body = body.shift(1)
    prev_bullish, prev_bearish = bullish.shift(1, fill_value=False), bearish.shift(1, fill_value=False)

    body_ratio = body / span
    close_position = (close - low) / span  # 0 = closed on the low, 1 = on the high

    # --- single-bar shapes -------------------------------------------------
    doji = body_ratio <= doji_body_ratio

    # Hammer: long lower wick, small upper wick, body in the upper part of the
    # range. The strategy only counts it when it appears at a demand level.
    hammer = (
        (lower_wick >= 2.0 * body)
        & (upper_wick <= body)
        & (close_position >= 0.6)
        & ~doji
    )
    shooting_star = (
        (upper_wick >= 2.0 * body)
        & (lower_wick <= body)
        & (close_position <= 0.4)
        & ~doji
    )

    # Pin bars are the looser, wick-dominance version of the above.
    pin_bar_bull = (lower_wick / span >= pin_wick_ratio) & (close_position >= 0.6)
    pin_bar_bear = (upper_wick / span >= pin_wick_ratio) & (close_position <= 0.4)

    # --- two-bar relationships --------------------------------------------
    inside_bar = (high < prev_high) & (low > prev_low)
    outside_bar = (high > prev_high) & (low < prev_low)

    engulfing_bull = (
        prev_bearish & bullish & (close >= prev_open) & (open_ <= prev_close) & (body > prev_body)
    )
    engulfing_bear = (
        prev_bullish & bearish & (close <= prev_open) & (open_ >= prev_close) & (body > prev_body)
    )

    # --- three-bar reversals ----------------------------------------------
    # bar-2 impulse, bar-1 small-bodied pause, bar-0 reversal closing back
    # through the midpoint of the impulse candle.
    first_open, first_close = open_.shift(2), close.shift(2)
    first_body = body.shift(2)
    middle_body = body.shift(1)
    small_middle = middle_body <= star_body_ratio * avg_body.shift(1)
    big_first = first_body >= avg_body.shift(2)

    first_midpoint = (first_open + first_close) / 2.0
    morning_star = (
        (first_close < first_open)  # bar-2 bearish
        & big_first
        & small_middle
        & bullish
        & (close > first_midpoint)
        & (pd.concat([open_.shift(1), close.shift(1)], axis=1).min(axis=1) < first_close)
    )
    evening_star = (
        (first_close > first_open)  # bar-2 bullish
        & big_first
        & small_middle
        & bearish
        & (close < first_midpoint)
        & (pd.concat([open_.shift(1), close.shift(1)], axis=1).max(axis=1) > first_close)
    )

    frame = pd.DataFrame(
        {
            "hammer": hammer,
            "pin_bar_bull": pin_bar_bull,
            "engulfing_bull": engulfing_bull,
            "morning_star": morning_star,
            "shooting_star": shooting_star,
            "pin_bar_bear": pin_bar_bear,
            "engulfing_bear": engulfing_bear,
            "evening_star": evening_star,
            "doji": doji,
            "inside_bar": inside_bar,
            "outside_bar": outside_bar,
        },
        index=df.index,
    )
    # A NaN comparison (zero-range bar, or the first bars of the series) means
    # "pattern not established", which is False.
    return frame.fillna(False).astype(bool)[list(PATTERN_COLUMNS)]


def pattern_bias(row: pd.Series) -> tuple[int, list[str]]:
    """Net directional vote of the patterns present on one bar.

    Args:
        row: a row of the frame returned by :func:`detect_patterns`.

    Returns:
        ``(bias, names)`` where ``bias`` is ``+1``/``-1``/``0`` and ``names``
        lists the directional patterns that fired. A bar showing both a bullish
        and a bearish pattern nets to ``0``: genuinely ambiguous, so the scorer
        awards it nothing rather than picking a side.
    """
    bulls = [name for name in BULLISH_PATTERNS if bool(row.get(name, False))]
    bears = [name for name in BEARISH_PATTERNS if bool(row.get(name, False))]
    if len(bulls) > len(bears):
        return 1, bulls
    if len(bears) > len(bulls):
        return -1, bears
    return 0, [*bulls, *bears]
