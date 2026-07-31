"""Fractal swing-point detection — the foundation of the structure module.

A swing high at bar *i* requires ``strength`` lower highs on **each** side. The
right-hand side is the important part: a swing can only be *confirmed*
``strength`` bars after it printed. Treating an unconfirmed extreme as a swing
is a subtle lookahead bug — the backtest would "know" a top formed before the
market did — so :func:`find_swings` drops the unconfirmed tail by default.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal

import numpy as np
import pandas as pd

__all__ = ["Swing", "find_swings", "last_swing", "swing_series"]

SwingKind = Literal["high", "low"]


@dataclass(frozen=True, slots=True)
class Swing:
    """A confirmed pivot in price.

    Attributes:
        position: integer position in the source frame.
        timestamp: bar-open time of the pivot.
        price: the extreme (high for a swing high, low for a swing low).
        kind: ``"high"`` or ``"low"``.
    """

    position: int
    timestamp: datetime
    price: float
    kind: SwingKind

    def to_dict(self) -> dict[str, object]:
        return {
            "position": self.position,
            "timestamp": self.timestamp.isoformat(),
            "price": self.price,
            "kind": self.kind,
        }


def _fractal_mask(values: np.ndarray, strength: int, *, find_high: bool) -> np.ndarray:
    """Boolean mask of fractal extremes in ``values``.

    Ties are broken toward the earlier bar: the centre must be strictly beyond
    every bar to its left and at least as extreme as every bar to its right.
    Without that rule a flat double top would register two swings at the same
    price and corrupt the HH/LL sequence.

    Vectorised with a sliding window because this runs once per timeframe per
    bar in a backtest; the equivalent Python loop dominated the profile.
    ``NaN`` centres compare ``False`` and are therefore skipped automatically.
    """
    n = values.size
    width = 2 * strength + 1
    mask = np.zeros(n, dtype=bool)
    if n < width:
        return mask

    windows = np.lib.stride_tricks.sliding_window_view(values, width)
    centre = windows[:, strength]
    left = windows[:, :strength]
    right = windows[:, strength + 1 :]

    with np.errstate(invalid="ignore"):
        if find_high:
            found = (centre > left.max(axis=1)) & (centre >= right.max(axis=1))
        else:
            found = (centre < left.min(axis=1)) & (centre <= right.min(axis=1))

    mask[strength : n - strength] = found
    return mask


def find_swings(
    df: pd.DataFrame,
    strength: int = 3,
    *,
    confirmed_only: bool = True,
    lookback: int | None = None,
) -> list[Swing]:
    """Locate swing highs and lows, oldest first.

    Args:
        df: OHLCV frame.
        strength: bars required on each side of the pivot.
        confirmed_only: drop pivots in the final ``strength`` bars, which do not
            yet have enough right-hand bars to be confirmed. Keep this ``True``
            for anything that feeds a trading decision.
        lookback: only consider the final ``lookback`` bars.

    Returns:
        Chronologically ordered list of :class:`Swing`.

    Raises:
        ValueError: if ``strength < 1``.
    """
    if strength < 1:
        raise ValueError("strength must be >= 1")

    window = df if lookback is None else df.iloc[-lookback:]
    offset = len(df) - len(window)
    highs = window["high"].to_numpy(dtype=float)
    lows = window["low"].to_numpy(dtype=float)
    timestamps = window.index

    high_mask = _fractal_mask(highs, strength, find_high=True)
    low_mask = _fractal_mask(lows, strength, find_high=False)

    if confirmed_only and strength > 0:
        high_mask[-strength:] = False
        low_mask[-strength:] = False

    swings: list[Swing] = []
    for position in np.flatnonzero(high_mask):
        swings.append(
            Swing(offset + int(position), timestamps[position].to_pydatetime(),
                  float(highs[position]), "high")
        )
    for position in np.flatnonzero(low_mask):
        swings.append(
            Swing(offset + int(position), timestamps[position].to_pydatetime(),
                  float(lows[position]), "low")
        )
    swings.sort(key=lambda s: s.position)
    return swings


def swing_series(swings: list[Swing], kind: SwingKind) -> list[Swing]:
    """Filter a swing list down to one kind, preserving order."""
    return [s for s in swings if s.kind == kind]


def last_swing(swings: list[Swing], kind: SwingKind) -> Swing | None:
    """Most recent swing of the requested kind, or ``None``."""
    for swing in reversed(swings):
        if swing.kind == kind:
            return swing
    return None
