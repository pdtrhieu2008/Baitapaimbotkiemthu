"""Volume interpretation: spike / dry, buying vs selling, confirmation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from config.settings import IndicatorConfig
from utils.helpers import safe_div

__all__ = ["VolumeState", "analyse_volume"]


@dataclass(frozen=True, slots=True)
class VolumeState:
    """Volume reading for the last bar.

    Attributes:
        relative: current volume divided by its moving average.
        spike: ``relative >= volume_spike_multiplier``.
        dry: ``relative <= volume_dry_multiplier``.
        pressure: net buying pressure in ``[-100, 100]`` (approximated from the
            close's position inside the bar — see
            :func:`indicators.volume.effort_split`).
        obv_rising: OBV above its own EMA, i.e. cumulative flow trending up.
        cmf: Chaikin Money Flow, ``[-1, 1]``.
        mfi: Money Flow Index, ``[0, 100]``.
        bias: ``+1`` buying, ``-1`` selling, ``0`` balanced.
    """

    relative: float
    spike: bool
    dry: bool
    pressure: float
    obv_rising: bool
    cmf: float
    mfi: float
    bias: int

    def confirms(self, side: int) -> bool:
        """Whether volume supports a trade in direction ``side``.

        Confirmation requires participation (a spike) **and** flow pointing the
        same way. A breakout on dry volume is the textbook failure mode, so the
        strategy makes this a hard gate rather than a scored nicety.
        """
        if self.dry:
            return False
        return self.spike and self.bias == side

    def to_dict(self) -> dict[str, object]:
        return {
            "relative": round(self.relative, 2),
            "spike": self.spike,
            "dry": self.dry,
            "pressure": round(self.pressure, 1),
            "obv_rising": self.obv_rising,
            "cmf": round(self.cmf, 3),
            "mfi": round(self.mfi, 1),
            "bias": self.bias,
        }


def _finite(value: object, default: float = float("nan")) -> float:
    """Coerce a possibly-NaN pandas scalar to a plain float."""
    try:
        out = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    return out if np.isfinite(out) else default


def analyse_volume(df: pd.DataFrame, cfg: IndicatorConfig) -> VolumeState:
    """Summarise the volume picture on the last bar of ``df``.

    Args:
        df: frame enriched by :class:`indicators.engine.IndicatorEngine`.
        cfg: the ``indicators`` section (spike / dry multipliers).

    Returns:
        A :class:`VolumeState`. Missing columns degrade to a neutral, non-
        confirming state instead of raising: a symbol with no volume data should
        produce no signal, not an exception.
    """
    if df.empty:
        return VolumeState(float("nan"), False, True, 0.0, False, 0.0, 50.0, 0)

    row = df.iloc[-1]
    relative = _finite(row.get("rel_volume"))
    pressure = _finite(row.get("volume_delta_pct"), 0.0)
    cmf_value = _finite(row.get("cmf"), 0.0)
    mfi_value = _finite(row.get("mfi"), 50.0)
    obv_value = _finite(row.get("obv"))
    obv_ema = _finite(row.get("obv_ema"))

    spike = bool(np.isfinite(relative) and relative >= cfg.volume_spike_multiplier)
    dry = bool(not np.isfinite(relative) or relative <= cfg.volume_dry_multiplier)
    obv_rising = bool(np.isfinite(obv_value) and np.isfinite(obv_ema) and obv_value > obv_ema)

    # Three independent flow measures vote; two must agree for a directional
    # read. One alone is too easy to trip on a single distorted bar.
    votes = 0
    votes += 1 if pressure > 10.0 else (-1 if pressure < -10.0 else 0)
    votes += 1 if cmf_value > 0.05 else (-1 if cmf_value < -0.05 else 0)
    votes += 1 if mfi_value > 55.0 else (-1 if mfi_value < 45.0 else 0)
    bias = 1 if votes >= 2 else (-1 if votes <= -2 else 0)

    return VolumeState(
        relative=relative,
        spike=spike,
        dry=dry,
        pressure=pressure,
        obv_rising=obv_rising,
        cmf=cmf_value,
        mfi=mfi_value,
        bias=bias,
    )


def volume_confirmation_ratio(df: pd.DataFrame, lookback: int = 5) -> float:
    """Fraction of the last ``lookback`` bars whose volume rose with price.

    A rough "is the move being funded?" statistic: values near 1 mean advances
    came on expanding volume, values near 0 mean the move is drifting.
    """
    if len(df) < lookback + 1 or "volume" not in df.columns:
        return 0.0
    window = df.iloc[-lookback:]
    price_up = window["close"].diff() > 0
    volume_up = window["volume"].diff() > 0
    agree = int((price_up == volume_up).sum())
    return safe_div(agree, len(window))
