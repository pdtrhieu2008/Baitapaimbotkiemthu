"""Market regime classification: trend vs range, strength, volatility, exhaustion.

Regime is the answer to "should this kind of strategy be trading at all right
now?". A confluence-breakout system needs an expanding, directional market; the
same rules applied inside a tight range produce a stream of losing trades. The
classifier below is therefore consulted *before* the score is even computed.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd

from config.settings import IndicatorConfig
from utils.helpers import safe_div

__all__ = [
    "MarketRegime",
    "RegimeReport",
    "TrendStrength",
    "VolatilityState",
    "classify_regime",
]


class MarketRegime(str, Enum):
    """Coarse market state."""

    TREND_UP = "trend_up"
    TREND_DOWN = "trend_down"
    RANGE = "range"        # bounded but active, mean-reverting
    SIDEWAYS = "sideways"  # bounded and quiet, compressed volatility

    @property
    def bias(self) -> int:
        return {"trend_up": 1, "trend_down": -1, "range": 0, "sideways": 0}[self.value]

    @property
    def is_trending(self) -> bool:
        return self in (MarketRegime.TREND_UP, MarketRegime.TREND_DOWN)


class TrendStrength(str, Enum):
    """How convincing the trend is, from ADX and the EMA stack."""

    NONE = "none"
    WEAK = "weak"
    STRONG = "strong"


class VolatilityState(str, Enum):
    """ATR relative to its own recent median."""

    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    EXTREME = "extreme"  # news / liquidation cascade - stand aside


@dataclass(frozen=True, slots=True)
class RegimeReport:
    """Regime verdict for the last bar.

    Attributes:
        regime: coarse state.
        strength: trend conviction.
        volatility: volatility bucket.
        adx: raw ADX value.
        natr: ATR as % of price.
        atr_ratio: ATR divided by its rolling median.
        momentum: normalised MACD histogram, in ATR units.
        exhaustion: trend is stretched and losing momentum.
        squeeze: Bollinger inside Keltner — compression, expansion pending.
    """

    regime: MarketRegime
    strength: TrendStrength
    volatility: VolatilityState
    adx: float
    natr: float
    atr_ratio: float
    momentum: float
    exhaustion: bool
    squeeze: bool

    @property
    def tradable(self) -> bool:
        """Whether a breakout/continuation system should be active.

        Extreme volatility is excluded because stop distances become
        unpredictable and slippage stops resembling the backtest's assumptions.
        A live squeeze is excluded because the direction of the release is
        unknown by definition.
        """
        return (
            self.volatility is not VolatilityState.EXTREME
            and self.strength is not TrendStrength.NONE
            and not self.squeeze
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "regime": self.regime.value,
            "strength": self.strength.value,
            "volatility": self.volatility.value,
            "adx": round(self.adx, 1),
            "natr": round(self.natr, 3),
            "atr_ratio": round(self.atr_ratio, 2),
            "momentum": round(self.momentum, 3),
            "exhaustion": self.exhaustion,
            "squeeze": self.squeeze,
            "tradable": self.tradable,
        }


def _finite(value: object, default: float) -> float:
    try:
        out = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    return out if np.isfinite(out) else default


def _detect_exhaustion(df: pd.DataFrame, direction: int) -> bool:
    """Trend still extended but momentum already rolling over.

    Two independent symptoms are required, because either one alone fires
    constantly in a healthy trend:

    1. price is stretched from the slow EMA by more than ~2.5 ATR, and
    2. the MACD histogram is shrinking while price still makes new extremes
       (classic momentum divergence).
    """
    if direction == 0 or len(df) < 6:
        return False

    row = df.iloc[-1]
    atr = _finite(row.get("atr"), float("nan"))
    ema_slow = _finite(row.get("ema_slow"), float("nan"))
    close = _finite(row.get("close"), float("nan"))
    if not all(np.isfinite(x) for x in (atr, ema_slow, close)) or atr <= 0:
        return False

    stretched = abs(close - ema_slow) / atr > 2.5

    histogram = df["macd_hist"].tail(4) if "macd_hist" in df.columns else None
    if histogram is None or histogram.isna().any():
        return False
    signed = histogram * direction
    momentum_fading = bool(signed.iloc[-1] < signed.iloc[-2] < signed.iloc[-3])

    extreme_column = "high" if direction > 0 else "low"
    recent = df[extreme_column].tail(4)
    still_extending = bool(
        recent.iloc[-1] == recent.max() if direction > 0 else recent.iloc[-1] == recent.min()
    )

    return stretched and momentum_fading and still_extending


def classify_regime(df: pd.DataFrame, cfg: IndicatorConfig) -> RegimeReport:
    """Classify the regime on the last bar of ``df``.

    Args:
        df: frame enriched by :class:`indicators.engine.IndicatorEngine`.
        cfg: the ``indicators`` section (ADX and ATR thresholds).

    Returns:
        A :class:`RegimeReport`. Missing indicators degrade to a non-tradable
        neutral verdict.
    """
    neutral = RegimeReport(
        MarketRegime.SIDEWAYS, TrendStrength.NONE, VolatilityState.NORMAL,
        0.0, 0.0, 1.0, 0.0, False, False,
    )
    if df.empty:
        return neutral

    row = df.iloc[-1]
    adx = _finite(row.get("adx"), float("nan"))
    natr = _finite(row.get("natr"), float("nan"))
    atr = _finite(row.get("atr"), float("nan"))
    atr_median = _finite(row.get("atr_median"), float("nan"))
    if not np.isfinite(adx) or not np.isfinite(natr):
        return neutral

    # --- direction: EMA stack, Supertrend and DI must broadly agree ---
    close = _finite(row.get("close"), float("nan"))
    ema_fast = _finite(row.get("ema_fast"), float("nan"))
    ema_slow = _finite(row.get("ema_slow"), float("nan"))
    ema_trend = _finite(row.get("ema_trend"), float("nan"))
    supertrend_dir = _finite(row.get("supertrend_dir"), 0.0)
    plus_di = _finite(row.get("plus_di"), 0.0)
    minus_di = _finite(row.get("minus_di"), 0.0)

    votes = 0
    if np.isfinite(ema_fast) and np.isfinite(ema_slow):
        votes += 1 if ema_fast > ema_slow else -1
    if np.isfinite(close) and np.isfinite(ema_trend):
        votes += 1 if close > ema_trend else -1
    votes += int(np.sign(supertrend_dir))
    votes += 1 if plus_di > minus_di else -1
    direction = 1 if votes >= 2 else (-1 if votes <= -2 else 0)

    # --- strength ---
    if adx >= cfg.adx_strong:
        strength = TrendStrength.STRONG
    elif adx >= cfg.adx_trend_min:
        strength = TrendStrength.WEAK if direction == 0 else TrendStrength.STRONG
    elif adx >= cfg.adx_trend_min * 0.7:
        strength = TrendStrength.WEAK
    else:
        strength = TrendStrength.NONE

    # A directionless market is never "strong", whatever ADX says.
    if direction == 0 and strength is TrendStrength.STRONG:
        strength = TrendStrength.WEAK

    # --- volatility ---
    atr_ratio = safe_div(atr, atr_median, 1.0) if np.isfinite(atr_median) else 1.0
    if natr < cfg.atr_min_pct:
        volatility = VolatilityState.LOW
    elif natr > cfg.atr_max_pct or atr_ratio >= 3.0:
        volatility = VolatilityState.EXTREME
    elif atr_ratio >= 1.6:
        volatility = VolatilityState.HIGH
    else:
        volatility = VolatilityState.NORMAL

    squeeze = bool(row.get("squeeze_on", False))

    # --- regime ---
    if strength is not TrendStrength.NONE and direction > 0:
        regime = MarketRegime.TREND_UP
    elif strength is not TrendStrength.NONE and direction < 0:
        regime = MarketRegime.TREND_DOWN
    elif volatility is VolatilityState.LOW or squeeze:
        regime = MarketRegime.SIDEWAYS
    else:
        regime = MarketRegime.RANGE

    momentum = safe_div(_finite(row.get("macd_hist"), 0.0), atr) if np.isfinite(atr) else 0.0

    return RegimeReport(
        regime=regime,
        strength=strength,
        volatility=volatility,
        adx=adx,
        natr=natr,
        atr_ratio=atr_ratio,
        momentum=momentum,
        exhaustion=_detect_exhaustion(df, direction),
        squeeze=squeeze,
    )
