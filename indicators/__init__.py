"""In-house technical indicator library (pandas/numpy only).

Sub-modules group indicators by what they measure:

* :mod:`indicators.trend` - SMA, EMA, WMA, VWAP, ADX/DI, Supertrend, Ichimoku
* :mod:`indicators.momentum` - RSI, MACD, Stochastic, Stochastic RSI, ROC
* :mod:`indicators.volatility` - ATR/NATR, Bollinger, Keltner, Donchian, squeeze
* :mod:`indicators.volume` - OBV, CMF, MFI, relative volume, volume profile
* :mod:`indicators.levels` - classic / Fibonacci pivot points

:class:`indicators.engine.IndicatorEngine` computes all of them in one pass and
is the only class the rest of the project talks to.
"""

from indicators.base import OHLCV_COLUMNS, true_range, validate_ohlcv, wilder_smooth
from indicators.engine import IndicatorEngine

__all__ = [
    "OHLCV_COLUMNS",
    "IndicatorEngine",
    "true_range",
    "validate_ohlcv",
    "wilder_smooth",
]
