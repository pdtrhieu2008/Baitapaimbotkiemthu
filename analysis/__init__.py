"""Market interpretation layer: price action, structure, volume, regime.

The indicator layer answers "what are the numbers?". This layer answers "what
do they mean?", and packages the answer as a :class:`analysis.context.TimeframeContext`
that the strategy layer scores.
"""

from analysis.context import ContextBuilder, MarketSnapshot, TimeframeContext
from analysis.patterns import (
    BEARISH_PATTERNS,
    BULLISH_PATTERNS,
    PATTERN_COLUMNS,
    detect_patterns,
    pattern_bias,
)
from analysis.regime import (
    MarketRegime,
    RegimeReport,
    TrendStrength,
    VolatilityState,
    classify_regime,
)
from analysis.structure import (
    BreakEvent,
    BreakKind,
    Level,
    StructureReport,
    StructureTrend,
    Zone,
    analyse_structure,
)
from analysis.swings import Swing, find_swings
from analysis.volume_analysis import VolumeState, analyse_volume

__all__ = [
    "BEARISH_PATTERNS",
    "BULLISH_PATTERNS",
    "PATTERN_COLUMNS",
    "BreakEvent",
    "BreakKind",
    "ContextBuilder",
    "Level",
    "MarketRegime",
    "MarketSnapshot",
    "RegimeReport",
    "StructureReport",
    "StructureTrend",
    "Swing",
    "TimeframeContext",
    "TrendStrength",
    "VolatilityState",
    "VolumeState",
    "Zone",
    "analyse_structure",
    "analyse_volume",
    "classify_regime",
    "detect_patterns",
    "find_swings",
    "pattern_bias",
]
