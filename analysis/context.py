"""Per-timeframe analysis context and the builder that produces it.

:class:`TimeframeContext` is the single object every strategy component reads.
Building it in one place means the live loop and the backtester cannot diverge:
the backtester enriches the whole series once and then asks the builder for the
context of each bar in turn, while the live loop asks for the context of the
latest closed bar.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd

from analysis.patterns import PATTERN_COLUMNS, detect_patterns, pattern_bias
from analysis.regime import RegimeReport, classify_regime
from analysis.structure import StructureReport, analyse_structure
from analysis.volume_analysis import VolumeState, analyse_volume
from config.settings import IndicatorConfig, StructureConfig
from data.models import ExternalContext
from indicators.engine import IndicatorEngine
from utils.helpers import safe_div
from utils.timeframes import timeframe_to_seconds

__all__ = ["ContextBuilder", "MarketSnapshot", "TimeframeContext"]

#: Indicator columns copied into the flat snapshot used by the scorer and the
#: notification. Kept explicit so a rename in the engine surfaces immediately.
_SNAPSHOT_COLUMNS: tuple[str, ...] = (
    "open", "high", "low", "close", "volume",
    "ema_fast", "ema_slow", "ema_trend", "sma", "ema_slow_slope", "vwap",
    "adx", "plus_di", "minus_di", "supertrend", "supertrend_dir",
    "tenkan", "kijun", "senkou_a", "senkou_b", "cloud_top", "cloud_bottom",
    "rsi", "macd", "macd_signal", "macd_hist",
    "stoch_k", "stoch_d", "stochrsi_k", "stochrsi_d",
    "atr", "natr", "atr_median",
    "bb_upper", "bb_middle", "bb_lower", "bb_width", "bb_pct",
    "kc_upper", "kc_lower", "dc_upper", "dc_middle", "dc_lower",
    "obv", "obv_ema", "cmf", "mfi", "volume_sma", "rel_volume",
    "buy_volume", "sell_volume", "volume_delta_pct",
    "pivot", "pivot_r1", "pivot_r2", "pivot_s1", "pivot_s2",
)


def _as_float(value: object) -> float:
    try:
        out = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return float("nan")
    return out


@dataclass(slots=True)
class TimeframeContext:
    """Complete analysis of one symbol on one timeframe, at one closed bar."""

    symbol: str
    timeframe: str
    bar_time: datetime
    values: dict[str, float]
    patterns: dict[str, bool]
    pattern_names: list[str]
    pattern_bias: int
    structure: StructureReport
    volume: VolumeState
    regime: RegimeReport
    bias: int
    bias_score: float
    #: Rows up to and including ``bar_time``. Needed by the trigger detector,
    #: which looks a few bars back. Excluded from ``repr`` to keep logs sane.
    frame: pd.DataFrame = field(repr=False, default_factory=pd.DataFrame)

    @property
    def close(self) -> float:
        return self.values.get("close", float("nan"))

    @property
    def atr(self) -> float:
        return self.values.get("atr", float("nan"))

    def value(self, name: str, default: float = float("nan")) -> float:
        """Indicator value with a NaN-safe default."""
        out = self.values.get(name, default)
        return out if np.isfinite(out) else default

    @property
    def is_complete(self) -> bool:
        """Whether the core indicators are seeded.

        A context missing any of these cannot be scored honestly, so the
        strategy refuses it outright instead of treating NaN as neutral.
        """
        required = ("close", "atr", "ema_fast", "ema_slow", "ema_trend", "adx", "rsi", "macd_hist")
        return all(np.isfinite(self.values.get(name, float("nan"))) for name in required)

    def to_dict(self) -> dict[str, object]:
        """Log-friendly summary (drops the frame and the raw indicator dump)."""
        return {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "bar_time": self.bar_time.isoformat(),
            "close": self.close,
            "bias": self.bias,
            "bias_score": round(self.bias_score, 3),
            "patterns": self.pattern_names,
            "structure": self.structure.to_dict(),
            "volume": self.volume.to_dict(),
            "regime": self.regime.to_dict(),
        }


@dataclass(slots=True)
class MarketSnapshot:
    """All timeframes of one symbol, plus its non-OHLCV context."""

    symbol: str
    primary_timeframe: str
    contexts: dict[str, TimeframeContext]
    external: ExternalContext

    @property
    def primary(self) -> TimeframeContext:
        """Context of the timeframe the entry decision is taken on."""
        try:
            return self.contexts[self.primary_timeframe]
        except KeyError as exc:
            raise KeyError(
                f"primary timeframe {self.primary_timeframe!r} missing from snapshot; "
                f"available: {sorted(self.contexts)}"
            ) from exc

    def get(self, timeframe: str) -> TimeframeContext | None:
        return self.contexts.get(timeframe)

    @property
    def bar_time(self) -> datetime:
        return self.primary.bar_time

    def to_dict(self) -> dict[str, object]:
        return {
            "symbol": self.symbol,
            "primary_timeframe": self.primary_timeframe,
            "contexts": {tf: ctx.to_dict() for tf, ctx in self.contexts.items()},
            "external": self.external.to_dict(),
        }


class ContextBuilder:
    """Turn raw OHLCV into a :class:`TimeframeContext`.

    Args:
        indicator_cfg: the ``indicators`` configuration section.
        structure_cfg: the ``structure`` configuration section.
    """

    def __init__(self, indicator_cfg: IndicatorConfig, structure_cfg: StructureConfig) -> None:
        self.indicator_cfg = indicator_cfg
        self.structure_cfg = structure_cfg
        self.engine = IndicatorEngine(indicator_cfg)

    # -- stage 1: expensive, done once per series ---------------------------
    def enrich(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """Add indicator and pattern columns to a raw OHLCV frame.

        Args:
            df: raw OHLCV.
            timeframe: used to decide whether pivot levels are meaningful.

        Returns:
            The enriched frame.
        """
        # Pivots derived from a daily period are degenerate on a >=1d chart:
        # every bar would map to a single previous bar.
        with_pivots = timeframe_to_seconds(timeframe) < timeframe_to_seconds(
            self.indicator_cfg.pivot_timeframe
        )
        enriched = self.engine.compute(df, with_pivots=with_pivots)
        return enriched.join(detect_patterns(enriched))

    # -- stage 2: cheap, done per bar ---------------------------------------
    def build(
        self, symbol: str, timeframe: str, enriched: pd.DataFrame
    ) -> TimeframeContext:
        """Build the context for the **last** row of ``enriched``.

        Args:
            symbol: unified symbol.
            timeframe: the timeframe this frame represents.
            enriched: output of :meth:`enrich`, sliced so its last row is the
                bar being evaluated.

        Returns:
            A :class:`TimeframeContext`.

        Raises:
            ValueError: if ``enriched`` is empty or was not passed through
                :meth:`enrich`.
        """
        if enriched.empty:
            raise ValueError(f"cannot build context for {symbol} {timeframe}: empty frame")
        if "atr" not in enriched.columns:
            raise ValueError(
                "frame was not enriched; call ContextBuilder.enrich() before build()"
            )

        row = enriched.iloc[-1]
        values = {
            name: _as_float(row[name]) for name in _SNAPSHOT_COLUMNS if name in enriched.columns
        }
        patterns = {name: bool(row.get(name, False)) for name in PATTERN_COLUMNS}
        bias_direction, names = pattern_bias(row)

        structure = analyse_structure(enriched, self.structure_cfg)
        volume_state = analyse_volume(enriched, self.indicator_cfg)
        regime = classify_regime(enriched, self.indicator_cfg)
        score = self._bias_score(values, structure, regime)

        return TimeframeContext(
            symbol=symbol,
            timeframe=timeframe,
            bar_time=enriched.index[-1].to_pydatetime(),
            values=values,
            patterns=patterns,
            pattern_names=names,
            pattern_bias=bias_direction,
            structure=structure,
            volume=volume_state,
            regime=regime,
            bias=1 if score >= 0.25 else (-1 if score <= -0.25 else 0),
            bias_score=score,
            frame=enriched,
        )

    @staticmethod
    def _bias_score(
        values: dict[str, float], structure: StructureReport, regime: RegimeReport
    ) -> float:
        """Weighted directional vote in ``[-1, 1]``.

        This single number is what multi-timeframe alignment compares across
        timeframes, so it deliberately blends independent families of evidence
        (moving averages, a trailing stop, the Ichimoku cloud, momentum, swing
        structure and the regime classifier) instead of trusting any one.

        Components whose inputs are NaN are skipped *and* removed from the
        denominator, so an early bar produces a small-sample score rather than
        an artificially damped one.
        """
        votes: list[tuple[int, float]] = []

        ema_fast, ema_slow = values.get("ema_fast", np.nan), values.get("ema_slow", np.nan)
        close, ema_trend = values.get("close", np.nan), values.get("ema_trend", np.nan)
        if np.isfinite(ema_fast) and np.isfinite(ema_slow):
            votes.append((1 if ema_fast > ema_slow else -1, 2.0))
        if np.isfinite(close) and np.isfinite(ema_trend):
            votes.append((1 if close > ema_trend else -1, 2.0))

        supertrend_dir = values.get("supertrend_dir", np.nan)
        if np.isfinite(supertrend_dir) and supertrend_dir != 0:
            votes.append((int(np.sign(supertrend_dir)), 1.0))

        cloud_top, cloud_bottom = values.get("cloud_top", np.nan), values.get("cloud_bottom", np.nan)
        if np.isfinite(close) and np.isfinite(cloud_top) and np.isfinite(cloud_bottom):
            if close > cloud_top:
                votes.append((1, 1.0))
            elif close < cloud_bottom:
                votes.append((-1, 1.0))
            else:
                votes.append((0, 1.0))  # inside the cloud: explicitly undecided

        macd_hist = values.get("macd_hist", np.nan)
        if np.isfinite(macd_hist) and macd_hist != 0:
            votes.append((int(np.sign(macd_hist)), 1.0))

        votes.append((structure.bias, 2.0))
        votes.append((regime.regime.bias, 1.0))

        total_weight = sum(weight for _, weight in votes)
        signed = sum(direction * weight for direction, weight in votes)
        return float(np.clip(safe_div(signed, total_weight), -1.0, 1.0))
