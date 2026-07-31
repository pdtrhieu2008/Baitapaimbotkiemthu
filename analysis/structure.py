"""Market structure: HH/HL/LH/LL, BOS, CHoCH, S&R, zones, sweeps.

Vocabulary used consistently across the project:

* **HH / HL / LH / LL** — each new confirmed swing compared with the previous
  swing of the same kind.
* **BOS** (Break of Structure) — price closes beyond the last confirmed swing
  *in the direction of the prevailing trend*: continuation.
* **CHoCH** (Change of Character) — price closes beyond the last confirmed swing
  *against* the prevailing trend: the first evidence the trend is over. A CHoCH
  is what turns a counter-trend setup from "fighting the trend" into "trading
  the new one".
* **Liquidity sweep** — a wick pushes through a swing extreme to trigger stops,
  then the bar closes back inside. Bullish when the *lows* are swept.
* **False breakout** — a close beyond the range boundary that is given back
  within a few bars.

Everything is computed from confirmed swings only (see :mod:`analysis.swings`),
which is what keeps the module causal.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import pandas as pd

from analysis.swings import Swing, find_swings, last_swing
from config.settings import StructureConfig
from utils.helpers import safe_div

__all__ = [
    "BreakEvent",
    "BreakKind",
    "Level",
    "StructureReport",
    "StructureTrend",
    "Zone",
    "analyse_structure",
]


class StructureTrend(str, Enum):
    """Trend as read from the swing sequence alone (no moving averages)."""

    UP = "up"
    DOWN = "down"
    RANGE = "range"

    @property
    def bias(self) -> int:
        return {"up": 1, "down": -1, "range": 0}[self.value]


class BreakKind(str, Enum):
    """Classification of the most recent structural break."""

    NONE = "none"
    BOS_BULL = "bos_bull"
    BOS_BEAR = "bos_bear"
    CHOCH_BULL = "choch_bull"
    CHOCH_BEAR = "choch_bear"

    @property
    def bias(self) -> int:
        if self in (BreakKind.BOS_BULL, BreakKind.CHOCH_BULL):
            return 1
        if self in (BreakKind.BOS_BEAR, BreakKind.CHOCH_BEAR):
            return -1
        return 0

    @property
    def is_choch(self) -> bool:
        return self in (BreakKind.CHOCH_BULL, BreakKind.CHOCH_BEAR)


@dataclass(frozen=True, slots=True)
class BreakEvent:
    """Where and how structure was broken."""

    kind: BreakKind
    price: float
    position: int
    bars_ago: int

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind.value,
            "price": self.price,
            "bars_ago": self.bars_ago,
        }


@dataclass(frozen=True, slots=True)
class Level:
    """A clustered horizontal level.

    Attributes:
        price: cluster centre (volume-free mean of its swing prices).
        kind: ``"support"`` or ``"resistance"`` relative to the current close.
        touches: bars whose high or low came within the cluster tolerance.
        swings: how many confirmed swings formed the cluster.
        last_touch_bars_ago: recency, in bars.
    """

    price: float
    kind: str
    touches: int
    swings: int
    last_touch_bars_ago: int

    @property
    def strength(self) -> float:
        """Heuristic 0-1 score: more touches and more recent is stronger."""
        touch_score = min(self.touches / 5.0, 1.0)
        recency = 1.0 / (1.0 + self.last_touch_bars_ago / 50.0)
        return round(0.7 * touch_score + 0.3 * recency, 3)

    def to_dict(self) -> dict[str, object]:
        return {
            "price": self.price,
            "kind": self.kind,
            "touches": self.touches,
            "strength": self.strength,
        }


@dataclass(frozen=True, slots=True)
class Zone:
    """A supply or demand area, thickness scaled by ATR."""

    low: float
    high: float
    kind: str  # "supply" | "demand"
    bars_ago: int

    def contains(self, price: float) -> bool:
        return self.low <= price <= self.high

    def to_dict(self) -> dict[str, object]:
        return {"low": self.low, "high": self.high, "kind": self.kind, "bars_ago": self.bars_ago}


@dataclass(slots=True)
class StructureReport:
    """Everything the strategy needs to know about structure on one timeframe."""

    trend: StructureTrend = StructureTrend.RANGE
    labels: list[str] = field(default_factory=list)
    swings: list[Swing] = field(default_factory=list)
    last_swing_high: float | None = None
    last_swing_low: float | None = None
    break_event: BreakEvent = field(
        default_factory=lambda: BreakEvent(BreakKind.NONE, float("nan"), -1, -1)
    )
    supports: list[Level] = field(default_factory=list)
    resistances: list[Level] = field(default_factory=list)
    zones: list[Zone] = field(default_factory=list)
    nearest_support: Level | None = None
    nearest_resistance: Level | None = None
    room_down_atr: float = float("inf")
    room_up_atr: float = float("inf")
    in_demand_zone: bool = False
    in_supply_zone: bool = False
    sweep_direction: int = 0
    false_breakout_direction: int = 0

    @property
    def bias(self) -> int:
        """Net structural bias, break events overriding a stale swing trend."""
        if self.break_event.kind is not BreakKind.NONE:
            return self.break_event.kind.bias
        return self.trend.bias

    def room_for(self, side: int) -> float:
        """ATR multiples of clear air in the direction of the trade."""
        return self.room_up_atr if side > 0 else self.room_down_atr

    def to_dict(self) -> dict[str, object]:
        """Compact, log-friendly summary."""
        return {
            "trend": self.trend.value,
            "labels": self.labels[-4:],
            "break": self.break_event.to_dict(),
            "nearest_support": self.nearest_support.to_dict() if self.nearest_support else None,
            "nearest_resistance": (
                self.nearest_resistance.to_dict() if self.nearest_resistance else None
            ),
            "room_up_atr": round(self.room_up_atr, 2),
            "room_down_atr": round(self.room_down_atr, 2),
            "in_demand_zone": self.in_demand_zone,
            "in_supply_zone": self.in_supply_zone,
            "sweep": self.sweep_direction,
            "false_breakout": self.false_breakout_direction,
        }


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------
def _label_swings(swings: list[Swing]) -> list[str]:
    """Label each swing HH/LH/HL/LL against the previous swing of its kind."""
    labels: list[str] = []
    previous: dict[str, float] = {}
    for swing in swings:
        reference = previous.get(swing.kind)
        if reference is None:
            labels.append("H" if swing.kind == "high" else "L")
        elif swing.kind == "high":
            labels.append("HH" if swing.price > reference else "LH")
        else:
            labels.append("HL" if swing.price > reference else "LL")
        previous[swing.kind] = swing.price
    return labels


def _trend_from_labels(labels: list[str]) -> StructureTrend:
    """Derive the trend from the two most recent high and low labels.

    An uptrend needs both a higher high and a higher low: rising highs on flat
    lows is an expanding range, not a trend, and treating it as one is how
    breakout systems get chopped up.
    """
    highs = [lab for lab in labels if lab in ("HH", "LH")]
    lows = [lab for lab in labels if lab in ("HL", "LL")]
    if not highs or not lows:
        return StructureTrend.RANGE
    if highs[-1] == "HH" and lows[-1] == "HL":
        return StructureTrend.UP
    if highs[-1] == "LH" and lows[-1] == "LL":
        return StructureTrend.DOWN
    return StructureTrend.RANGE


def _detect_break(
    df: pd.DataFrame, swings: list[Swing], trend: StructureTrend
) -> BreakEvent:
    """Find the most recent close beyond the last confirmed swing extreme."""
    none = BreakEvent(BreakKind.NONE, float("nan"), -1, -1)
    close = df["close"].to_numpy(dtype=float)
    n = len(close)

    swing_high = last_swing(swings, "high")
    swing_low = last_swing(swings, "low")

    def _first_cross(start: int, level: float, upward: bool) -> int | None:
        segment = close[start + 1 :]
        if segment.size == 0:
            return None
        hits = np.flatnonzero(segment > level) if upward else np.flatnonzero(segment < level)
        return start + 1 + int(hits[0]) if hits.size else None

    bull_pos = (
        _first_cross(swing_high.position, swing_high.price, upward=True) if swing_high else None
    )
    bear_pos = (
        _first_cross(swing_low.position, swing_low.price, upward=False) if swing_low else None
    )

    if bull_pos is None and bear_pos is None:
        return none
    if bear_pos is None or (bull_pos is not None and bull_pos >= bear_pos):
        assert swing_high is not None  # noqa: S101 - bull_pos implies swing_high
        kind = BreakKind.BOS_BULL if trend is StructureTrend.UP else BreakKind.CHOCH_BULL
        return BreakEvent(kind, swing_high.price, bull_pos, n - 1 - bull_pos)
    assert swing_low is not None  # noqa: S101 - bear_pos implies swing_low
    kind = BreakKind.BOS_BEAR if trend is StructureTrend.DOWN else BreakKind.CHOCH_BEAR
    return BreakEvent(kind, swing_low.price, bear_pos, n - 1 - bear_pos)


def _cluster_levels(
    df: pd.DataFrame, swings: list[Swing], cluster_pct: float, close_price: float
) -> tuple[list[Level], list[Level]]:
    """Group swing prices into levels and count how often price touched them."""
    if not swings:
        return [], []

    prices = sorted(s.price for s in swings)
    clusters: list[list[float]] = [[prices[0]]]
    for price in prices[1:]:
        centre = float(np.mean(clusters[-1]))
        if abs(price - centre) / max(centre, 1e-12) * 100.0 <= cluster_pct:
            clusters[-1].append(price)
        else:
            clusters.append([price])

    highs = df["high"].to_numpy(dtype=float)
    lows = df["low"].to_numpy(dtype=float)
    n = len(df)

    supports: list[Level] = []
    resistances: list[Level] = []
    for cluster in clusters:
        centre = float(np.mean(cluster))
        tolerance = centre * cluster_pct / 100.0
        touched = (np.abs(highs - centre) <= tolerance) | (np.abs(lows - centre) <= tolerance)
        touch_positions = np.flatnonzero(touched)
        if touch_positions.size == 0:
            continue
        level = Level(
            price=centre,
            kind="resistance" if centre > close_price else "support",
            touches=int(touch_positions.size),
            swings=len(cluster),
            last_touch_bars_ago=int(n - 1 - touch_positions[-1]),
        )
        (resistances if level.kind == "resistance" else supports).append(level)

    supports.sort(key=lambda lv: lv.price, reverse=True)  # nearest first
    resistances.sort(key=lambda lv: lv.price)
    return supports, resistances


def _pick_nearest(levels: list[Level], min_touches: int) -> Level | None:
    """Nearest level with enough touches, falling back to the nearest of any.

    The fallback matters: refusing to see a single-touch level right above price
    would let the "room to resistance" gate wave through a trade into an obvious
    ceiling.
    """
    for level in levels:
        if level.touches >= min_touches:
            return level
    return levels[0] if levels else None


def _build_zones(
    swings: list[Swing], atr: float, multiplier: float, total_bars: int, limit: int = 3
) -> list[Zone]:
    """Turn recent swing extremes into ATR-thick supply/demand zones."""
    if not np.isfinite(atr) or atr <= 0:
        return []
    thickness = atr * multiplier
    zones: list[Zone] = []
    for kind, swing_kind in (("supply", "high"), ("demand", "low")):
        selected = [s for s in swings if s.kind == swing_kind][-limit:]
        for swing in selected:
            low = swing.price - thickness if kind == "supply" else swing.price
            high = swing.price if kind == "supply" else swing.price + thickness
            zones.append(
                Zone(low=low, high=high, kind=kind, bars_ago=total_bars - 1 - swing.position)
            )
    return zones


def _detect_sweep(
    df: pd.DataFrame, swings: list[Swing], wick_ratio: float, window: int = 2
) -> int:
    """Detect a stop-run through a swing extreme that closed back inside.

    Returns:
        ``+1`` when sell-side liquidity (lows) was swept — bullish; ``-1`` when
        buy-side liquidity (highs) was swept — bearish; ``0`` otherwise.
    """
    swing_high = last_swing(swings, "high")
    swing_low = last_swing(swings, "low")
    tail = df.iloc[-window:]

    for position in range(len(tail) - 1, -1, -1):
        bar = tail.iloc[position]
        high, low, close_, open_ = (
            float(bar["high"]), float(bar["low"]), float(bar["close"]), float(bar["open"])
        )
        span = high - low
        if span <= 0:
            continue
        upper_wick = high - max(open_, close_)
        lower_wick = min(open_, close_) - low

        if (
            swing_low is not None
            and low < swing_low.price <= close_
            and safe_div(lower_wick, span) >= wick_ratio
        ):
            return 1
        if (
            swing_high is not None
            and high > swing_high.price >= close_
            and safe_div(upper_wick, span) >= wick_ratio
        ):
            return -1
    return 0


def _detect_false_breakout(df: pd.DataFrame, bars: int) -> int:
    """Detect a range breakout that was given back within ``bars`` bars.

    Returns:
        ``+1`` for a failed *downside* break (bullish), ``-1`` for a failed
        *upside* break (bearish), ``0`` if none.
    """
    if "dc_upper" not in df.columns or "dc_lower" not in df.columns or len(df) < bars + 2:
        return 0
    # The channel of the previous bar is the boundary the current bar breaks.
    upper = df["dc_upper"].shift(1)
    lower = df["dc_lower"].shift(1)
    close = df["close"]
    current = float(close.iloc[-1])

    for offset in range(1, bars + 1):
        position = -1 - offset
        boundary_up = float(upper.iloc[position])
        boundary_down = float(lower.iloc[position])
        broke_up = float(close.iloc[position]) > boundary_up
        broke_down = float(close.iloc[position]) < boundary_down
        if broke_up and current < boundary_up:
            return -1
        if broke_down and current > boundary_down:
            return 1
    return 0


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def analyse_structure(df: pd.DataFrame, cfg: StructureConfig) -> StructureReport:
    """Produce a full :class:`StructureReport` for the last bar of ``df``.

    Args:
        df: frame already enriched by :class:`indicators.engine.IndicatorEngine`
            (``atr``, ``dc_upper``, ``dc_lower`` are used when present).
        cfg: the ``structure`` configuration section.

    Returns:
        A :class:`StructureReport`. On too little data it returns the neutral
        default rather than raising, so a fresh listing simply produces no
        signal instead of crashing the scan loop.
    """
    minimum = 2 * cfg.swing_strength + 2
    if len(df) < minimum:
        return StructureReport()

    window = df.iloc[-cfg.lookback :] if len(df) > cfg.lookback else df
    swings = find_swings(window, cfg.swing_strength, confirmed_only=True)
    if not swings:
        return StructureReport()

    labels = _label_swings(swings)
    trend = _trend_from_labels(labels)
    break_event = _detect_break(window, swings, trend)

    close_price = float(window["close"].iloc[-1])
    atr_value = float(window["atr"].iloc[-1]) if "atr" in window.columns else float("nan")

    supports, resistances = _cluster_levels(window, swings, cfg.sr_cluster_pct, close_price)
    nearest_support = _pick_nearest(supports, cfg.sr_min_touches)
    nearest_resistance = _pick_nearest(resistances, cfg.sr_min_touches)

    # Unknown ATR must not silently satisfy the room-to-level gate, so distance
    # stays at +inf only when there genuinely is no level on that side.
    room_up = float("inf")
    room_down = float("inf")
    if np.isfinite(atr_value) and atr_value > 0:
        if nearest_resistance is not None:
            room_up = max(0.0, (nearest_resistance.price - close_price) / atr_value)
        if nearest_support is not None:
            room_down = max(0.0, (close_price - nearest_support.price) / atr_value)

    zones = _build_zones(swings, atr_value, cfg.zone_atr_multiplier, len(window))
    swing_high = last_swing(swings, "high")
    swing_low = last_swing(swings, "low")

    return StructureReport(
        trend=trend,
        labels=labels,
        swings=swings,
        last_swing_high=swing_high.price if swing_high else None,
        last_swing_low=swing_low.price if swing_low else None,
        break_event=break_event,
        supports=supports,
        resistances=resistances,
        zones=zones,
        nearest_support=nearest_support,
        nearest_resistance=nearest_resistance,
        room_down_atr=room_down,
        room_up_atr=room_up,
        in_demand_zone=any(z.kind == "demand" and z.contains(close_price) for z in zones),
        in_supply_zone=any(z.kind == "supply" and z.contains(close_price) for z in zones),
        sweep_direction=_detect_sweep(window, swings, cfg.sweep_wick_ratio),
        false_breakout_direction=_detect_false_breakout(window, cfg.false_breakout_bars),
    )
