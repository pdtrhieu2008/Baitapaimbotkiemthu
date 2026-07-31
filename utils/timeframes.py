"""Timeframe arithmetic shared by the data layer, the MTF analyser and metrics."""

from __future__ import annotations

import re
from datetime import UTC, datetime, timedelta

__all__ = [
    "TIMEFRAME_ORDER",
    "bars_per_year",
    "floor_to_timeframe",
    "is_bar_closed",
    "sort_timeframes",
    "timeframe_to_ms",
    "timeframe_to_seconds",
    "timeframe_to_timedelta",
]

_TF_PATTERN = re.compile(r"^(\d+)([smhdwM])$")

#: Multiplier from a timeframe unit to seconds. ``M`` (month) is approximated
#: as 30 days, which is only used for annualisation, never for bar alignment.
_UNIT_SECONDS: dict[str, int] = {
    "s": 1,
    "m": 60,
    "h": 3_600,
    "d": 86_400,
    "w": 604_800,
    "M": 2_592_000,
}

#: Canonical ordering, coarsest last. Used to sort MTF timeframes.
TIMEFRAME_ORDER: tuple[str, ...] = (
    "1s", "5s", "15s", "30s",
    "1m", "3m", "5m", "15m", "30m",
    "1h", "2h", "4h", "6h", "8h", "12h",
    "1d", "3d", "1w", "1M",
)


def timeframe_to_seconds(timeframe: str) -> int:
    """Convert a ccxt-style timeframe such as ``"15m"`` to seconds.

    Args:
        timeframe: e.g. ``"1m"``, ``"4h"``, ``"1d"``, ``"1w"``, ``"1M"``.

    Returns:
        Length of one bar in seconds.

    Raises:
        ValueError: on an unparsable timeframe.
    """
    match = _TF_PATTERN.match(timeframe.strip())
    if not match:
        raise ValueError(
            f"invalid timeframe {timeframe!r}; expected <int><unit> with unit in s/m/h/d/w/M"
        )
    amount, unit = int(match.group(1)), match.group(2)
    if amount <= 0:
        raise ValueError(f"timeframe {timeframe!r} must have a positive amount")
    return amount * _UNIT_SECONDS[unit]


def timeframe_to_ms(timeframe: str) -> int:
    """Length of one bar in milliseconds (ccxt's native unit)."""
    return timeframe_to_seconds(timeframe) * 1_000


def timeframe_to_timedelta(timeframe: str) -> timedelta:
    """Length of one bar as a :class:`datetime.timedelta`."""
    return timedelta(seconds=timeframe_to_seconds(timeframe))


def sort_timeframes(timeframes: list[str], *, descending: bool = True) -> list[str]:
    """Sort timeframes by real duration.

    Args:
        timeframes: timeframe strings.
        descending: coarsest first when ``True`` (the natural top-down reading
            order for multi-timeframe analysis).

    Returns:
        A new sorted list.
    """
    return sorted(timeframes, key=timeframe_to_seconds, reverse=descending)


def floor_to_timeframe(moment: datetime, timeframe: str) -> datetime:
    """Round ``moment`` down to the start of its bar.

    Uses the Unix epoch as the alignment anchor, which is exactly how exchanges
    bucket candles for sub-daily timeframes.

    Args:
        moment: timezone-aware instant (naive input is assumed UTC).
        timeframe: bar size.

    Returns:
        The bar's open time, in UTC.
    """
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=UTC)
    moment = moment.astimezone(UTC)
    step = timeframe_to_seconds(timeframe)
    epoch_seconds = int(moment.timestamp())
    return datetime.fromtimestamp(epoch_seconds - (epoch_seconds % step), tz=UTC)


def is_bar_closed(bar_open: datetime, timeframe: str, now: datetime | None = None) -> bool:
    """Whether the bar starting at ``bar_open`` has finished forming.

    The whole system only ever acts on closed bars; this is the single check
    that guarantees it.

    Args:
        bar_open: the bar's open timestamp.
        timeframe: bar size.
        now: current time, defaults to ``datetime.now(UTC)``.

    Returns:
        ``True`` when ``bar_open + timeframe <= now``.
    """
    if bar_open.tzinfo is None:
        bar_open = bar_open.replace(tzinfo=UTC)
    reference = now or datetime.now(UTC)
    if reference.tzinfo is None:
        reference = reference.replace(tzinfo=UTC)
    return bar_open + timeframe_to_timedelta(timeframe) <= reference


def bars_per_year(timeframe: str, *, market_hours_per_day: float = 24.0,
                  trading_days_per_year: float = 365.0) -> float:
    """Number of bars in a year, for annualising Sharpe/Sortino.

    Defaults describe a 24/7 crypto market. For equities pass
    ``market_hours_per_day=6.5, trading_days_per_year=252``; for FX
    ``market_hours_per_day=24, trading_days_per_year=252``.

    Args:
        timeframe: bar size.
        market_hours_per_day: hours the market is open each session.
        trading_days_per_year: number of sessions per year.

    Returns:
        Bars per year as a float.
    """
    seconds = timeframe_to_seconds(timeframe)
    seconds_per_year = trading_days_per_year * market_hours_per_day * 3_600.0
    return seconds_per_year / seconds
