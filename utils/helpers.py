"""Small, dependency-light utilities used across the project."""

from __future__ import annotations

import asyncio
import math
import random
from collections.abc import Awaitable, Callable, Iterable, Sequence
from datetime import UTC, datetime
from typing import Any, TypeVar

from utils.logger import get_logger

__all__ = [
    "async_retry",
    "chunked",
    "clamp",
    "ensure_utc",
    "format_price",
    "pct_change",
    "safe_div",
    "sign",
    "utc_now",
]

_T = TypeVar("_T")
_log = get_logger("utils.helpers")


def utc_now() -> datetime:
    """Current time as a timezone-aware UTC datetime."""
    return datetime.now(UTC)


def ensure_utc(moment: datetime) -> datetime:
    """Return ``moment`` as timezone-aware UTC (naive input is assumed UTC)."""
    if moment.tzinfo is None:
        return moment.replace(tzinfo=UTC)
    return moment.astimezone(UTC)


def safe_div(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Divide without ever raising.

    Guards the countless ``x / atr`` and ``wins / losses`` expressions in this
    codebase, where a zero or NaN denominator is a normal market condition
    (a doji has zero range, a run with no losses has zero gross loss) rather
    than a bug.

    Args:
        numerator: dividend.
        denominator: divisor.
        default: returned when the divisor is 0 or either side is not finite.
    """
    try:
        if denominator == 0 or not math.isfinite(denominator) or not math.isfinite(numerator):
            return default
        return numerator / denominator
    except (TypeError, ValueError):
        return default


def clamp(value: float, low: float, high: float) -> float:
    """Constrain ``value`` to ``[low, high]``."""
    if low > high:
        raise ValueError(f"clamp bounds inverted: low={low} > high={high}")
    return max(low, min(high, value))


def sign(value: float) -> int:
    """Return ``-1``, ``0`` or ``1``."""
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0


def pct_change(new: float, old: float) -> float:
    """Percentage change from ``old`` to ``new`` (``0.0`` if ``old`` is 0)."""
    return safe_div(new - old, abs(old)) * 100.0


def format_price(value: float, *, digits: int | None = None) -> str:
    """Format a price with a sensible number of decimals for its magnitude.

    BTC at 67000 wants 2 decimals; SHIB at 0.000023 wants 8. Hard-coding a
    single precision would make notifications unreadable for one of them.

    Args:
        value: the price.
        digits: force a specific precision instead of auto-selecting.
    """
    if not math.isfinite(value):
        return "n/a"
    if digits is not None:
        return f"{value:,.{digits}f}"
    magnitude = abs(value)
    if magnitude >= 1_000:
        digits = 2
    elif magnitude >= 1:
        digits = 4
    elif magnitude >= 0.01:
        digits = 5
    elif magnitude >= 0.0001:
        digits = 6
    else:
        digits = 8
    return f"{value:,.{digits}f}"


def chunked(items: Sequence[_T], size: int) -> Iterable[Sequence[_T]]:
    """Yield consecutive slices of at most ``size`` elements."""
    if size < 1:
        raise ValueError("chunk size must be >= 1")
    for start in range(0, len(items), size):
        yield items[start : start + size]


async def async_retry(
    func: Callable[[], Awaitable[_T]],
    *,
    attempts: int = 4,
    backoff: float = 2.0,
    exceptions: tuple[type[BaseException], ...] = (Exception,),
    description: str = "operation",
    jitter: float = 0.25,
) -> _T:
    """Await ``func`` with exponential backoff and jitter.

    Network calls to exchanges and to the Telegram API fail transiently all the
    time (rate limits, 5xx, DNS blips). Retrying with jitter avoids the
    thundering herd that a fixed delay creates when several symbols fail at
    once.

    Args:
        func: zero-argument coroutine factory. It is re-invoked on each try, so
            pass a lambda rather than an already-created coroutine.
        attempts: total number of tries (``1`` disables retrying).
        backoff: base delay in seconds; try *n* waits ``backoff * 2**(n-1)``.
        exceptions: exception types considered transient.
        description: label used in log messages.
        jitter: fraction of the delay added at random, in ``[0, 1]``.

    Returns:
        Whatever ``func`` returns.

    Raises:
        The last exception, once the attempts are exhausted.
    """
    if attempts < 1:
        raise ValueError("attempts must be >= 1")
    last: BaseException | None = None
    for attempt in range(1, attempts + 1):
        try:
            return await func()
        except asyncio.CancelledError:
            raise  # never swallow cancellation: it is a shutdown request
        except exceptions as exc:
            last = exc
            if attempt == attempts:
                break
            delay = backoff * (2 ** (attempt - 1))
            delay += delay * jitter * random.random()  # noqa: S311 - not cryptographic
            _log.warning(
                "%s failed (attempt %d/%d): %s: %s - retrying in %.1fs",
                description, attempt, attempts, type(exc).__name__, exc, delay,
            )
            await asyncio.sleep(delay)
    assert last is not None  # noqa: S101 - unreachable unless attempts < 1
    _log.error("%s failed after %d attempts: %s", description, attempts, last)
    raise last


def describe_dict(data: dict[str, Any], *, sep: str = ", ") -> str:
    """Render a small mapping compactly for a log line."""
    parts = []
    for key, value in data.items():
        parts.append(f"{key}={value:.4g}" if isinstance(value, float) else f"{key}={value}")
    return sep.join(parts)
