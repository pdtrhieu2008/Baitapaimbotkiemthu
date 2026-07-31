"""The market-data interface every venue adapter implements.

The strategy, risk and backtest layers only ever see this protocol, which is
what makes the "crypto first, but designed for forex and stocks" requirement
real rather than aspirational: adding MetaTrader, Interactive Brokers or a
vendor CSV means writing one class that satisfies :class:`MarketDataProvider`,
with no change anywhere else.

Methods that a venue genuinely cannot serve should return ``None`` (and record a
reason) rather than raise or fabricate a value. Spot markets have no funding
rate; equities have no open interest in this sense. The scorer already treats
missing inputs by redistributing their weight, so ``None`` degrades the analysis
honestly instead of poisoning it with zeros.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import pandas as pd

from data.models import FundingSnapshot, OpenInterestSnapshot, Ticker

__all__ = ["MarketDataProvider"]


@runtime_checkable
class MarketDataProvider(Protocol):
    """Minimum surface the bot needs from a venue."""

    #: Venue identifier, used in logs.
    name: str

    async def load_markets(self) -> None:
        """Fetch instrument metadata. Called once at start-up."""
        ...

    async def fetch_ohlcv(
        self, symbol: str, timeframe: str, limit: int = 500, since: int | None = None
    ) -> pd.DataFrame:
        """Return candles as a frame indexed by bar-open time (UTC, ascending).

        The frame must contain ``open``, ``high``, ``low``, ``close``, ``volume``
        and must **exclude the currently forming bar** — every consumer assumes
        the last row is closed.
        """
        ...

    async def fetch_ticker(self, symbol: str) -> Ticker | None:
        """Last price and, when the venue provides it, the top of book."""
        ...

    async def fetch_funding_rate(self, symbol: str) -> FundingSnapshot | None:
        """Perpetual funding rate, or ``None`` on markets that have none."""
        ...

    async def fetch_open_interest(self, symbol: str) -> OpenInterestSnapshot | None:
        """Open interest, or ``None`` when unsupported."""
        ...

    async def close(self) -> None:
        """Release sockets and sessions."""
        ...
