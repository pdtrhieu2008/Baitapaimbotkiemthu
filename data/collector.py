"""The data collector: raw feeds in, analysed :class:`MarketSnapshot` out.

This is the seam between I/O and analysis. Everything above it (strategy, risk)
is pure and synchronous; everything below it (ccxt, HTTP) is async and failure
prone. Keeping the boundary here is what lets the backtester construct snapshots
from cached frames and drive the identical strategy code.

Failure policy: a missing *optional* feed degrades the snapshot and records a
reason in :attr:`~data.models.ExternalContext.unavailable`. A missing *required*
feed (the primary timeframe's candles) returns ``None`` — the symbol is skipped
this cycle rather than analysed on stale or partial data.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from analysis.context import ContextBuilder, MarketSnapshot, TimeframeContext
from config.settings import Settings
from data.base import MarketDataProvider
from data.models import ExternalContext, SentimentSnapshot
from data.repository import OHLCVRepository
from data.sentiment import SentimentClient
from utils.logger import get_logger
from utils.timeframes import is_bar_closed, sort_timeframes

__all__ = ["DataCollector"]

_log = get_logger("data.collector")


class DataCollector:
    """Assemble analysed snapshots for the live loop.

    Args:
        settings: validated configuration.
        provider: venue adapter, or ``None`` for an offline (cache-only) run.
    """

    def __init__(self, settings: Settings, provider: MarketDataProvider | None) -> None:
        self.settings = settings
        self.provider = provider
        self.repository = OHLCVRepository(settings.data, provider)
        self.sentiment = SentimentClient(settings.data)
        self.builder = ContextBuilder(settings.indicators, settings.structure)

        #: Timeframes the strategy actually consults.
        self.timeframes = sort_timeframes(
            list({settings.data.primary_timeframe, *settings.strategy.mtf.all_timeframes}),
            descending=False,
        )
        self._external_cache: dict[str, tuple[float, ExternalContext]] = {}
        self._last_bar: dict[str, pd.Timestamp] = {}

    async def close(self) -> None:
        """Release the sentiment HTTP session."""
        await self.sentiment.close()

    # -- external context ---------------------------------------------------
    async def collect_external(self, symbol: str) -> ExternalContext:
        """Ticker, funding, open interest and market-wide sentiment.

        Cached for ``data.context_ttl_seconds``: these change far more slowly
        than price and every one of them is a separate network round trip.
        """
        cached = self._external_cache.get(symbol)
        if cached is not None:
            fetched_at, context = cached
            if time.monotonic() - fetched_at < self.settings.data.context_ttl_seconds:
                return context

        cfg = self.settings.data
        unavailable: dict[str, str] = {}
        ticker = funding = open_interest = None
        sentiment: SentimentSnapshot | None = None

        if self.provider is None:
            unavailable["exchange"] = "offline mode: no provider configured"
        else:
            ticker = await self.provider.fetch_ticker(symbol)
            if ticker is None:
                unavailable["ticker"] = "venue returned no ticker"

            if cfg.funding_rate:
                funding = await self.provider.fetch_funding_rate(symbol)
                if funding is None:
                    unavailable["funding_rate"] = "not provided for this market"

            if cfg.open_interest:
                open_interest = await self.provider.fetch_open_interest(symbol)
                if open_interest is None:
                    unavailable["open_interest"] = "not provided for this market"

        if cfg.liquidations:
            # Config validation cannot catch this because it depends on the
            # venue; say it plainly here instead of pretending the feed exists.
            unavailable["liquidations"] = (
                "no REST endpoint: requires the !forceOrder websocket stream "
                "(see CCXTProvider.fetch_liquidations)"
            )

        if cfg.fear_greed or cfg.btc_dominance or cfg.news_sentiment:
            sentiment = await self.sentiment.fetch()
            unavailable.update(self.sentiment.failures)

        context = ExternalContext(
            symbol=symbol,
            ticker=ticker,
            funding=funding,
            open_interest=open_interest,
            sentiment=sentiment,
            unavailable=unavailable,
        )
        self._external_cache[symbol] = (time.monotonic(), context)
        return context

    # -- snapshots ----------------------------------------------------------
    async def collect(self, symbol: str, *, refresh: bool = True) -> MarketSnapshot | None:
        """Build a complete snapshot for one symbol.

        Args:
            symbol: unified symbol.
            refresh: fetch fresh candles; ``False`` uses the cache only.

        Returns:
            A :class:`~analysis.context.MarketSnapshot`, or ``None`` when the
            primary timeframe has insufficient data (the symbol is then skipped
            for this cycle).
        """
        primary_tf = self.settings.data.primary_timeframe
        frames = await self.repository.get_many(symbol, self.timeframes, refresh=refresh)

        primary = frames.get(primary_tf)
        needed = self.builder.engine.required_history
        if primary is None or len(primary) < needed:
            _log.warning(
                "%s: %d bars of %s, need %d to seed the indicators; skipping this cycle",
                symbol, 0 if primary is None else len(primary), primary_tf, needed,
            )
            return None

        if self.settings.app.only_closed_bars and not is_bar_closed(
            primary.index[-1].to_pydatetime(), primary_tf
        ):
            # The repository already drops the forming bar; reaching here means
            # the venue's clock disagrees with ours, which is worth knowing.
            _log.error(
                "%s: last %s bar at %s is not closed yet - the venue's candle boundaries "
                "disagree with system time. Check the host clock (NTP).",
                symbol, primary_tf, primary.index[-1],
            )
            return None

        contexts: dict[str, TimeframeContext] = {}
        for timeframe, frame in frames.items():
            if frame.empty:
                _log.debug("%s %s: no data", symbol, timeframe)
                continue
            if len(frame) < needed:
                _log.info(
                    "%s %s: only %d bars (need %d); this timeframe will read as missing data "
                    "in the MTF gate",
                    symbol, timeframe, len(frame), needed,
                )
                continue
            try:
                enriched = self.builder.enrich(frame, timeframe)
                contexts[timeframe] = self.builder.build(symbol, timeframe, enriched)
            except (ValueError, TypeError, KeyError) as exc:
                _log.error("%s %s: analysis failed: %s: %s", symbol, timeframe, type(exc).__name__, exc)

        if primary_tf not in contexts:
            return None

        external = await self.collect_external(symbol)
        if not self._passes_liquidity(symbol, external):
            return None

        return MarketSnapshot(
            symbol=symbol,
            primary_timeframe=primary_tf,
            contexts=contexts,
            external=external,
        )

    def _passes_liquidity(self, symbol: str, external: ExternalContext) -> bool:
        """Reject instruments too thin for the configured position sizes."""
        minimum = self.settings.universe.min_quote_volume_24h
        if minimum <= 0 or external.ticker is None:
            return True
        volume = external.ticker.quote_volume_24h
        if not np.isfinite(volume):
            return True  # unknown, not zero - the guard logs the spread case
        if volume < minimum:
            _log.info(
                "%s: 24h quote volume %.0f is below the %.0f minimum; skipping (thin books "
                "make fills diverge from the backtest)",
                symbol, volume, minimum,
            )
            return False
        return True

    def is_new_bar(self, symbol: str, snapshot: MarketSnapshot) -> bool:
        """Whether this snapshot's primary bar has not been evaluated yet.

        The live loop polls far more often than a bar closes; without this check
        the same closed bar would be scored dozens of times and, worse, could
        emit the same signal repeatedly.
        """
        current = pd.Timestamp(snapshot.primary.bar_time)
        previous = self._last_bar.get(symbol)
        if previous is not None and current <= previous:
            return False
        self._last_bar[symbol] = current
        return True

    async def warm_up(self) -> dict[str, int]:
        """Pre-load every configured symbol and timeframe.

        Returns:
            Mapping of ``"symbol timeframe"`` to the number of bars available,
            for a start-up log line that makes missing history obvious before the
            first cycle rather than after it.
        """
        report: dict[str, int] = {}
        for symbol in self.settings.universe.symbols:
            frames = await self.repository.get_many(symbol, self.timeframes, refresh=True)
            for timeframe, frame in frames.items():
                report[f"{symbol} {timeframe}"] = len(frame)
        return report
