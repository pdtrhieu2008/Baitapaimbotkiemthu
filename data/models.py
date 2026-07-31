"""Plain data carriers for everything the data layer fetches.

Keeping these in their own module (with no exchange dependency) lets the
analysis and strategy layers depend on the *shape* of market context without
importing ccxt or aiohttp — which is also what makes them trivial to construct
in unit tests.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

from utils.helpers import utc_now

__all__ = [
    "ExternalContext",
    "FundingSnapshot",
    "OpenInterestSnapshot",
    "SentimentSnapshot",
    "Ticker",
]


@dataclass(frozen=True, slots=True)
class Ticker:
    """Last traded price plus the top of book.

    Attributes:
        symbol: unified symbol.
        last: last traded price.
        bid: best bid, ``nan`` if the venue does not report it.
        ask: best ask.
        quote_volume_24h: rolling 24h turnover in the quote currency.
        timestamp: when the venue produced the snapshot.
    """

    symbol: str
    last: float
    bid: float = float("nan")
    ask: float = float("nan")
    quote_volume_24h: float = float("nan")
    timestamp: datetime = field(default_factory=utc_now)

    @property
    def spread_pct(self) -> float:
        """Bid-ask spread as a percentage of the mid price.

        Returns ``nan`` when the book is unavailable — callers must treat that
        as "unknown", never as "zero", otherwise the spread guard silently
        stops guarding.
        """
        if not (np.isfinite(self.bid) and np.isfinite(self.ask)) or self.bid <= 0:
            return float("nan")
        mid = (self.bid + self.ask) / 2.0
        return 100.0 * (self.ask - self.bid) / mid if mid > 0 else float("nan")

    @property
    def mid(self) -> float:
        if np.isfinite(self.bid) and np.isfinite(self.ask):
            return (self.bid + self.ask) / 2.0
        return self.last


@dataclass(frozen=True, slots=True)
class FundingSnapshot:
    """Perpetual funding rate, expressed in percent per funding interval."""

    symbol: str
    rate_pct: float
    next_funding_time: datetime | None = None
    timestamp: datetime = field(default_factory=utc_now)

    @property
    def is_extreme(self) -> bool:
        """Crowded positioning: |funding| above 0.1% per 8h is historically high."""
        return np.isfinite(self.rate_pct) and abs(self.rate_pct) > 0.1

    @property
    def bias(self) -> int:
        """Contrarian read: heavily positive funding is a crowded long book.

        Funding is a *positioning* signal, not a direction signal. Extremely
        positive funding means longs are paying to stay in, which historically
        precedes long squeezes — hence the inverted sign.
        """
        if not np.isfinite(self.rate_pct):
            return 0
        if self.rate_pct > 0.05:
            return -1
        if self.rate_pct < -0.05:
            return 1
        return 0


@dataclass(frozen=True, slots=True)
class OpenInterestSnapshot:
    """Open interest now and its change over the recent window."""

    symbol: str
    value: float
    change_pct: float = float("nan")
    timestamp: datetime = field(default_factory=utc_now)

    def bias(self, price_change_pct: float) -> int:
        """Classic OI/price cross-read.

        * price up + OI up   -> new money joining the move: confirming (+1)
        * price up + OI down -> short covering, weak: fading (-1)
        * price down + OI up -> new shorts: confirming the downside (-1)
        * price down + OI down -> long liquidation exhausting: fading (+1)
        """
        if not (np.isfinite(self.change_pct) and np.isfinite(price_change_pct)):
            return 0
        if abs(self.change_pct) < 0.5 or abs(price_change_pct) < 0.1:
            return 0
        rising_oi = self.change_pct > 0
        rising_price = price_change_pct > 0
        if rising_price:
            return 1 if rising_oi else -1
        return -1 if rising_oi else 1


@dataclass(frozen=True, slots=True)
class SentimentSnapshot:
    """Market-wide sentiment, none of it instrument-specific.

    Attributes:
        fear_greed: alternative.me index, ``0`` (extreme fear) to ``100``
            (extreme greed). ``nan`` when unavailable.
        btc_dominance: BTC share of total market cap, in percent.
        btc_dominance_change_pct: 24h change of that share.
        news_score: aggregated news polarity in ``[-1, 1]``, ``nan`` when no
            news provider is configured.
        news_headlines: a few headlines behind the score, for the notification.
    """

    fear_greed: float = float("nan")
    btc_dominance: float = float("nan")
    btc_dominance_change_pct: float = float("nan")
    news_score: float = float("nan")
    news_headlines: tuple[str, ...] = ()
    timestamp: datetime = field(default_factory=utc_now)

    @property
    def fear_greed_bias(self) -> int:
        """Contrarian at the extremes, neutral in the middle.

        Extreme fear (<25) has historically been a better place to buy than
        extreme greed (>75), so the sign is inverted. Between the extremes the
        index carries no usable information and returns 0 rather than a coin
        flip.
        """
        if not np.isfinite(self.fear_greed):
            return 0
        if self.fear_greed <= 25:
            return 1
        if self.fear_greed >= 75:
            return -1
        return 0

    @property
    def news_bias(self) -> int:
        if not np.isfinite(self.news_score):
            return 0
        if self.news_score > 0.15:
            return 1
        if self.news_score < -0.15:
            return -1
        return 0

    def to_dict(self) -> dict[str, object]:
        return {
            "fear_greed": self.fear_greed,
            "btc_dominance": self.btc_dominance,
            "news_score": self.news_score,
        }


@dataclass(frozen=True, slots=True)
class ExternalContext:
    """Non-OHLCV context for one symbol at one moment.

    Every field is optional. The scorer checks availability and **redistributes
    the sentiment weight** when nothing is available, rather than scoring a
    missing input as zero — otherwise running without these feeds would quietly
    lower every score by up to 10 points and change the effective thresholds.
    """

    symbol: str
    ticker: Ticker | None = None
    funding: FundingSnapshot | None = None
    open_interest: OpenInterestSnapshot | None = None
    sentiment: SentimentSnapshot | None = None
    #: Populated with a human-readable reason for each feed that was skipped.
    unavailable: dict[str, str] = field(default_factory=dict)

    @property
    def has_any_sentiment(self) -> bool:
        """Whether at least one sentiment-ish input carries information."""
        if self.funding is not None and np.isfinite(self.funding.rate_pct):
            return True
        if self.open_interest is not None and np.isfinite(self.open_interest.change_pct):
            return True
        if self.sentiment is None:
            return False
        return any(
            np.isfinite(value)
            for value in (
                self.sentiment.fear_greed,
                self.sentiment.btc_dominance_change_pct,
                self.sentiment.news_score,
            )
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "symbol": self.symbol,
            "spread_pct": self.ticker.spread_pct if self.ticker else None,
            "funding_pct": self.funding.rate_pct if self.funding else None,
            "oi_change_pct": self.open_interest.change_pct if self.open_interest else None,
            "sentiment": self.sentiment.to_dict() if self.sentiment else None,
            "unavailable": self.unavailable,
        }
