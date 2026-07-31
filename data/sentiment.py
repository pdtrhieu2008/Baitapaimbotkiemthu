"""Market-wide sentiment feeds: Fear & Greed, BTC dominance, news.

All three are **optional** and all three are cached with a TTL, because they are
market-wide numbers that change far more slowly than price and because two of the
three are free endpoints that will rate-limit an impolite client.

Sources:

* Fear & Greed — ``alternative.me``, free, no key, one number 0-100.
* BTC dominance — ``coingecko.com/api/v3/global``, free, no key.
* News — optional. Two providers are supported and **neither is enabled by
  default** because both need a key:
  ``cryptopanic`` (has a native bullish/bearish vote, which is what makes it
  usable without an NLP model) and ``newsapi`` (headlines only, scored with the
  keyword heuristic below).

On the news score, stated plainly: the keyword scorer is a crude lexicon, not
sentiment analysis. It is good enough to notice "SEC lawsuit" versus "ETF
approved" in a headline and nothing more. It is weighted as one vote inside one
of eight components, and the honest alternative — leaving news off — is the
default.
"""

from __future__ import annotations

import time
from typing import Any

from config.settings import DataConfig
from data.models import SentimentSnapshot
from utils.helpers import async_retry
from utils.logger import get_logger

__all__ = ["SentimentClient"]

_log = get_logger("data.sentiment")

_FEAR_GREED_URL = "https://api.alternative.me/fng/?limit=1"
_GLOBAL_URL = "https://api.coingecko.com/api/v3/global"
_CRYPTOPANIC_URL = "https://cryptopanic.com/api/v1/posts/"
_NEWSAPI_URL = "https://newsapi.org/v2/everything"

#: Deliberately small, hand-checked lexicon. Weight, not word count, is what
#: keeps this from doing damage.
_BULLISH_TERMS: tuple[str, ...] = (
    "surge", "rally", "soar", "breakout", "adoption", "approved", "approval",
    "inflow", "accumulate", "bullish", "upgrade", "partnership", "record high",
    "institutional", "halving", "buyback",
)
_BEARISH_TERMS: tuple[str, ...] = (
    "crash", "plunge", "dump", "hack", "exploit", "lawsuit", "sue", "ban",
    "outflow", "liquidation", "bearish", "downgrade", "fraud", "insolvency",
    "delist", "investigation", "sec charges",
)


class SentimentClient:
    """Fetch and cache market-wide sentiment.

    Args:
        cfg: the ``data`` configuration section.
    """

    def __init__(self, cfg: DataConfig) -> None:
        self.cfg = cfg
        self._session: Any | None = None
        self._cached: SentimentSnapshot | None = None
        self._cached_at: float = 0.0
        self._failed: dict[str, str] = {}

    # -- session ------------------------------------------------------------
    async def _get_session(self) -> Any | None:
        """Create the aiohttp session on first use."""
        if self._session is not None:
            return self._session
        try:
            import aiohttp
        except ImportError:
            self._failed["http"] = "aiohttp not installed"
            _log.warning(
                "aiohttp is not installed; sentiment feeds are disabled. "
                "Install it with: pip install aiohttp"
            )
            return None
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=15),
            headers={"User-Agent": "quantbot/0.1 (+https://github.com)"},
        )
        return self._session

    async def close(self) -> None:
        """Close the HTTP session."""
        if self._session is not None:
            await self._session.close()
            self._session = None

    async def _get_json(self, url: str, params: dict[str, Any] | None = None) -> Any | None:
        """GET and decode JSON, returning ``None`` on any failure."""
        session = await self._get_session()
        if session is None:
            return None

        async def _request() -> Any:
            async with session.get(url, params=params) as response:
                if response.status == 429:
                    raise RuntimeError("rate limited (429)")
                response.raise_for_status()
                return await response.json(content_type=None)

        try:
            return await async_retry(_request, attempts=2, backoff=2.0, description=f"GET {url}")
        except Exception as exc:  # noqa: BLE001 - any failure means "no data"
            _log.warning("sentiment fetch failed for %s: %s: %s", url, type(exc).__name__, exc)
            return None

    # -- individual feeds ---------------------------------------------------
    async def fetch_fear_greed(self) -> float:
        """Fear & Greed index, or ``nan`` when unavailable."""
        payload = await self._get_json(_FEAR_GREED_URL)
        try:
            return float(payload["data"][0]["value"])  # type: ignore[index]
        except (TypeError, KeyError, IndexError, ValueError):
            self._failed["fear_greed"] = "alternative.me unavailable or changed shape"
            return float("nan")

    async def fetch_dominance(self) -> tuple[float, float]:
        """BTC dominance and its 24h change, both ``nan`` when unavailable.

        CoinGecko's ``/global`` endpoint does not publish a dominance *change*
        directly, so it is derived from the change in total market cap versus
        BTC's own — an approximation, and reported as such.
        """
        payload = await self._get_json(_GLOBAL_URL)
        try:
            data = payload["data"]  # type: ignore[index]
            dominance = float(data["market_cap_percentage"]["btc"])
            total_change = float(data.get("market_cap_change_percentage_24h_usd", 0.0))
        except (TypeError, KeyError, ValueError):
            self._failed["btc_dominance"] = "coingecko /global unavailable or changed shape"
            return float("nan"), float("nan")

        # Rough proxy: when the whole market falls, capital rotates into BTC and
        # dominance rises. Sign only; the magnitude is not meaningful.
        change = -total_change * 0.1
        return dominance, change

    async def fetch_news(self) -> tuple[float, tuple[str, ...]]:
        """News polarity in ``[-1, 1]`` plus the headlines behind it."""
        if not self.cfg.news_sentiment:
            return float("nan"), ()
        if not self.cfg.news_api_key:
            self._failed["news"] = "NEWS_API_KEY is empty"
            return float("nan"), ()

        if self.cfg.news_provider == "cryptopanic":
            return await self._fetch_cryptopanic()
        if self.cfg.news_provider == "newsapi":
            return await self._fetch_newsapi()
        self._failed["news"] = f"unknown provider {self.cfg.news_provider!r}"
        return float("nan"), ()

    async def _fetch_cryptopanic(self) -> tuple[float, tuple[str, ...]]:
        """Use CryptoPanic's own bullish/bearish votes — no NLP guessing."""
        payload = await self._get_json(
            _CRYPTOPANIC_URL,
            {"auth_token": self.cfg.news_api_key, "kind": "news", "filter": "important"},
        )
        results = (payload or {}).get("results") or []
        if not results:
            self._failed["news"] = "cryptopanic returned no results"
            return float("nan"), ()

        score = 0.0
        counted = 0
        headlines: list[str] = []
        for item in results[:20]:
            votes = item.get("votes") or {}
            positive = float(votes.get("positive", 0)) + float(votes.get("important", 0))
            negative = float(votes.get("negative", 0)) + float(votes.get("toxic", 0))
            if positive + negative > 0:
                score += (positive - negative) / (positive + negative)
                counted += 1
            if len(headlines) < 3 and item.get("title"):
                headlines.append(str(item["title"]))

        if counted == 0:
            return float("nan"), tuple(headlines)
        return max(-1.0, min(1.0, score / counted)), tuple(headlines)

    async def _fetch_newsapi(self) -> tuple[float, tuple[str, ...]]:
        """Headlines only, scored with the keyword lexicon."""
        payload = await self._get_json(
            _NEWSAPI_URL,
            {
                "q": "bitcoin OR crypto OR ethereum",
                "language": "en",
                "sortBy": "publishedAt",
                "pageSize": 30,
                "apiKey": self.cfg.news_api_key,
            },
        )
        articles = (payload or {}).get("articles") or []
        if not articles:
            self._failed["news"] = "newsapi returned no articles"
            return float("nan"), ()

        titles = [str(a.get("title") or "") for a in articles]
        score = self.score_headlines(titles)
        return score, tuple(titles[:3])

    @staticmethod
    def score_headlines(headlines: list[str]) -> float:
        """Keyword polarity of a list of headlines, in ``[-1, 1]``.

        Crude by construction — see the module docstring. Returns ``nan`` when
        no keyword matched at all, so "no signal" is distinguishable from
        "balanced signal".
        """
        bullish = bearish = 0
        for headline in headlines:
            lowered = headline.lower()
            bullish += sum(1 for term in _BULLISH_TERMS if term in lowered)
            bearish += sum(1 for term in _BEARISH_TERMS if term in lowered)
        total = bullish + bearish
        if total == 0:
            return float("nan")
        return (bullish - bearish) / total

    # -- aggregate ----------------------------------------------------------
    async def fetch(self, *, force: bool = False) -> SentimentSnapshot:
        """Fetch every enabled feed, honouring the cache TTL.

        Args:
            force: bypass the cache.

        Returns:
            A :class:`~data.models.SentimentSnapshot`. Fields whose feed is
            disabled or failed are ``nan``, never a placeholder value.
        """
        age = time.monotonic() - self._cached_at
        if not force and self._cached is not None and age < self.cfg.context_ttl_seconds:
            return self._cached

        self._failed.clear()
        fear_greed = await self.fetch_fear_greed() if self.cfg.fear_greed else float("nan")
        dominance, dominance_change = (
            await self.fetch_dominance() if self.cfg.btc_dominance else (float("nan"), float("nan"))
        )
        news_score, headlines = await self.fetch_news()

        snapshot = SentimentSnapshot(
            fear_greed=fear_greed,
            btc_dominance=dominance,
            btc_dominance_change_pct=dominance_change,
            news_score=news_score,
            news_headlines=headlines,
        )
        self._cached = snapshot
        self._cached_at = time.monotonic()
        return snapshot

    @property
    def failures(self) -> dict[str, str]:
        """Feeds that did not produce data on the last fetch, and why."""
        return dict(self._failed)
