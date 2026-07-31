"""ccxt-backed market data provider (Binance by default).

Only public, read-only endpoints are used. API keys are optional and, when
supplied, buy nothing more than higher rate limits — this project never places
an order.

Two details that matter for correctness rather than style:

* **The forming bar is dropped.** ``fetch_ohlcv`` returns the in-progress candle
  as its last row; its high, low and close are still moving. Every consumer here
  assumes closed bars, so it is removed on arrival.
* **Timestamps are bar-*open* times in UTC**, which is what ccxt returns and what
  the whole project assumes.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pandas as pd

from config.settings import ExchangeConfig
from data.models import FundingSnapshot, OpenInterestSnapshot, Ticker
from utils.helpers import async_retry, utc_now
from utils.logger import get_logger
from utils.timeframes import timeframe_to_ms

__all__ = ["CCXTProvider", "ExchangeUnavailable"]

_log = get_logger("data.exchange")


class ExchangeUnavailable(RuntimeError):
    """Raised when ccxt is missing or the venue cannot be reached at all."""


def _import_ccxt() -> Any:
    """Import ``ccxt.async_support`` with an actionable error message."""
    try:
        import ccxt.async_support as ccxt_async
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ExchangeUnavailable(
            "ccxt is not installed. Install it with:  pip install ccxt\n"
            "It is required for live data; backtests can run from cached CSV without it."
        ) from exc
    return ccxt_async


class CCXTProvider:
    """Async market-data adapter over ccxt.

    Args:
        cfg: the ``exchange`` configuration section.
    """

    def __init__(self, cfg: ExchangeConfig) -> None:
        self.cfg = cfg
        self.name = cfg.id
        self._client: Any | None = None
        self._markets_loaded = False
        #: Remembers which optional endpoints this venue rejected, so an
        #: unsupported feature is attempted once and then skipped quietly.
        self._unsupported: set[str] = set()

    # -- lifecycle ----------------------------------------------------------
    @property
    def client(self) -> Any:
        """Lazily construct the ccxt client."""
        if self._client is None:
            ccxt_async = _import_ccxt()
            if not hasattr(ccxt_async, self.cfg.id):
                raise ExchangeUnavailable(
                    f"ccxt has no exchange named {self.cfg.id!r}. "
                    f"Check exchange.id in config.yaml."
                )
            options: dict[str, Any] = {
                "enableRateLimit": self.cfg.enable_rate_limit,
                "timeout": self.cfg.timeout_ms,
                "options": {
                    "defaultType": "future" if self.cfg.is_futures else "spot",
                },
            }
            if self.cfg.has_credentials:
                options["apiKey"] = self.cfg.api_key
                options["secret"] = self.cfg.api_secret
            else:
                _log.info(
                    "%s: no API credentials configured, using public endpoints only "
                    "(sufficient for OHLCV, tickers, funding and open interest)",
                    self.cfg.id,
                )
            client = getattr(ccxt_async, self.cfg.id)(options)
            if self.cfg.testnet:
                if not client.has.get("sandbox", True):
                    _log.warning("%s has no sandbox; ignoring exchange.testnet", self.cfg.id)
                else:
                    client.set_sandbox_mode(True)
                    _log.info("%s sandbox mode enabled", self.cfg.id)
            self._client = client
        return self._client

    async def load_markets(self) -> None:
        """Fetch instrument metadata once."""
        if self._markets_loaded:
            return
        await self._retry(lambda: self.client.load_markets(), "load_markets")
        self._markets_loaded = True
        _log.info("%s: loaded %d markets", self.name, len(self.client.markets or {}))

    async def close(self) -> None:
        """Close the underlying aiohttp session (ccxt requires this explicitly)."""
        if self._client is not None:
            await self._client.close()
            self._client = None
            self._markets_loaded = False

    async def __aenter__(self) -> CCXTProvider:
        await self.load_markets()
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.close()

    # -- helpers ------------------------------------------------------------
    async def _retry(self, func: Any, description: str) -> Any:
        return await async_retry(
            func,
            attempts=self.cfg.max_retries + 1,
            backoff=self.cfg.retry_backoff_seconds,
            description=f"{self.name}.{description}",
        )

    def _mark_unsupported(self, feature: str, symbol: str, exc: Exception) -> None:
        key = f"{feature}:{symbol}"
        if key not in self._unsupported:
            self._unsupported.add(key)
            _log.warning(
                "%s does not provide %s for %s (%s); it will be skipped and its scoring "
                "weight redistributed",
                self.name, feature, symbol, type(exc).__name__,
            )

    # -- data ---------------------------------------------------------------
    async def fetch_ohlcv(
        self, symbol: str, timeframe: str, limit: int = 500, since: int | None = None
    ) -> pd.DataFrame:
        """Fetch closed candles.

        Args:
            symbol: unified symbol, e.g. ``"BTC/USDT"``.
            timeframe: ccxt timeframe, e.g. ``"15m"``.
            limit: number of candles to request.
            since: earliest bar-open time, in milliseconds.

        Returns:
            Frame indexed by bar-open timestamp (UTC), ascending, with the
            forming bar removed. Empty frame if the venue returns nothing.

        Raises:
            ExchangeUnavailable: if ccxt is not installed.
        """
        raw = await self._retry(
            lambda: self.client.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit),
            f"fetch_ohlcv({symbol},{timeframe})",
        )
        return self.to_frame(raw, timeframe)

    @staticmethod
    def to_frame(raw: list[list[float]], timeframe: str) -> pd.DataFrame:
        """Convert ccxt's nested-list OHLCV into a validated frame.

        Exposed as a static method so cached CSV and vendor files can reuse the
        exact same normalisation.
        """
        columns = ["timestamp", "open", "high", "low", "close", "volume"]
        if not raw:
            return pd.DataFrame(
                columns=columns[1:], index=pd.DatetimeIndex([], tz="UTC", name="timestamp")
            )

        frame = pd.DataFrame(raw, columns=columns)
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], unit="ms", utc=True)
        frame = frame.set_index("timestamp").sort_index()
        frame = frame[~frame.index.duplicated(keep="last")]
        frame = frame.astype(float)

        # Drop the still-forming candle: its OHLC values are not final, and any
        # signal computed from them would change on the next poll.
        step_ms = timeframe_to_ms(timeframe)
        now_ms = int(utc_now().timestamp() * 1000)
        last_open_ms = int(frame.index[-1].timestamp() * 1000)
        if last_open_ms + step_ms > now_ms:
            frame = frame.iloc[:-1]

        return frame

    async def fetch_ticker(self, symbol: str) -> Ticker | None:
        """Last price plus the top of book when the venue includes it."""
        try:
            raw = await self._retry(
                lambda: self.client.fetch_ticker(symbol), f"fetch_ticker({symbol})"
            )
        except Exception as exc:  # noqa: BLE001 - ccxt raises a wide family
            self._mark_unsupported("ticker", symbol, exc)
            return None
        if not raw:
            return None

        timestamp = raw.get("timestamp")
        return Ticker(
            symbol=symbol,
            last=float(raw.get("last") or raw.get("close") or float("nan")),
            bid=float(raw["bid"]) if raw.get("bid") else float("nan"),
            ask=float(raw["ask"]) if raw.get("ask") else float("nan"),
            quote_volume_24h=float(raw["quoteVolume"]) if raw.get("quoteVolume") else float("nan"),
            timestamp=(
                datetime.fromtimestamp(timestamp / 1000, tz=UTC) if timestamp else utc_now()
            ),
        )

    async def fetch_funding_rate(self, symbol: str) -> FundingSnapshot | None:
        """Current funding rate, as a percentage per funding interval.

        Returns ``None`` on spot markets or venues without the endpoint — never
        a fabricated zero, which the scorer would otherwise read as "neutral
        positioning" rather than "unknown".
        """
        if not self.cfg.is_futures:
            return None
        try:
            raw = await self._retry(
                lambda: self.client.fetch_funding_rate(symbol), f"fetch_funding_rate({symbol})"
            )
        except Exception as exc:  # noqa: BLE001
            self._mark_unsupported("funding rate", symbol, exc)
            return None
        if not raw:
            return None

        rate = raw.get("fundingRate")
        if rate is None:
            return None
        next_time = raw.get("fundingTimestamp") or raw.get("nextFundingTimestamp")
        return FundingSnapshot(
            symbol=symbol,
            rate_pct=float(rate) * 100.0,  # ccxt reports a fraction
            next_funding_time=(
                datetime.fromtimestamp(next_time / 1000, tz=UTC) if next_time else None
            ),
        )

    async def fetch_open_interest(self, symbol: str) -> OpenInterestSnapshot | None:
        """Open interest now, with its change over the recent window.

        The change is computed from the OI *history* endpoint when the venue has
        one; without history there is no change to report and the field stays
        ``nan`` so the OI/price cross-read is skipped rather than guessed.
        """
        if not self.cfg.is_futures:
            return None
        try:
            raw = await self._retry(
                lambda: self.client.fetch_open_interest(symbol), f"fetch_open_interest({symbol})"
            )
        except Exception as exc:  # noqa: BLE001
            self._mark_unsupported("open interest", symbol, exc)
            return None
        if not raw:
            return None

        value = raw.get("openInterestAmount") or raw.get("openInterestValue")
        if value is None:
            return None

        change_pct = float("nan")
        if self.client.has.get("fetchOpenInterestHistory"):
            try:
                history = await self._retry(
                    lambda: self.client.fetch_open_interest_history(symbol, "5m", limit=12),
                    f"fetch_open_interest_history({symbol})",
                )
                series = [
                    float(item.get("openInterestAmount") or item.get("openInterestValue") or 0.0)
                    for item in (history or [])
                ]
                series = [x for x in series if x > 0]
                if len(series) >= 2 and series[0] > 0:
                    change_pct = 100.0 * (series[-1] - series[0]) / series[0]
            except Exception as exc:  # noqa: BLE001
                self._mark_unsupported("open interest history", symbol, exc)

        return OpenInterestSnapshot(symbol=symbol, value=float(value), change_pct=change_pct)

    async def fetch_liquidations(self, symbol: str) -> None:
        """Not implemented — deliberately.

        Binance exposes forced liquidations only over the
        ``!forceOrder@arr`` **websocket** stream; there is no REST endpoint that
        returns historical liquidations, and third-party aggregators (Coinglass
        and similar) are paid and rate-limited.

        Rather than ship a scraper that silently breaks, ``data.liquidations`` is
        ``false`` by default and this method is a documented no-op. To add it:
        subscribe to that stream in a separate task, aggregate notional per
        interval, and feed the result into
        :class:`data.models.ExternalContext` as an extra sentiment input.
        """
        _log.debug("liquidation feed is not implemented; see the docstring for why")
        return
