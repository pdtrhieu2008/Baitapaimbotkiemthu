"""OHLCV storage: in-memory cache plus an on-disk CSV mirror.

Two jobs:

* **Incremental refresh.** After the first load only the missing tail is
  requested, which keeps a multi-symbol multi-timeframe scan inside the venue's
  rate limits.
* **Offline backtests.** Cached CSV means a backtest can be re-run, and an
  optimiser can evaluate hundreds of candidates, without touching the network —
  and, importantly, over the *same* bars every time, so two runs are comparable.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from config.settings import DataConfig
from data.base import MarketDataProvider
from indicators.base import validate_ohlcv
from utils.logger import get_logger
from utils.timeframes import timeframe_to_ms

__all__ = ["OHLCVRepository"]

_log = get_logger("data.repository")


class OHLCVRepository:
    """Fetch, cache and persist candles.

    Args:
        cfg: the ``data`` configuration section.
        provider: venue adapter. May be ``None`` for a purely offline session,
            in which case only cached data is available.
    """

    def __init__(self, cfg: DataConfig, provider: MarketDataProvider | None = None) -> None:
        self.cfg = cfg
        self.provider = provider
        self._memory: dict[tuple[str, str], pd.DataFrame] = {}
        self._cache_dir = Path(cfg.cache_dir)
        if cfg.persist:
            self._cache_dir.mkdir(parents=True, exist_ok=True)

    # -- paths --------------------------------------------------------------
    def _path(self, symbol: str, timeframe: str) -> Path:
        safe = symbol.replace("/", "").replace(":", "_")
        return self._cache_dir / f"{safe}_{timeframe}.csv"

    # -- disk ---------------------------------------------------------------
    def load_from_disk(self, symbol: str, timeframe: str) -> pd.DataFrame | None:
        """Read the cached CSV, or ``None`` if absent or unreadable."""
        path = self._path(symbol, timeframe)
        if not path.is_file():
            return None
        try:
            frame = pd.read_csv(path, index_col=0, parse_dates=True)
        except (OSError, ValueError, pd.errors.ParserError) as exc:
            _log.warning("cannot read cache %s: %s", path, exc)
            return None

        if frame.empty:
            return None
        if frame.index.tz is None:
            frame.index = frame.index.tz_localize("UTC")
        frame.index.name = "timestamp"
        frame = frame.sort_index()
        frame = frame[~frame.index.duplicated(keep="last")]
        try:
            validate_ohlcv(frame)
        except (TypeError, ValueError) as exc:
            _log.warning("cache %s is corrupt (%s); ignoring it", path, exc)
            return None
        return frame

    def save_to_disk(self, symbol: str, timeframe: str, frame: pd.DataFrame) -> None:
        """Persist the frame, replacing any previous file."""
        if not self.cfg.persist or frame.empty:
            return
        path = self._path(symbol, timeframe)
        try:
            # Write to a temporary file first: a crash mid-write would otherwise
            # leave a truncated cache that the next run has to discard.
            temporary = path.with_suffix(".csv.tmp")
            frame.to_csv(temporary)
            temporary.replace(path)
        except OSError as exc:
            _log.error("cannot write cache %s: %s", path, exc)

    # -- fetching -----------------------------------------------------------
    async def get(
        self, symbol: str, timeframe: str, *, refresh: bool = True
    ) -> pd.DataFrame:
        """Return candles for one symbol and timeframe.

        Args:
            symbol: unified symbol.
            timeframe: bar size.
            refresh: fetch the missing tail from the venue. Set ``False`` for a
                fully offline run.

        Returns:
            Frame of closed candles, ascending. May be empty when there is
            neither cache nor connectivity.
        """
        key = (symbol, timeframe)
        frame = self._memory.get(key)
        if frame is None:
            frame = self.load_from_disk(symbol, timeframe)
            if frame is not None:
                _log.debug("%s %s: loaded %d cached bars", symbol, timeframe, len(frame))

        if refresh and self.provider is not None:
            frame = await self._refresh(symbol, timeframe, frame)

        if frame is None:
            frame = pd.DataFrame(
                columns=["open", "high", "low", "close", "volume"],
                index=pd.DatetimeIndex([], tz="UTC", name="timestamp"),
            )

        self._memory[key] = frame
        return frame

    async def _refresh(
        self, symbol: str, timeframe: str, existing: pd.DataFrame | None
    ) -> pd.DataFrame:
        """Fetch the missing tail and merge it into ``existing``."""
        assert self.provider is not None  # noqa: S101 - guarded by the caller
        since: int | None = None
        if existing is not None and not existing.empty:
            # Re-request the last cached bar as well: overlapping by one bar is
            # how a gap caused by a missed poll gets healed instead of silently
            # persisting.
            last_open_ms = int(existing.index[-1].timestamp() * 1000)
            since = last_open_ms - timeframe_to_ms(timeframe)

        try:
            fresh = await self.provider.fetch_ohlcv(
                symbol, timeframe, limit=self.cfg.ohlcv_limit, since=since
            )
        except Exception as exc:  # noqa: BLE001 - network/venue errors are varied
            _log.error(
                "%s %s: fetch failed (%s: %s); continuing with %d cached bars",
                symbol, timeframe, type(exc).__name__, exc,
                0 if existing is None else len(existing),
            )
            return existing if existing is not None else pd.DataFrame()

        if fresh.empty:
            return existing if existing is not None else fresh

        if existing is None or existing.empty:
            merged = fresh
        else:
            merged = pd.concat([existing, fresh])
            merged = merged[~merged.index.duplicated(keep="last")].sort_index()

        # Bound memory and disk: keep a generous multiple of what any indicator
        # needs, not the entire history of the instrument.
        maximum = max(self.cfg.ohlcv_limit * 3, self.cfg.warmup_bars * 4)
        if len(merged) > maximum:
            merged = merged.iloc[-maximum:]

        gaps = self._detect_gaps(merged, timeframe)
        if gaps:
            _log.warning(
                "%s %s: %d gap(s) in the series, first at %s. Indicators spanning a gap are "
                "computed across it as if the bars were contiguous.",
                symbol, timeframe, len(gaps), gaps[0],
            )

        self.save_to_disk(symbol, timeframe, merged)
        return merged

    @staticmethod
    def _detect_gaps(frame: pd.DataFrame, timeframe: str) -> list[pd.Timestamp]:
        """Timestamps after which one or more bars are missing."""
        if len(frame) < 3:
            return []
        step = pd.Timedelta(milliseconds=timeframe_to_ms(timeframe))
        deltas = frame.index.to_series().diff()
        # Allow a small tolerance: venues occasionally shift a candle by seconds.
        breaks = deltas > step * 1.5
        return list(frame.index[breaks.fillna(False)])

    async def get_many(
        self, symbol: str, timeframes: list[str], *, refresh: bool = True
    ) -> dict[str, pd.DataFrame]:
        """Fetch several timeframes for one symbol.

        Requests are issued sequentially rather than concurrently: ccxt's own
        rate limiter is per-client, and firing six requests at once is the
        quickest way to earn a 429 and a temporary ban.
        """
        out: dict[str, pd.DataFrame] = {}
        for timeframe in timeframes:
            out[timeframe] = await self.get(symbol, timeframe, refresh=refresh)
        return out

    def cached_symbols(self) -> list[tuple[str, str]]:
        """List ``(symbol, timeframe)`` pairs present on disk."""
        if not self._cache_dir.is_dir():
            return []
        found: list[tuple[str, str]] = []
        for path in sorted(self._cache_dir.glob("*.csv")):
            stem = path.stem
            if "_" not in stem:
                continue
            base, timeframe = stem.rsplit("_", 1)
            found.append((base, timeframe))
        return found
