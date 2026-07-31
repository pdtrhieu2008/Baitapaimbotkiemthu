"""Event-driven, bar-closed backtester.

Design constraints, in priority order:

1. **No lookahead.** A signal is computed from bar *i*'s closed values and filled
   at bar *i+1*'s **open**. Higher timeframes are sliced to the last bar that had
   genuinely *closed* by bar *i*'s close, never to the bar that contains it.
2. **Same code as live.** The backtester drives the real
   :class:`~strategies.base.Strategy`, :class:`~risk.manager.RiskManager` and
   :class:`~risk.portfolio.Portfolio`. There is no parallel implementation of the
   rules, which is the usual source of backtest/live divergence.
3. **Pessimistic intrabar.** When a bar's range contains both the stop and a
   target, the stop is assumed to have been hit first. Bar data cannot tell us
   the order, and assuming the good outcome is how backtests become fiction.

The per-bar sequence is:

    fill pending order (at this bar's open)
      -> resolve open positions against this bar's high/low
      -> advance break-even / trailing stops using this bar's extremes
      -> evaluate the strategy on this now-closed bar
      -> queue an order for the next bar
      -> record equity
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd

from analysis.context import ContextBuilder, MarketSnapshot, TimeframeContext
from backtest.metrics import PerformanceMetrics, compute_metrics, equity_dataframe
from config.settings import Settings
from data.models import ExternalContext
from risk.manager import RiskManager
from risk.portfolio import ExitReason, Portfolio, Position, Trade
from strategies.base import Rejection, Signal, SignalSide, Strategy
from utils.helpers import safe_div
from utils.logger import get_logger
from utils.timeframes import bars_per_year, sort_timeframes, timeframe_to_timedelta

__all__ = ["BacktestResult", "Backtester"]

_log = get_logger("backtest.engine")


@dataclass(slots=True)
class BacktestResult:
    """Everything a run produced."""

    symbol: str
    timeframe: str
    metrics: PerformanceMetrics
    trades: list[Trade]
    equity_curve: pd.DataFrame
    #: Count of rejections per gate — shows *why* the bot stayed out.
    rejections: dict[str, int] = field(default_factory=dict)
    signals_emitted: int = 0
    signals_taken: int = 0
    signals_skipped: dict[str, int] = field(default_factory=dict)
    settings_used: dict[str, object] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @property
    def objective_values(self) -> dict[str, float]:
        """Values the optimiser can maximise."""
        m = self.metrics
        return {
            "expectancy": m.expectancy_r,
            "profit_factor": m.profit_factor if np.isfinite(m.profit_factor) else 0.0,
            "sharpe": m.sharpe,
            "sortino": m.sortino,
            "net_profit": m.net_profit,
            "calmar": m.calmar,
        }

    def to_dict(self) -> dict[str, object]:
        return {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "metrics": self.metrics.to_dict(),
            "signals_emitted": self.signals_emitted,
            "signals_taken": self.signals_taken,
            "signals_skipped": self.signals_skipped,
            "rejections": dict(
                sorted(self.rejections.items(), key=lambda kv: kv[1], reverse=True)
            ),
            "warnings": self.warnings,
        }


@dataclass(slots=True)
class _PendingOrder:
    """A signal waiting for the next bar's open."""

    signal: Signal
    quantity: float


class Backtester:
    """Replay history through the live strategy and risk stack.

    Args:
        settings: validated configuration.
        strategy: the strategy to test.
    """

    def __init__(self, settings: Settings, strategy: Strategy) -> None:
        self.settings = settings
        self.strategy = strategy
        self.builder = ContextBuilder(settings.indicators, settings.structure)

    # -- public API ---------------------------------------------------------
    def run(
        self,
        frames: dict[str, pd.DataFrame],
        *,
        symbol: str | None = None,
        external: ExternalContext | None = None,
    ) -> BacktestResult:
        """Run a single-symbol backtest.

        Args:
            frames: raw OHLCV per timeframe. Must contain
                ``settings.backtest.timeframe``; extra timeframes are used for
                multi-timeframe confirmation and any that the strategy needs but
                that are absent will simply fail the MTF gate.
            symbol: symbol label; defaults to ``settings.backtest.symbol``.
            external: static external context (funding/sentiment). Historical
                funding and sentiment series are **not** reconstructed here — see
                the note in the README; passing ``None`` means the sentiment
                component is marked unavailable and its weight redistributed,
                which is the honest treatment.

        Returns:
            A :class:`BacktestResult`.

        Raises:
            ValueError: when the primary timeframe is missing or too short.
        """
        cfg = self.settings.backtest
        symbol = symbol or cfg.symbol
        primary_tf = cfg.timeframe

        if primary_tf not in frames:
            raise ValueError(
                f"frames must include the primary timeframe {primary_tf!r}; got {sorted(frames)}"
            )

        required = {primary_tf, *self.settings.strategy.mtf.all_timeframes}
        enriched = self._prepare(frames, primary_tf, required)
        primary = enriched[primary_tf]
        primary = self._slice_window(primary, cfg.start, cfg.end)

        warmup = self.settings.data.warmup_bars
        if len(primary) <= warmup + 2:
            raise ValueError(
                f"{len(primary)} bars of {primary_tf} data is not enough: warm-up alone needs "
                f"{warmup}. Fetch more history."
            )

        # Costs come from the risk section, so a backtest can never be run with
        # cheaper assumptions than the live bot uses.
        portfolio = Portfolio(
            cfg.initial_equity, self.settings.risk.fee_pct, self.settings.risk.slippage_pct
        )
        risk = RiskManager(self.settings.risk, portfolio)
        if hasattr(self.strategy, "reset"):
            self.strategy.reset()

        state = _RunState(external=external or ExternalContext(symbol=symbol))
        cache = _ContextCache(self.builder, enriched, symbol, required)

        opens = primary["open"].to_numpy(dtype=float)
        highs = primary["high"].to_numpy(dtype=float)
        lows = primary["low"].to_numpy(dtype=float)
        closes = primary["close"].to_numpy(dtype=float)
        atrs = primary["atr"].to_numpy(dtype=float)
        timestamps = primary.index

        for i in range(warmup, len(primary)):
            moment = timestamps[i].to_pydatetime()
            risk.roll_day(moment)

            # 1. fill the order queued on the previous bar, at this bar's open
            if state.pending is not None:
                self._fill_pending(state, portfolio, risk, opens[i], moment, symbol, i)

            # 2 & 3. resolve and then manage open positions on this bar
            self._process_positions(
                portfolio, risk, state,
                symbol=symbol, index=i, moment=moment,
                high=highs[i], low=lows[i], close=closes[i], atr=atrs[i],
            )

            if portfolio.open_count:
                state.bars_in_market += 1

            # 4 & 5. evaluate the closed bar and queue the next order
            if state.pending is None and not risk.is_halted:
                self._evaluate(
                    state, cache, portfolio, risk,
                    symbol=symbol, primary_tf=primary_tf,
                    bar_close_time=timestamps[i], index=i,
                )

            # 6. mark the book
            portfolio.record_equity(moment, {symbol: closes[i]})
            state.bars_processed += 1

        self._close_remaining(portfolio, risk, state, symbol, timestamps[-1], closes[-1], len(primary) - 1)
        return self._build_result(symbol, primary_tf, portfolio, risk, state, primary)

    # -- preparation --------------------------------------------------------
    def _prepare(
        self, frames: dict[str, pd.DataFrame], primary_tf: str, required: set[str]
    ) -> dict[str, pd.DataFrame]:
        """Enrich each *required* timeframe once.

        Timeframes the strategy never consults are skipped: enriching a 1-minute
        series that no MTF role references costs a full indicator pass and a lot
        of memory for nothing.
        """
        enriched: dict[str, pd.DataFrame] = {}
        for timeframe in sort_timeframes(list(frames), descending=False):
            if timeframe not in required:
                _log.debug("timeframe %s is not used by the strategy; not enriched", timeframe)
                continue
            frame = frames[timeframe]
            if frame is None or frame.empty:
                _log.warning("timeframe %s has no data; skipping", timeframe)
                continue
            enriched[timeframe] = self.builder.enrich(frame, timeframe)

        if primary_tf not in enriched:
            raise ValueError(f"primary timeframe {primary_tf!r} produced no enriched data")

        # A required timeframe with too little history can never satisfy the MTF
        # gate, and the resulting "no trades" is easy to misread as "no setups".
        # Say so explicitly, once, up front.
        needed = self.builder.engine.required_history
        for timeframe in sorted(required):
            frame = enriched.get(timeframe)
            if frame is None:
                _log.warning(
                    "timeframe %s is required by strategy.mtf but was not supplied: "
                    "every signal will be blocked by the mtf_alignment gate",
                    timeframe,
                )
            elif len(frame) < needed:
                _log.warning(
                    "timeframe %s has %d bars but %d are needed to seed the indicators "
                    "(longest lookback is EMA%d): its context will read as missing data and "
                    "the mtf_alignment gate will block every signal",
                    timeframe, len(frame), needed, self.settings.indicators.ema_trend,
                )
        return enriched

    @staticmethod
    def _slice_window(
        frame: pd.DataFrame, start: str | None, end: str | None
    ) -> pd.DataFrame:
        """Trim to the configured date window, keeping the index timezone-aware."""
        out = frame
        if start:
            out = out[out.index >= pd.Timestamp(start, tz=out.index.tz or "UTC")]
        if end:
            out = out[out.index <= pd.Timestamp(end, tz=out.index.tz or "UTC")]
        return out

    # -- per-bar steps ------------------------------------------------------
    def _fill_pending(
        self,
        state: _RunState,
        portfolio: Portfolio,
        risk: RiskManager,
        open_price: float,
        moment: datetime,
        symbol: str,
        index: int,
    ) -> None:
        """Fill the queued order at this bar's open, or drop it if it gapped."""
        pending = state.pending
        state.pending = None
        if pending is None:
            return

        signal = pending.signal
        long_side = signal.side is SignalSide.LONG

        # A gap straight through the stop means the trade was never viable at the
        # planned risk. Taking it anyway at a broken RR would flatter the result.
        if (long_side and open_price <= signal.stop_loss) or (
            not long_side and open_price >= signal.stop_loss
        ):
            state.skipped["gapped_through_stop"] = state.skipped.get("gapped_through_stop", 0) + 1
            return

        first_target = signal.take_profits[0].price
        if (long_side and open_price >= first_target) or (
            not long_side and open_price <= first_target
        ):
            state.skipped["gapped_past_target"] = state.skipped.get("gapped_past_target", 0) + 1
            return

        position = portfolio.open_position(
            signal, pending.quantity, timestamp=moment, price=open_price
        )
        # Store the loop index, which is the same unit _exit() subtracts from.
        state.entry_bar[position.id] = index
        state.signals_taken += 1
        _log.debug(
            "opened #%d %s %s at %.6g (planned %.6g)",
            position.id, signal.side.name, symbol, position.entry_price, signal.entry,
        )
        risk.update_breakers({symbol: open_price})

    def _process_positions(
        self,
        portfolio: Portfolio,
        risk: RiskManager,
        state: _RunState,
        *,
        symbol: str,
        index: int,
        moment: datetime,
        high: float,
        low: float,
        close: float,
        atr: float,
    ) -> None:
        """Resolve stops and targets on this bar, then advance the stops."""
        pessimistic = self.settings.backtest.intrabar == "pessimistic"

        for position in list(portfolio.positions.values()):
            if position.symbol != symbol:
                continue
            long_side = position.side is SignalSide.LONG
            stop_hit = (low <= position.stop_loss) if long_side else (high >= position.stop_loss)

            target = position.next_target()
            target_hit = False
            if target is not None:
                target_hit = (high >= target.price) if long_side else (low <= target.price)

            if stop_hit and (pessimistic or not target_hit):
                self._exit(
                    portfolio, risk, state, position, position.stop_loss,
                    ExitReason.TRAILING_STOP if position.trailing_active
                    else (ExitReason.BREAK_EVEN if position.breakeven_done else ExitReason.STOP_LOSS),
                    moment, index, fraction=1.0,
                )
                continue

            # Targets are taken in order; only one per bar, which is the
            # conservative reading of a single OHLC record.
            if target_hit and target is not None:
                position.targets_hit += 1
                is_last_target = position.targets_hit >= len(position.take_profits)
                fraction = (
                    1.0 if is_last_target else self._target_fraction(position, target.fraction)
                )
                self._exit(
                    portfolio, risk, state, position, target.price,
                    ExitReason.TAKE_PROFIT, moment, index, fraction=fraction,
                )
                if not position.is_open:
                    continue

            if stop_hit and target_hit and not pessimistic and position.is_open:
                self._exit(
                    portfolio, risk, state, position, position.stop_loss,
                    ExitReason.STOP_LOSS, moment, index, fraction=1.0,
                )
                continue

            if position.is_open:
                risk.manage(position, high=high, low=low, atr=atr)

    @staticmethod
    def _target_fraction(position: Position, planned_fraction: float) -> float:
        """Translate a fraction-of-total into a fraction-of-remaining.

        ``tp_split`` is expressed as a share of the *original* position, but
        :meth:`Portfolio.close` takes a share of what is *left*. Without this
        conversion a 50/50 split would close 50% and then 25%.
        """
        share_of_original = planned_fraction * position.quantity
        return float(np.clip(safe_div(share_of_original, position.remaining, 1.0), 0.0, 1.0))

    def _exit(
        self,
        portfolio: Portfolio,
        risk: RiskManager,
        state: _RunState,
        position: Position,
        price: float,
        reason: ExitReason,
        moment: datetime,
        index: int,
        *,
        fraction: float,
    ) -> None:
        """Close all or part of a position and update the risk counters."""
        bars_held = index - state.entry_bar.get(position.id, index)
        trade = portfolio.close(
            position, price, reason, timestamp=moment, fraction=fraction, bars_held=bars_held
        )
        if trade is not None:
            risk.on_trade_closed(trade)
            state.entry_bar.pop(position.id, None)

    def _evaluate(
        self,
        state: _RunState,
        cache: _ContextCache,
        portfolio: Portfolio,
        risk: RiskManager,
        *,
        symbol: str,
        primary_tf: str,
        bar_close_time: pd.Timestamp,
        index: int,
    ) -> None:
        """Run the strategy on the just-closed bar and queue an order."""
        contexts = cache.contexts_at(bar_close_time, primary_tf)
        if primary_tf not in contexts:
            return

        snapshot = MarketSnapshot(
            symbol=symbol,
            primary_timeframe=primary_tf,
            contexts=contexts,
            external=state.external,
        )
        verdict = self.strategy.evaluate(snapshot)

        if isinstance(verdict, Rejection):
            state.rejections[verdict.gate] = state.rejections.get(verdict.gate, 0) + 1
            return

        state.signals_emitted += 1
        decision = risk.approve(verdict, marks={symbol: verdict.entry})
        if not decision.approved:
            key = decision.reason.split(":")[0][:48] or "risk_rejected"
            state.skipped[key] = state.skipped.get(key, 0) + 1
            return

        if hasattr(self.strategy, "register_signal"):
            self.strategy.register_signal(verdict)
        state.pending = _PendingOrder(signal=verdict, quantity=decision.quantity)

    def _close_remaining(
        self,
        portfolio: Portfolio,
        risk: RiskManager,
        state: _RunState,
        symbol: str,
        last_timestamp: pd.Timestamp,
        last_close: float,
        index: int,
    ) -> None:
        """Flatten anything still open at the end of the sample.

        Reported under its own exit reason so these trades are never mistaken
        for strategy exits when reading the metrics.
        """
        moment = last_timestamp.to_pydatetime()
        for position in list(portfolio.positions.values()):
            self._exit(
                portfolio, risk, state, position, last_close,
                ExitReason.END_OF_DATA, moment, index, fraction=1.0,
            )

    def _build_result(
        self,
        symbol: str,
        primary_tf: str,
        portfolio: Portfolio,
        risk: RiskManager,
        state: _RunState,
        primary: pd.DataFrame,
    ) -> BacktestResult:
        """Assemble metrics and diagnostics."""
        cfg = self.settings.backtest
        periods = cfg.periods_per_year or bars_per_year(primary_tf)
        metrics = compute_metrics(
            portfolio.trades,
            portfolio.equity_curve,
            cfg.initial_equity,
            periods_per_year=periods,
            risk_free_rate=cfg.risk_free_rate,
            bars_in_market=state.bars_in_market,
            total_bars=state.bars_processed,
        )

        warnings = list(metrics.warnings)
        if state.external.sentiment is None and state.external.funding is None:
            warnings.append(
                "no historical funding/sentiment was supplied, so the sentiment component was "
                "marked unavailable throughout. Live scores will be computed over a slightly "
                "different weight base."
            )
        if risk.state.halt.value != "none":
            warnings.append(f"run ended with trading halted: {risk.state.halt.value}")

        return BacktestResult(
            symbol=symbol,
            timeframe=primary_tf,
            metrics=metrics,
            trades=portfolio.trades,
            equity_curve=equity_dataframe(portfolio.equity_curve),
            rejections=state.rejections,
            signals_emitted=state.signals_emitted,
            signals_taken=state.signals_taken,
            signals_skipped=state.skipped,
            settings_used=self.settings.to_dict(),
            warnings=warnings,
        )


@dataclass(slots=True)
class _RunState:
    """Mutable bookkeeping for one run."""

    external: ExternalContext
    pending: _PendingOrder | None = None
    entry_bar: dict[int, int] = field(default_factory=dict)
    rejections: dict[str, int] = field(default_factory=dict)
    skipped: dict[str, int] = field(default_factory=dict)
    signals_emitted: int = 0
    signals_taken: int = 0
    bars_processed: int = 0
    bars_in_market: int = 0


class _ContextCache:
    """Build per-timeframe contexts lazily, memoised by bar position.

    Higher-timeframe contexts change far more slowly than the entry timeframe: a
    daily context is identical for every 15-minute bar inside that day. Caching
    by ``(timeframe, position)`` turns an O(bars x timeframes) structure analysis
    into roughly O(bars) and is what makes a multi-timeframe backtest finish in
    seconds instead of minutes.
    """

    #: Only a handful of recent entries are ever needed, so the cache is capped
    #: to keep memory flat over a long run.
    MAX_ENTRIES = 64

    def __init__(
        self,
        builder: ContextBuilder,
        enriched: dict[str, pd.DataFrame],
        symbol: str,
        required: set[str] | None = None,
    ) -> None:
        self.builder = builder
        # Only analyse timeframes the strategy actually consults. A timeframe
        # that is downloaded but unused by the MTF configuration would otherwise
        # cost a full structure analysis on every bar for nothing.
        self.enriched = (
            enriched if required is None else {tf: f for tf, f in enriched.items() if tf in required}
        )
        self.symbol = symbol
        self._cache: dict[tuple[str, int], TimeframeContext] = {}
        self._durations = {tf: timeframe_to_timedelta(tf) for tf in self.enriched}

    def contexts_at(
        self, primary_bar_open: pd.Timestamp, primary_tf: str
    ) -> dict[str, TimeframeContext]:
        """Contexts for every timeframe, as known at the primary bar's close.

        Args:
            primary_bar_open: open timestamp of the primary bar being evaluated.
            primary_tf: the primary timeframe.

        Returns:
            Mapping of timeframe to context. A timeframe with no closed bar yet
            is simply absent, which the MTF gate treats as missing data.
        """
        primary_close = primary_bar_open + self._durations[primary_tf]
        contexts: dict[str, TimeframeContext] = {}

        for timeframe, frame in self.enriched.items():
            if timeframe == primary_tf:
                position = int(frame.index.get_indexer([primary_bar_open], method="pad")[0])
            else:
                # Latest bar whose own close is at or before the primary close:
                # open + duration <= primary_close  =>  open <= primary_close - duration.
                cutoff = primary_close - self._durations[timeframe]
                position = int(frame.index.searchsorted(cutoff, side="right")) - 1
            if position < 0:
                continue

            key = (timeframe, position)
            cached = self._cache.get(key)
            if cached is None:
                cached = self.builder.build(
                    self.symbol, timeframe, frame.iloc[: position + 1]
                )
                self._evict()
                self._cache[key] = cached
            contexts[timeframe] = cached

        return contexts

    def _evict(self) -> None:
        if len(self._cache) <= self.MAX_ENTRIES:
            return
        # Insertion-ordered dict: dropping the oldest keys is a cheap FIFO.
        for key in list(self._cache)[: len(self._cache) - self.MAX_ENTRIES]:
            self._cache.pop(key, None)
