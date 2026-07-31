"""The live loop: scan, analyse, notify, and track paper positions.

What it does *not* do is place orders. See the README section "Why there is no
execution module"; ``app.mode`` accepts ``signal`` and ``paper`` only, and the
configuration layer refuses ``live`` outright.

Per cycle, for each symbol:

1. Collect a snapshot (candles + ticker + funding + sentiment).
2. If the primary bar has closed since last time, apply that bar to any open
   position via the *same* resolver the backtester uses.
3. Run the market-condition guards.
4. Evaluate the strategy; log the rejection or notify the signal.
5. In ``paper`` mode, open the position the risk manager approved.

Then update the circuit breakers, persist state, maybe send the daily summary,
and sleep.
"""

from __future__ import annotations

import asyncio
import contextlib
import signal as signal_module

from analysis.context import MarketSnapshot
from backtest.metrics import compute_metrics
from config.settings import Settings
from data.collector import DataCollector
from data.exchange import CCXTProvider, ExchangeUnavailable
from live.state import StateStore
from notify.notifier import Notifier
from risk.execution import resolve_position_on_bar
from risk.guards import TradingGuards
from risk.manager import RiskManager
from risk.portfolio import Portfolio
from strategies import build_strategy
from strategies.base import Rejection, Signal
from utils.helpers import utc_now
from utils.logger import get_logger
from utils.timeframes import bars_per_year

__all__ = ["LiveTrader"]

_log = get_logger("live.trader")


class LiveTrader:
    """Signal-generating (and optionally paper-trading) runtime.

    Args:
        settings: validated configuration.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.provider = CCXTProvider(settings.exchange)
        self.collector = DataCollector(settings, self.provider)
        self.strategy = build_strategy(settings)
        self.portfolio = Portfolio(
            settings.risk.account_equity, settings.risk.fee_pct, settings.risk.slippage_pct
        )
        self.risk = RiskManager(settings.risk, self.portfolio)
        self.guards = TradingGuards(settings.risk)
        self.notifier = Notifier(settings)
        self.state = StateStore(settings.app.state_file, enabled=settings.app.persist_state)

        self._stop = asyncio.Event()
        self._last_bars: dict[str, str] = {}
        self._entry_bar_count: dict[int, int] = {}
        self._bar_counter = 0
        self._cycles = 0
        self._signals_today = 0
        self._scanned = 0

    @property
    def is_paper(self) -> bool:
        """Whether virtual positions are tracked (as opposed to signal-only)."""
        return self.settings.app.mode == "paper"

    # -- lifecycle ----------------------------------------------------------
    def request_stop(self, reason: str = "signal received") -> None:
        """Ask the loop to finish the current cycle and exit."""
        if not self._stop.is_set():
            _log.info("stop requested: %s", reason)
            self._stop.set()

    def _install_signal_handlers(self) -> None:
        """Route SIGINT/SIGTERM into a graceful shutdown.

        A hard kill in the middle of a cycle would leave the state file behind
        the in-memory truth; finishing the cycle and saving is what keeps the
        risk counters accurate across a deploy.
        """
        loop = asyncio.get_running_loop()
        for sig in (signal_module.SIGINT, signal_module.SIGTERM):
            with contextlib.suppress(NotImplementedError, ValueError):
                loop.add_signal_handler(sig, self.request_stop, f"{sig.name}")

    async def run(self) -> int:
        """Run until stopped.

        Returns:
            Process exit code: ``0`` normal, ``1`` on a fatal start-up error.
        """
        self._install_signal_handlers()

        try:
            await self.provider.load_markets()
        except ExchangeUnavailable as exc:
            _log.error("cannot start: %s", exc)
            return 1
        except Exception as exc:  # noqa: BLE001 - venue errors are varied
            _log.error("cannot reach %s: %s: %s", self.settings.exchange.id, type(exc).__name__, exc)
            await self.notifier.error("startup", exc)
            await self._teardown("startup failure")
            return 1

        restored_bars, restored = self.state.restore(self.portfolio, self.risk)
        if restored:
            self._last_bars = restored_bars

        report = await self.collector.warm_up()
        thin = {key: count for key, count in report.items() if count < 100}
        _log.info("warm-up complete: %d symbol/timeframe series loaded", len(report))
        if thin:
            _log.warning("series with little history (signals will be blocked): %s", thin)

        await self.notifier.startup(self.portfolio.equity())

        try:
            while not self._stop.is_set():
                started = utc_now()
                try:
                    await self._cycle()
                except asyncio.CancelledError:
                    raise
                except Exception as exc:  # noqa: BLE001 - the loop must survive
                    _log.exception("cycle failed")
                    await self.notifier.error("trading cycle", exc)

                elapsed = (utc_now() - started).total_seconds()
                delay = max(1.0, self.settings.app.loop_interval_seconds - elapsed)
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(self._stop.wait(), timeout=delay)
        finally:
            await self._teardown("stop requested")
        return 0

    async def _teardown(self, reason: str) -> None:
        """Persist state and release every resource."""
        self.state.save(self.portfolio, self.risk, self._last_bars)
        await self.notifier.shutdown(reason)
        await self.notifier.close()
        await self.collector.close()
        await self.provider.close()
        _log.info(
            "stopped after %d cycles; equity %.4f, %d open position(s)",
            self._cycles, self.portfolio.equity(), self.portfolio.open_count,
        )

    # -- one cycle ----------------------------------------------------------
    async def _cycle(self) -> None:
        """Scan every symbol once."""
        self._cycles += 1
        marks: dict[str, float] = {}

        for symbol in self.settings.universe.symbols:
            if self._stop.is_set():
                break
            snapshot = await self.collector.collect(symbol)
            if snapshot is None:
                continue
            self._scanned += 1
            marks[symbol] = snapshot.primary.close

            new_bar = self._is_new_bar(symbol, snapshot)
            if new_bar:
                await self._apply_bar(symbol, snapshot)

            if not new_bar:
                # Nothing has changed since the last evaluation of this symbol.
                continue

            self.risk.roll_day(snapshot.primary.bar_time)
            await self._evaluate(symbol, snapshot)

        halt = self.risk.update_breakers(marks)
        if halt.is_permanent and self.portfolio.open_count:
            closed = self.risk.force_close_all(marks, utc_now())
            for trade in closed:
                await self.notifier.trade_closed(trade, self._metrics())
            _log.error("permanent halt (%s): flattened %d position(s)", halt.value, len(closed))

        await self.notifier.performance(self.risk.snapshot())
        await self.notifier.daily_summary(
            self._metrics(),
            self.risk.snapshot(),
            scanned=self._scanned,
            signals=self._signals_today,
        )
        self.state.save(self.portfolio, self.risk, self._last_bars)

    def _is_new_bar(self, symbol: str, snapshot: MarketSnapshot) -> bool:
        """Whether this symbol's primary bar has not been processed yet."""
        current = snapshot.primary.bar_time.isoformat()
        if self._last_bars.get(symbol) == current:
            return False
        self._last_bars[symbol] = current
        return True

    async def _apply_bar(self, symbol: str, snapshot: MarketSnapshot) -> None:
        """Apply the newly closed bar to any open position in this symbol."""
        positions = self.portfolio.open_for(symbol)
        if not positions:
            return

        context = snapshot.primary
        row = context.frame.iloc[-1]
        high, low = float(row["high"]), float(row["low"])
        moment = context.bar_time
        self._bar_counter += 1

        for position in positions:
            outcome = resolve_position_on_bar(
                self.portfolio, self.risk, position,
                timestamp=moment, high=high, low=low, atr=context.atr,
                pessimistic=self.settings.backtest.intrabar == "pessimistic",
                bars_held=self._bar_counter - self._entry_bar_count.get(position.id, self._bar_counter),
            )
            if outcome.stop_action:
                await self.notifier.stop_moved(symbol, outcome.stop_action)
            if outcome.closed and outcome.trade is not None:
                self._entry_bar_count.pop(position.id, None)
                await self.notifier.trade_closed(outcome.trade, self._metrics())

    async def _evaluate(self, symbol: str, snapshot: MarketSnapshot) -> None:
        """Run the guards and the strategy on a freshly closed bar."""
        guard = self.guards.check(symbol, snapshot.primary, snapshot.external)
        if not guard.ok:
            await self.notifier.rejection(
                Rejection(
                    symbol=symbol,
                    timeframe=snapshot.primary_timeframe,
                    gate=f"guard:{guard.blocked_by}",
                    detail=guard.detail,
                    bar_time=snapshot.primary.bar_time,
                )
            )
            return

        verdict = self.strategy.evaluate(snapshot)
        if isinstance(verdict, Rejection):
            await self.notifier.rejection(verdict, snapshot)
            return

        decision = self.risk.approve(verdict, marks={symbol: snapshot.primary.close})
        if not decision.approved:
            await self.notifier.rejection(
                Rejection(
                    symbol=symbol,
                    timeframe=verdict.timeframe,
                    gate="risk_manager",
                    detail=decision.reason,
                    side=verdict.side,
                    score=verdict.score,
                    bar_time=verdict.bar_time,
                )
            )
            return

        if decision.reason:
            _log.info("%s: %s", symbol, decision.reason)

        self._signals_today += 1
        if hasattr(self.strategy, "register_signal"):
            self.strategy.register_signal(verdict)

        await self.notifier.signal(
            verdict, equity=self.portfolio.equity(), quantity=decision.quantity
        )

        if self.is_paper:
            await self._open_paper_position(verdict, decision.quantity)

    async def _open_paper_position(self, signal: Signal, quantity: float) -> None:
        """Track a virtual position for the signal.

        The fill is modelled at the signal bar's close plus slippage. Live, the
        real fill would be the next tick; the backtester uses the next bar's
        open. The difference is one bar of drift and it is documented rather than
        hidden, because paper results are meant to be compared with the backtest.
        """
        try:
            position = self.portfolio.open_position(signal, quantity, timestamp=utc_now())
        except ValueError as exc:
            _log.error("cannot open paper position for %s: %s", signal.symbol, exc)
            return
        self._entry_bar_count[position.id] = self._bar_counter
        await self.notifier.trade_opened(position.to_dict())

    # -- helpers ------------------------------------------------------------
    def _metrics(self) -> object:
        """Performance over the trades closed so far this run."""
        return compute_metrics(
            self.portfolio.trades,
            self.portfolio.equity_curve
            or [(utc_now(), self.portfolio.equity())],
            self.portfolio.initial_equity,
            periods_per_year=bars_per_year(self.settings.data.primary_timeframe),
        )

    async def scan_once(self) -> list[Signal | Rejection]:
        """Analyse every symbol exactly once and return the verdicts.

        Used by the ``scan`` CLI command: a single pass that shows what the bot
        currently sees, without starting the loop or tracking positions.
        """
        await self.provider.load_markets()
        verdicts: list[Signal | Rejection] = []
        try:
            for symbol in self.settings.universe.symbols:
                snapshot = await self.collector.collect(symbol)
                if snapshot is None:
                    verdicts.append(
                        Rejection(
                            symbol=symbol,
                            timeframe=self.settings.data.primary_timeframe,
                            gate="data",
                            detail="insufficient history or the symbol was filtered out",
                        )
                    )
                    continue
                guard = self.guards.check(symbol, snapshot.primary, snapshot.external)
                if not guard.ok:
                    verdicts.append(
                        Rejection(
                            symbol=symbol,
                            timeframe=snapshot.primary_timeframe,
                            gate=f"guard:{guard.blocked_by}",
                            detail=guard.detail,
                            bar_time=snapshot.primary.bar_time,
                        )
                    )
                    continue
                verdicts.append(self.strategy.evaluate(snapshot))
        finally:
            await self.notifier.close()
            await self.collector.close()
            await self.provider.close()
        return verdicts

