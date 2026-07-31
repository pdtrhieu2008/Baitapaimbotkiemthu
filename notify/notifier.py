"""Notification facade.

One object the rest of the bot talks to, so no other module needs to know
whether Telegram is configured, enabled or reachable.

Two guarantees:

* **Nothing is ever lost.** Every notification is written to
  ``logs/telegram.log`` and to the JSONL event stream *before* it is sent, so a
  run with Telegram disabled still leaves a complete record.
* **Notifications cannot break trading.** Every method swallows its exceptions.
  A failed message is an operational annoyance; an unhandled exception in the
  trading loop is a position left unmanaged.
"""

from __future__ import annotations

from datetime import datetime

from analysis.context import MarketSnapshot
from backtest.metrics import PerformanceMetrics
from config.settings import Settings
from notify import formatter
from notify.telegram_client import TelegramClient
from risk.portfolio import Trade
from strategies.base import Rejection, Signal
from utils.helpers import utc_now
from utils.logger import EventLog, get_logger

__all__ = ["Notifier"]

_log = get_logger("telegram")
_signal_log = get_logger("signal")
_trade_log = get_logger("trade")


class Notifier:
    """Route signals, trades, errors and summaries to Telegram and the logs.

    Args:
        settings: validated configuration.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.cfg = settings.telegram
        self.client = TelegramClient(settings.telegram)
        self.events = EventLog(settings.logging)
        self._last_summary_day: str = ""

        if not self.cfg.enabled:
            _log.warning(
                "Telegram is disabled (telegram.enabled=false). Signals and trades will be "
                "written to logs/signal.log, logs/trade.log and the .jsonl streams only."
            )
        elif not self.cfg.is_configured:  # pragma: no cover - config validation catches this
            _log.error("Telegram is enabled but not configured; messages will be dropped")

    async def close(self) -> None:
        """Close the HTTP session."""
        await self.client.close()

    # -- internals ----------------------------------------------------------
    async def _send(self, text: str, *, enabled: bool, silent: bool = False) -> bool:
        """Send if the relevant switch is on; never raise."""
        if not enabled or not self.cfg.is_configured:
            return False
        try:
            return await self.client.send_message(text, disable_notification=silent)
        except Exception as exc:  # noqa: BLE001 - notifications must never propagate
            _log.error("notification failed: %s: %s", type(exc).__name__, exc)
            return False

    # -- public API ---------------------------------------------------------
    async def signal(
        self, signal: Signal, *, equity: float | None = None, quantity: float | None = None
    ) -> None:
        """Announce a signal and record it."""
        _signal_log.info("SIGNAL %s", signal.summary())
        self.events.write("signal", "signal_emitted", signal.to_dict())
        await self._send(
            formatter.format_signal(signal, equity=equity, quantity=quantity),
            enabled=self.cfg.send_signals,
        )

    async def rejection(  # noqa: ARG002 - snapshot kept for a stable call signature
        self, rejection: Rejection, snapshot: MarketSnapshot | None = None
    ) -> None:
        """Record a rejection.

        Deliberately **not** sent to Telegram: a 30-second loop over several
        symbols produces hundreds of these a day, and a chat full of "no trade"
        teaches the operator to ignore it. The full trail lives in
        ``logs/signal.jsonl`` and is what the backtest's rejection histogram is
        built from.
        """
        _signal_log.debug("REJECT %s", rejection.summary())
        self.events.write("signal", "signal_rejected", rejection.to_dict())

    async def trade_closed(self, trade: Trade, metrics: PerformanceMetrics) -> None:
        """Announce a completed round trip."""
        _trade_log.info(
            "CLOSED %s %s pnl=%.4f r=%.2f reason=%s",
            trade.symbol, trade.side.name, trade.pnl, trade.r_multiple, trade.exit_reason.value,
        )
        self.events.write("trade", "trade_closed", trade.to_dict())
        await self._send(
            formatter.format_trade_closed(trade, metrics), enabled=self.cfg.send_trades
        )

    async def trade_opened(self, trade_info: dict[str, object]) -> None:
        """Record a position open (no message: the signal already announced it)."""
        _trade_log.info("OPENED %s", trade_info)
        self.events.write("trade", "trade_opened", trade_info)

    async def stop_moved(self, symbol: str, description: str) -> None:
        """Record an in-trade stop adjustment."""
        _trade_log.info("STOP %s: %s", symbol, description)
        self.events.write("trade", "stop_moved", {"symbol": symbol, "action": description})

    async def error(self, context: str, exc: BaseException) -> None:
        """Alert on an error."""
        _log.error("error in %s: %s: %s", context, type(exc).__name__, exc, exc_info=exc)
        self.events.write(
            "error", "exception",
            {"context": context, "type": type(exc).__name__, "message": str(exc)[:1000]},
        )
        await self._send(formatter.format_error(context, exc), enabled=self.cfg.send_errors)

    async def startup(self, equity: float) -> None:
        """Announce start-up."""
        message = formatter.format_startup(
            self.settings.app.mode,
            self.settings.universe.symbols,
            self.settings.data.primary_timeframe,
            self.settings.strategy.name,
            equity,
        )
        _log.info("starting: mode=%s symbols=%s", self.settings.app.mode, self.settings.universe.symbols)
        await self._send(message, enabled=True, silent=True)

    async def shutdown(self, reason: str) -> None:
        """Announce shutdown."""
        _log.info("shutting down: %s", reason)
        await self._send(
            f"<b>QUANTBOT STOPPED</b>\nReason: {formatter.escape(reason)}",
            enabled=True,
            silent=True,
        )

    async def performance(self, snapshot: dict[str, object]) -> None:
        """Record a periodic performance sample (no message)."""
        get_logger("performance").info("%s", snapshot)
        self.events.write("performance", "snapshot", snapshot)

    async def daily_summary(
        self,
        metrics: PerformanceMetrics,
        risk_snapshot: dict[str, object],
        *,
        scanned: int,
        signals: int,
        now: datetime | None = None,
    ) -> bool:
        """Send the daily summary once, at the configured UTC hour.

        Returns:
            ``True`` if a summary was sent on this call.
        """
        moment = now or utc_now()
        day = moment.date().isoformat()
        if not self.cfg.send_daily_summary:
            return False
        if moment.hour != self.cfg.daily_summary_hour_utc or self._last_summary_day == day:
            return False

        self._last_summary_day = day
        self.events.write(
            "performance", "daily_summary",
            {"metrics": metrics.to_dict(), "risk": risk_snapshot, "scanned": scanned},
        )
        await self._send(
            formatter.format_daily_summary(metrics, risk_snapshot, scanned, signals),
            enabled=True,
        )
        return True

    async def test(self) -> tuple[bool, str]:
        """Verify the Telegram configuration end to end."""
        ok, detail = await self.client.test_connection()
        if ok:
            ok = await self.client.send_message(
                "<b>QUANTBOT TEST</b>\nIf you can read this, notifications work.",
                disable_notification=True,
            )
            detail = f"{detail}; test message {'delivered' if ok else 'failed'}"
        return ok, detail
