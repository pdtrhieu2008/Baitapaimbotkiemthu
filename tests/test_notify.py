"""Notification formatting and transport behaviour.

The formatter tests matter more than they look: an unescaped ``<`` or ``&`` makes
Telegram reject the *whole* message with HTTP 400, so a signal would be silently
lost at exactly the moment it was needed.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from backtest.metrics import compute_metrics
from config.settings import Settings
from notify import formatter
from notify.notifier import Notifier
from notify.telegram_client import MAX_MESSAGE_LENGTH, TelegramClient
from risk.planner import TradePlanner
from risk.portfolio import ExitReason, Trade
from strategies.base import Rejection, Signal, SignalSide, SignalStrength

BAR = datetime(2024, 6, 1, 12, 0, tzinfo=UTC)


def make_signal(settings: Settings, symbol: str = "BTC/USDT") -> Signal:
    plan = TradePlanner(settings.risk).plan(SignalSide.LONG, 67_000.0, 250.0)
    return Signal(
        symbol=symbol,
        timeframe="15m",
        side=SignalSide.LONG,
        strength=SignalStrength.STRONG,
        score=84.0,
        entry=plan.entry,
        stop_loss=plan.stop_loss,
        take_profits=plan.take_profits,
        risk_reward=plan.risk_reward_net,
        atr=250.0,
        bar_time=BAR,
        reasons=["EMA21 > EMA50", "MACD cross in the last 4 bars", "volume spike 2.1x average"],
        trigger="breakout",
        notes={
            "regime": "trend_up", "trend_strength": "strong", "volatility": "normal",
            "structure": "up", "last_break": "bos_bull", "volume": "2.10x average",
            "volume_flow": "+64%", "adx": 31.2, "rsi": 58.4, "natr_pct": 0.373,
            "stop_source": "structure", "rr_gross": 3.0, "objections": ["RSI already high"],
        },
    )


def make_trade(pnl: float = 42.0) -> Trade:
    return Trade(
        position_id=1, symbol="BTC/USDT", side=SignalSide.LONG, timeframe="15m",
        entry_price=67_000.0, exit_price=67_000.0 + pnl, quantity=0.01,
        opened_at=BAR, closed_at=BAR, pnl=pnl, fees=0.5,
        exit_reason=ExitReason.TAKE_PROFIT, initial_risk=20.0,
        r_multiple=pnl / 20.0, mae_r=-0.3, mfe_r=2.4, bars_held=18,
        signal_score=84.0, trigger="breakout",
    )


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------
def test_signal_message_contains_every_required_field(settings: Settings) -> None:
    message = formatter.format_signal(make_signal(settings), equity=1000.0, quantity=0.0123)

    for expected in (
        "BUY SIGNAL", "Coin:", "BTC/USDT", "Timeframe:", "15m", "Entry:", "Stop Loss:",
        "Take Profit:", "Risk/Reward:", "Risk:", "Confidence:", "Trend:", "Volume:",
        "ATR:", "Reason:", "Trigger:",
    ):
        assert expected in message, f"missing {expected!r}"

    assert "no order was placed" in message
    assert "EMA21 &gt; EMA50" in message, "the '>' in a reason must be escaped"


def test_signal_message_reports_cash_risk(settings: Settings) -> None:
    signal = make_signal(settings)
    quantity = 0.02
    message = formatter.format_signal(signal, equity=5000.0, quantity=quantity)
    expected = signal.risk_per_unit * quantity
    assert f"{expected:.2f}" in message
    assert f"{100.0 * expected / 5000.0:.2f}%" in message


def test_escape_neutralises_html(settings: Settings) -> None:
    signal = make_signal(settings, symbol="<b>EVIL</b>&CO")
    message = formatter.format_signal(signal)
    assert "<b>EVIL</b>&CO" not in message
    assert "&lt;b&gt;EVIL&lt;/b&gt;&amp;CO" in message


def test_trade_message_reports_running_performance() -> None:
    trade = make_trade()
    metrics = compute_metrics(
        [trade], [(BAR, 1000.0), (BAR, 1042.0)], 1000.0, periods_per_year=365.0
    )
    message = formatter.format_trade_closed(trade, metrics)

    assert "TRADE CLOSED - PROFIT" in message
    assert "R multiple" in message
    assert "Win rate" in message
    assert "Profit factor" in message
    # A one-trade sample must carry the noise caveat.
    assert "still noise" in message


def test_loss_is_labelled_as_such() -> None:
    trade = make_trade(pnl=-25.0)
    metrics = compute_metrics(
        [trade], [(BAR, 1000.0), (BAR, 975.0)], 1000.0, periods_per_year=365.0
    )
    assert "TRADE CLOSED - LOSS" in formatter.format_trade_closed(trade, metrics)


def test_error_message_truncates_and_escapes() -> None:
    message = formatter.format_error("cycle", ValueError("<script>" + "x" * 2000))
    assert "&lt;script&gt;" in message
    assert len(message) < 1200


def test_daily_summary_includes_the_risk_state() -> None:
    metrics = compute_metrics([], [(BAR, 1000.0)], 1000.0)
    snapshot = {
        "equity": 1013.5, "peak_equity": 1050.0, "drawdown_pct": 3.5,
        "daily_loss_pct": 0.0, "open_positions": 1, "halt": "none",
    }
    message = formatter.format_daily_summary(metrics, snapshot, scanned=42, signals=3)
    assert "DAILY SUMMARY" in message
    assert "1013.5" in message
    assert "Symbols scanned: 42" in message
    assert "Halt state" in message


def test_rejection_formatting() -> None:
    rejection = Rejection(
        symbol="ETH/USDT", timeframe="15m", gate="min_rr",
        detail="net RR 1.42 below the required 2.00", side=SignalSide.LONG, score=76.0,
    )
    message = formatter.format_rejection(rejection)
    assert "NO TRADE" in message
    assert "min_rr" in message
    assert "76" in message


# ---------------------------------------------------------------------------
# Transport
# ---------------------------------------------------------------------------
def test_long_messages_split_on_line_boundaries(settings: Settings) -> None:
    client = TelegramClient(settings.telegram)
    body = "\n".join(f"line {i} " + "x" * 80 for i in range(400))
    chunks = client._split(body)

    assert len(chunks) > 1
    assert all(len(chunk) <= MAX_MESSAGE_LENGTH for chunk in chunks)
    # Nothing may be dropped: truncating would lose the risk numbers at the end.
    assert "".join(chunks) == body


def test_short_messages_are_not_split(settings: Settings) -> None:
    client = TelegramClient(settings.telegram)
    assert client._split("hello") == ["hello"]


async def test_send_is_a_noop_when_unconfigured(settings: Settings) -> None:
    """A disabled Telegram must never raise, only decline."""
    client = TelegramClient(settings.telegram)
    assert settings.telegram.is_configured is False
    assert await client.send_message("anything") is False

    ok, detail = await client.test_connection()
    assert ok is False
    assert "disabled" in detail or "empty" in detail


async def test_notifier_records_everything_even_when_disabled(
    settings: Settings, tmp_path
) -> None:
    """With Telegram off, the JSONL trail must still be complete."""
    tuned = settings.with_overrides({"logging.dir": str(tmp_path), "logging.json_lines": True})
    notifier = Notifier(tuned)
    try:
        signal = make_signal(tuned)
        await notifier.signal(signal, equity=1000.0, quantity=0.01)
        await notifier.rejection(
            Rejection("ETH/USDT", "15m", "adx_min", "ADX 12 below 25")
        )
        trade = make_trade()
        metrics = compute_metrics([trade], [(BAR, 1000.0)], 1000.0, periods_per_year=365.0)
        await notifier.trade_closed(trade, metrics)
        await notifier.error("cycle", RuntimeError("boom"))
    finally:
        await notifier.close()

    signal_log = (tmp_path / "signal.jsonl").read_text()
    assert "signal_emitted" in signal_log
    assert "signal_rejected" in signal_log
    assert "BTC/USDT" in signal_log
    assert "trade_closed" in (tmp_path / "trade.jsonl").read_text()
    assert "exception" in (tmp_path / "error.jsonl").read_text()


async def test_daily_summary_fires_once_per_day(settings: Settings, tmp_path) -> None:
    tuned = settings.with_overrides(
        {"logging.dir": str(tmp_path), "telegram.daily_summary_hour_utc": "9"}
    )
    notifier = Notifier(tuned)
    try:
        metrics = compute_metrics([], [(BAR, 1000.0)], 1000.0)
        snapshot = {"equity": 1000.0}

        wrong_hour = datetime(2024, 6, 1, 8, 0, tzinfo=UTC)
        assert await notifier.daily_summary(
            metrics, snapshot, scanned=1, signals=0, now=wrong_hour
        ) is False

        right_hour = datetime(2024, 6, 1, 9, 5, tzinfo=UTC)
        assert await notifier.daily_summary(
            metrics, snapshot, scanned=1, signals=0, now=right_hour
        ) is True
        # Second call in the same hour must not repeat.
        assert await notifier.daily_summary(
            metrics, snapshot, scanned=1, signals=0, now=right_hour
        ) is False

        next_day = datetime(2024, 6, 2, 9, 5, tzinfo=UTC)
        assert await notifier.daily_summary(
            metrics, snapshot, scanned=1, signals=0, now=next_day
        ) is True
    finally:
        await notifier.close()


@pytest.mark.parametrize("value", [123, 45.6, None, "plain"])
def test_escape_accepts_any_scalar(value: object) -> None:
    assert isinstance(formatter.escape(value), str)
