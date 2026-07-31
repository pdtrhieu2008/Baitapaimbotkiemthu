"""Message templates for Telegram (HTML parse mode).

The signal layout follows the format requested in the specification, with the
risk figures kept together so the important numbers are readable on a phone
without scrolling.

Every value that reaches a message goes through :func:`escape`, because a symbol
or an exchange-supplied string containing ``<`` or ``&`` would otherwise make
Telegram reject the whole message with HTTP 400.
"""

from __future__ import annotations

import html

from analysis.context import MarketSnapshot
from backtest.metrics import PerformanceMetrics
from risk.portfolio import Trade
from strategies.base import Rejection, Signal, SignalSide, SignalStrength
from utils.helpers import format_price

__all__ = [
    "escape",
    "format_daily_summary",
    "format_error",
    "format_rejection",
    "format_signal",
    "format_startup",
    "format_trade_closed",
]

_DIVIDER = "=" * 26


def escape(value: object) -> str:
    """HTML-escape a value for Telegram's HTML parse mode."""
    return html.escape(str(value), quote=False)


def _strength_badge(strength: SignalStrength) -> str:
    return {"strong": "STRONG ", "normal": "", "weak": "WEAK "}[strength.value]


def format_signal(signal: Signal, *, equity: float | None = None, quantity: float | None = None) -> str:
    """Render a trade signal.

    Args:
        signal: the emitted signal.
        equity: account equity, so the cash at risk can be shown.
        quantity: approved position size.

    Returns:
        An HTML message body.
    """
    side_word = "BUY" if signal.side is SignalSide.LONG else "SELL"
    targets = " / ".join(
        f"{format_price(tp.price)} ({tp.rr:g}R, {tp.fraction:.0%})" for tp in signal.take_profits
    )

    lines = [
        f"<b>{_DIVIDER}</b>",
        f"<b>{_strength_badge(signal.strength)}{side_word} SIGNAL</b>",
        f"<b>{_DIVIDER}</b>",
        f"Coin: <b>{escape(signal.symbol)}</b>",
        f"Timeframe: <b>{escape(signal.timeframe)}</b>",
        "",
        f"Entry: <code>{format_price(signal.entry)}</code>",
        f"Stop Loss: <code>{format_price(signal.stop_loss)}</code>"
        f"  (-{signal.stop_distance_pct:.2f}%)",
        f"Take Profit: <code>{targets}</code>",
        f"Risk/Reward: <b>{signal.risk_reward:.2f}</b> (net of fees)",
    ]

    if equity is not None and quantity is not None:
        risk_cash = signal.risk_per_unit * quantity
        lines.append(
            f"Risk: <b>{risk_cash:.2f}</b> "
            f"({100.0 * risk_cash / equity:.2f}% of {equity:.2f})"
        )
        lines.append(f"Size: <code>{quantity:.6g}</code> @ {format_price(signal.entry)}")

    notes = signal.notes
    lines += [
        f"Confidence: <b>{signal.score:.0f}/100</b> ({escape(signal.strength.value)})",
        "",
        f"Trend: {escape(notes.get('regime', 'n/a'))} / "
        f"{escape(notes.get('trend_strength', 'n/a'))} "
        f"(ADX {escape(notes.get('adx', 'n/a'))})",
        f"Structure: {escape(notes.get('structure', 'n/a'))}, "
        f"last break {escape(notes.get('last_break', 'none'))}",
        f"Volume: {escape(notes.get('volume', 'n/a'))}, "
        f"flow {escape(notes.get('volume_flow', 'n/a'))}",
        f"ATR: {format_price(signal.atr)} "
        f"({escape(notes.get('natr_pct', 'n/a'))}% of price)",
        f"Volatility: {escape(notes.get('volatility', 'n/a'))}",
    ]

    if "funding_pct" in notes:
        lines.append(f"Funding: {float(notes['funding_pct']):+.4f}%")  # type: ignore[arg-type]
    if "fear_greed" in notes:
        lines.append(f"Fear &amp; Greed: {float(notes['fear_greed']):.0f}")  # type: ignore[arg-type]

    if signal.mtf is not None:
        scores = " ".join(f"{tf}:{value:+.2f}" for tf, value in signal.mtf.scores.items())
        lines += ["", f"MTF ({signal.mtf.agreement:.0%} agreement): <code>{escape(scores)}</code>"]

    lines += ["", f"Trigger: <b>{escape(signal.trigger)}</b>", "Reason:"]
    for reason in signal.reasons[:12]:
        lines.append(f"  • {escape(reason)}")

    objections = notes.get("objections") or []
    if isinstance(objections, list) and objections:
        lines += ["", "Against:"]
        for objection in objections[:5]:
            lines.append(f"  • {escape(objection)}")

    gaps = notes.get("data_gaps")
    if isinstance(gaps, dict) and gaps:
        lines += ["", "<i>Data gaps: " + escape(", ".join(gaps)) + "</i>"]

    lines += [
        "",
        f"<i>Bar {signal.bar_time:%Y-%m-%d %H:%M} UTC</i>",
        "<i>Signal only - no order was placed.</i>",
        f"<b>{_DIVIDER}</b>",
    ]
    return "\n".join(lines)


def format_trade_closed(trade: Trade, metrics: PerformanceMetrics) -> str:
    """Render a closed-trade report with running performance.

    Args:
        trade: the completed round trip.
        metrics: performance over every trade so far.

    Returns:
        An HTML message body.
    """
    outcome = "PROFIT" if trade.pnl > 0 else ("LOSS" if trade.pnl < 0 else "SCRATCH")
    sign = "+" if trade.pnl > 0 else ""

    lines = [
        f"<b>{_DIVIDER}</b>",
        f"<b>TRADE CLOSED - {outcome}</b>",
        f"<b>{_DIVIDER}</b>",
        f"Coin: <b>{escape(trade.symbol)}</b>  ({escape(trade.side.name)}, {escape(trade.timeframe)})",
        f"Entry -> Exit: <code>{format_price(trade.entry_price)} -> "
        f"{format_price(trade.exit_price)}</code>",
        f"Exit reason: <b>{escape(trade.exit_reason.value)}</b>",
        "",
        f"PnL: <b>{sign}{trade.pnl:.4f}</b> ({sign}{trade.return_pct:.2f}% of notional)",
        f"R multiple: <b>{trade.r_multiple:+.2f}R</b>",
        f"Fees paid: {trade.fees:.4f}",
        f"Excursion: best {trade.mfe_r:+.2f}R, worst {trade.mae_r:+.2f}R",
        f"Held: {trade.bars_held} bars ({trade.duration_hours:.1f}h)",
        f"Signal score was: {trade.signal_score:.0f} ({escape(trade.trigger)})",
        "",
        "<b>Running performance</b>",
        f"Trades: {metrics.total_trades}  |  Win rate: {metrics.win_rate:.1f}%",
        f"Profit factor: {metrics.profit_factor:.2f}  |  "
        f"Expectancy: {metrics.expectancy_r:+.3f}R",
        f"Net: {metrics.net_profit:+.4f} ({metrics.net_profit_pct:+.2f}%)",
        f"Max drawdown: {metrics.max_drawdown_pct:.2f}%",
    ]
    if not metrics.is_statistically_meaningful:
        lines.append(
            f"<i>Only {metrics.total_trades} trades - these ratios are still noise.</i>"
        )
    lines.append(f"<b>{_DIVIDER}</b>")
    return "\n".join(lines)


def format_daily_summary(
    metrics: PerformanceMetrics, risk_snapshot: dict[str, object], scanned: int, signals: int
) -> str:
    """Render the end-of-day report."""
    lines = [
        f"<b>{_DIVIDER}</b>",
        "<b>DAILY SUMMARY</b>",
        f"<b>{_DIVIDER}</b>",
        f"Equity: <b>{risk_snapshot.get('equity')}</b>  "
        f"(peak {risk_snapshot.get('peak_equity')})",
        f"Drawdown: {risk_snapshot.get('drawdown_pct')}%  |  "
        f"today {risk_snapshot.get('daily_loss_pct')}%",
        f"Open positions: {risk_snapshot.get('open_positions')}",
        f"Halt state: <b>{escape(risk_snapshot.get('halt', 'none'))}</b>",
        "",
        f"Symbols scanned: {scanned}  |  Signals emitted: {signals}",
        f"Closed trades: {metrics.total_trades}  |  Win rate: {metrics.win_rate:.1f}%",
        f"Expectancy: {metrics.expectancy_r:+.3f}R  |  "
        f"Profit factor: {metrics.profit_factor:.2f}",
        f"Net: {metrics.net_profit:+.4f} ({metrics.net_profit_pct:+.2f}%)",
    ]
    if metrics.warnings:
        lines += ["", "<b>Notes</b>"]
        for warning in metrics.warnings[:4]:
            lines.append(f"  • {escape(warning)}")
    lines.append(f"<b>{_DIVIDER}</b>")
    return "\n".join(lines)


def format_error(context: str, exc: BaseException) -> str:
    """Render an error alert."""
    return "\n".join(
        [
            "<b>BOT ERROR</b>",
            f"Where: <code>{escape(context)}</code>",
            f"Type: <code>{escape(type(exc).__name__)}</code>",
            f"Message: <code>{escape(str(exc)[:600])}</code>",
            "",
            "<i>See logs/error.log for the traceback.</i>",
        ]
    )


def format_startup(
    mode: str, symbols: list[str], timeframe: str, strategy: str, equity: float
) -> str:
    """Render the start-up banner."""
    return "\n".join(
        [
            "<b>QUANTBOT STARTED</b>",
            f"Mode: <b>{escape(mode)}</b>  (no orders are ever placed)",
            f"Strategy: {escape(strategy)}",
            f"Decision timeframe: {escape(timeframe)}",
            f"Symbols: {escape(', '.join(symbols))}",
            f"Tracked equity: {equity:.2f}",
        ]
    )


def format_rejection(rejection: Rejection, snapshot: MarketSnapshot | None = None) -> str:
    """Render a rejection — used only at DEBUG verbosity.

    Not sent by default: on a 30-second loop over several symbols this would be
    hundreds of messages a day and would train the operator to ignore the chat.
    """
    lines = [
        "<b>NO TRADE</b>",
        f"{escape(rejection.symbol)} {escape(rejection.timeframe)}",
        f"Blocked by: <b>{escape(rejection.gate)}</b>",
        f"Detail: {escape(rejection.detail)}",
    ]
    if rejection.score is not None:
        lines.append(f"Score reached: {rejection.score:.0f}")
    if snapshot is not None:
        lines.append(f"Close: {format_price(snapshot.primary.close)}")
    return "\n".join(lines)
