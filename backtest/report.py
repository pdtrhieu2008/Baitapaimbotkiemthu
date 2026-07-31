"""Backtest reporting: console tables, CSV exports and an optional HTML report.

Plotly and matplotlib are imported lazily inside the functions that need them,
so the bot runs with neither installed. Charts are a convenience; the numbers
and the CSVs are the deliverable.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd

from backtest.engine import BacktestResult
from backtest.metrics import PerformanceMetrics
from utils.logger import get_logger

__all__ = ["export_csv", "render_html", "text_report"]

_log = get_logger("backtest.report")


def _row(label: str, value: object, width: int = 30) -> str:
    return f"  {label:<{width}} {value}"


def _fmt(value: float, digits: int = 2, suffix: str = "") -> str:
    if not np.isfinite(value):
        return "inf" if value > 0 else "n/a"
    return f"{value:,.{digits}f}{suffix}"


def text_report(result: BacktestResult) -> str:
    """Render a plain-text performance report.

    Args:
        result: a completed backtest.

    Returns:
        A multi-line string suitable for the console or a log file.
    """
    m: PerformanceMetrics = result.metrics
    lines: list[str] = []
    add = lines.append

    period = "n/a"
    if m.period_start and m.period_end:
        period = f"{m.period_start:%Y-%m-%d %H:%M} -> {m.period_end:%Y-%m-%d %H:%M} UTC"

    add("=" * 72)
    add(f"  BACKTEST  {result.symbol}  {result.timeframe}")
    add(f"  {period}   ({m.bars:,} bars)")
    add("=" * 72)

    add("\n-- Result " + "-" * 61)
    add(_row("Net profit", f"{_fmt(m.net_profit, 2)}  ({_fmt(m.net_profit_pct, 2, '%')})"))
    add(_row("Equity", f"{_fmt(m.initial_equity)} -> {_fmt(m.final_equity)}"))
    add(_row("Annualised return", _fmt(m.annualised_return_pct, 2, "%")))
    add(_row("Total fees paid", _fmt(m.total_fees, 4)))

    add("\n-- Trades " + "-" * 61)
    add(_row("Total / win / loss", f"{m.total_trades} / {m.wins} / {m.losses}"))
    add(_row("Win rate", _fmt(m.win_rate, 2, "%")))
    add(_row("Profit factor", _fmt(m.profit_factor, 3)))
    add(_row("Average win / loss", f"{_fmt(m.average_win, 4)} / {_fmt(m.average_loss, 4)}"))
    add(_row("Win/loss size ratio", _fmt(m.win_loss_ratio, 3)))
    add(_row("Largest win / loss", f"{_fmt(m.largest_win, 4)} / {_fmt(m.largest_loss, 4)}"))
    add(_row("Expectancy per trade", f"{_fmt(m.expectancy, 4)}  ({_fmt(m.expectancy_r, 3)} R)"))
    add(_row("Average realised R", _fmt(m.average_r, 3)))
    add(_row("Max consecutive win/loss", f"{m.max_consecutive_wins} / {m.max_consecutive_losses}"))
    add(_row("Average hold", f"{m.average_bars_held:.1f} bars ({m.average_duration_hours:.1f}h)"))

    add("\n-- Risk " + "-" * 63)
    add(_row("Max drawdown", f"{_fmt(m.max_drawdown_pct, 2, '%')}  ({_fmt(m.max_drawdown_abs, 2)})"))
    add(_row("Longest underwater", f"{m.max_drawdown_duration_bars:,} bars"))
    add(_row("Sharpe / Sortino", f"{_fmt(m.sharpe, 3)} / {_fmt(m.sortino, 3)}"))
    add(_row("Calmar", _fmt(m.calmar, 3)))
    add(_row("SQN (system quality)", _fmt(m.sqn, 3)))
    add(_row("Recovery factor", _fmt(m.recovery_factor, 3)))
    add(_row("Market exposure", _fmt(m.exposure_pct, 2, "%")))

    add("\n-- Signal funnel " + "-" * 54)
    add(_row("Signals emitted / taken", f"{result.signals_emitted} / {result.signals_taken}"))
    if result.signals_skipped:
        add("  skipped by risk manager:")
        for reason, count in sorted(
            result.signals_skipped.items(), key=lambda kv: kv[1], reverse=True
        ):
            add(f"      {count:>6}  {reason}")
    if result.rejections:
        add("  rejected by gate (why the bot stayed out):")
        for gate, count in sorted(result.rejections.items(), key=lambda kv: kv[1], reverse=True):
            add(f"      {count:>6}  {gate}")
    if m.by_exit_reason:
        add("  exits by reason:")
        for reason, count in sorted(m.by_exit_reason.items(), key=lambda kv: kv[1], reverse=True):
            add(f"      {count:>6}  {reason}")

    if result.warnings:
        add("\n-- Warnings " + "-" * 59)
        for warning in result.warnings:
            add(f"  ! {warning}")

    add("\n" + "=" * 72)
    verdict = (
        "POSITIVE expectancy on this sample"
        if m.has_positive_expectancy
        else "NEGATIVE expectancy - do not trade this configuration"
    )
    add(f"  {verdict}")
    if not m.is_statistically_meaningful:
        add("  Sample too small to be conclusive. Extend the history before deciding.")
    add(
        "  A backtest is a necessary filter, not a forecast. Paper-trade a validated\n"
        "  configuration before risking money."
    )
    add("=" * 72)
    return "\n".join(lines)


def export_csv(result: BacktestResult, directory: str | Path) -> dict[str, Path]:
    """Write trades and the equity curve to CSV.

    Args:
        result: a completed backtest.
        directory: output directory, created if missing.

    Returns:
        Mapping of artefact name to the path written.
    """
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    stem = f"{result.symbol.replace('/', '')}_{result.timeframe}"
    written: dict[str, Path] = {}

    trades_path = out / f"{stem}_trades.csv"
    rows = [trade.to_dict() for trade in result.trades]
    with trades_path.open("w", encoding="utf-8", newline="") as handle:
        if rows:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        else:
            handle.write("# no trades were taken\n")
    written["trades"] = trades_path

    equity_path = out / f"{stem}_equity.csv"
    result.equity_curve.to_csv(equity_path)
    written["equity"] = equity_path

    _log.info("exported %s", ", ".join(str(p) for p in written.values()))
    return written


def render_html(result: BacktestResult, path: str | Path) -> Path | None:
    """Render an interactive HTML report with Plotly.

    Args:
        result: a completed backtest.
        path: output ``.html`` file.

    Returns:
        The path written, or ``None`` when Plotly is not installed (the text
        report and the CSVs remain available either way).
    """
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        _log.warning(
            "plotly is not installed, skipping the HTML report. "
            "Install it with: pip install plotly"
        )
        return None

    curve = result.equity_curve
    if curve.empty:
        _log.warning("equity curve is empty, nothing to render")
        return None

    figure = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.5, 0.25, 0.25],
        vertical_spacing=0.06,
        subplot_titles=("Equity", "Drawdown %", "Trade R multiples"),
    )
    figure.add_trace(
        go.Scatter(x=curve.index, y=curve["equity"], name="equity", line={"width": 1.6}),
        row=1, col=1,
    )
    figure.add_trace(
        go.Scatter(x=curve.index, y=curve["peak"], name="peak", line={"width": 1, "dash": "dot"}),
        row=1, col=1,
    )
    figure.add_trace(
        go.Scatter(
            x=curve.index, y=curve["drawdown_pct"], name="drawdown %",
            fill="tozeroy", line={"width": 1},
        ),
        row=2, col=1,
    )
    if result.trades:
        closed = [t.closed_at for t in result.trades]
        r_values = [t.r_multiple for t in result.trades]
        figure.add_trace(
            go.Bar(
                x=closed, y=r_values, name="R multiple",
                marker_color=["#2e9e5b" if r > 0 else "#c0392b" for r in r_values],
            ),
            row=3, col=1,
        )

    m = result.metrics
    subtitle = (
        f"{result.symbol} {result.timeframe} | {m.total_trades} trades | "
        f"win {m.win_rate:.1f}% | PF {m.profit_factor:.2f} | "
        f"expectancy {m.expectancy_r:+.3f}R | maxDD {m.max_drawdown_pct:.2f}%"
    )
    figure.update_layout(
        title={"text": f"Backtest report<br><sub>{subtitle}</sub>"},
        template="plotly_white",
        height=900,
        hovermode="x unified",
        legend={"orientation": "h", "y": 1.02, "x": 0},
    )

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    warning_block = ""
    if result.warnings:
        items = "".join(f"<li>{w}</li>" for w in result.warnings)
        warning_block = (
            "<div style='font-family:system-ui;max-width:1100px;margin:1rem auto;"
            "padding:1rem;border-left:4px solid #d68910;background:#fdf5e6'>"
            f"<b>Warnings</b><ul>{items}</ul></div>"
        )
    body = figure.to_html(full_html=True, include_plotlyjs="inline")
    if warning_block:
        body = body.replace("</body>", warning_block + "</body>")
    target.write_text(body, encoding="utf-8")
    _log.info("wrote HTML report to %s", target)
    return target


def trades_dataframe(result: BacktestResult) -> pd.DataFrame:
    """Trades as a DataFrame, for notebook analysis."""
    if not result.trades:
        return pd.DataFrame()
    return pd.DataFrame([trade.to_dict() for trade in result.trades])
