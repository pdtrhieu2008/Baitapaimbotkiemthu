"""Backtesting, performance measurement and parameter search."""

from backtest.engine import Backtester, BacktestResult
from backtest.metrics import PerformanceMetrics, compute_metrics, equity_dataframe
from backtest.optimizer import OptimisationReport, Optimiser, TrialResult
from backtest.report import export_csv, render_html, text_report

__all__ = [
    "BacktestResult",
    "Backtester",
    "OptimisationReport",
    "Optimiser",
    "PerformanceMetrics",
    "TrialResult",
    "compute_metrics",
    "equity_dataframe",
    "export_csv",
    "render_html",
    "text_report",
]
