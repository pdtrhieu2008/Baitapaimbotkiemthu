"""Performance statistics.

Every figure below is computed from the trade list and the equity curve the
backtester actually produced — there is no separate "reporting" path that could
flatter the result.

Two honesty features are built in rather than bolted on:

* **Sample-size warnings.** A profit factor computed from 7 trades is not a
  measurement, and :attr:`PerformanceMetrics.warnings` says so explicitly.
* **No hidden annualisation.** Sharpe and Sortino need a bars-per-year figure;
  it is passed in from the timeframe rather than assumed, because getting it
  wrong scales the ratio by a constant and makes a mediocre system look
  excellent.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd

from risk.portfolio import Trade
from utils.helpers import safe_div

__all__ = ["PerformanceMetrics", "compute_metrics", "equity_dataframe"]

#: Below this many trades, ratio-style metrics are dominated by noise.
MIN_MEANINGFUL_TRADES = 30


@dataclass(slots=True)
class PerformanceMetrics:
    """Aggregated performance of a backtest or a live run."""

    # -- counts ---
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    breakeven: int = 0
    win_rate: float = 0.0
    # -- money ---
    initial_equity: float = 0.0
    final_equity: float = 0.0
    net_profit: float = 0.0
    net_profit_pct: float = 0.0
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    total_fees: float = 0.0
    # -- per-trade ---
    average_win: float = 0.0
    average_loss: float = 0.0
    win_loss_ratio: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    average_r: float = 0.0
    expectancy: float = 0.0          # cash per trade
    expectancy_r: float = 0.0        # R per trade
    average_rr_realised: float = 0.0
    # -- ratios ---
    profit_factor: float = 0.0
    sharpe: float = 0.0
    sortino: float = 0.0
    calmar: float = 0.0
    sqn: float = 0.0
    recovery_factor: float = 0.0
    # -- drawdown ---
    max_drawdown_pct: float = 0.0
    max_drawdown_abs: float = 0.0
    max_drawdown_duration_bars: int = 0
    # -- streaks & timing ---
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0
    average_bars_held: float = 0.0
    average_duration_hours: float = 0.0
    exposure_pct: float = 0.0
    # -- meta ---
    period_start: datetime | None = None
    period_end: datetime | None = None
    bars: int = 0
    annualised_return_pct: float = 0.0
    warnings: list[str] = field(default_factory=list)
    by_exit_reason: dict[str, int] = field(default_factory=dict)
    by_symbol: dict[str, float] = field(default_factory=dict)

    @property
    def is_statistically_meaningful(self) -> bool:
        """Whether the sample is large enough to read the ratios seriously."""
        return self.total_trades >= MIN_MEANINGFUL_TRADES

    @property
    def has_positive_expectancy(self) -> bool:
        """The only question that actually matters, in R terms."""
        return self.expectancy_r > 0

    def to_dict(self) -> dict[str, object]:
        return {
            "total_trades": self.total_trades,
            "win_rate_pct": round(self.win_rate, 2),
            "net_profit": round(self.net_profit, 4),
            "net_profit_pct": round(self.net_profit_pct, 3),
            "profit_factor": round(self.profit_factor, 3),
            "expectancy": round(self.expectancy, 4),
            "expectancy_r": round(self.expectancy_r, 4),
            "average_r": round(self.average_r, 3),
            "sharpe": round(self.sharpe, 3),
            "sortino": round(self.sortino, 3),
            "calmar": round(self.calmar, 3),
            "sqn": round(self.sqn, 3),
            "max_drawdown_pct": round(self.max_drawdown_pct, 3),
            "average_win": round(self.average_win, 4),
            "average_loss": round(self.average_loss, 4),
            "win_loss_ratio": round(self.win_loss_ratio, 3),
            "total_fees": round(self.total_fees, 4),
            "exposure_pct": round(self.exposure_pct, 2),
            "statistically_meaningful": self.is_statistically_meaningful,
            "warnings": self.warnings,
            "by_exit_reason": self.by_exit_reason,
        }


def equity_dataframe(curve: list[tuple[datetime, float]]) -> pd.DataFrame:
    """Convert the raw equity samples into an indexed frame.

    Args:
        curve: ``(timestamp, equity)`` samples.

    Returns:
        Frame indexed by timestamp with ``equity``, ``peak``, ``drawdown`` and
        ``drawdown_pct`` columns. Empty input yields an empty, correctly-typed
        frame so callers never need a special case.
    """
    if not curve:
        return pd.DataFrame(
            columns=["equity", "peak", "drawdown", "drawdown_pct"],
            index=pd.DatetimeIndex([], name="timestamp"),
        )
    frame = pd.DataFrame(curve, columns=["timestamp", "equity"]).set_index("timestamp")
    frame = frame[~frame.index.duplicated(keep="last")].sort_index()
    frame["peak"] = frame["equity"].cummax()
    frame["drawdown"] = frame["equity"] - frame["peak"]
    frame["drawdown_pct"] = 100.0 * frame["drawdown"] / frame["peak"].replace(0.0, np.nan)
    return frame


def _max_drawdown(frame: pd.DataFrame) -> tuple[float, float, int]:
    """Return ``(max_dd_pct, max_dd_abs, longest_underwater_bars)``."""
    if frame.empty:
        return 0.0, 0.0, 0
    dd_pct = float(-frame["drawdown_pct"].min(skipna=True) or 0.0)
    dd_abs = float(-frame["drawdown"].min(skipna=True) or 0.0)

    underwater = (frame["drawdown"] < 0).to_numpy()
    longest = current = 0
    for flag in underwater:
        current = current + 1 if flag else 0
        longest = max(longest, current)
    return max(0.0, dd_pct), max(0.0, dd_abs), longest


def _streaks(trades: list[Trade]) -> tuple[int, int]:
    """Longest winning and losing streaks."""
    best_win = best_loss = current_win = current_loss = 0
    for trade in trades:
        if trade.pnl > 0:
            current_win += 1
            current_loss = 0
        elif trade.pnl < 0:
            current_loss += 1
            current_win = 0
        else:
            current_win = current_loss = 0
        best_win = max(best_win, current_win)
        best_loss = max(best_loss, current_loss)
    return best_win, best_loss


def _ratios(
    equity: pd.Series, periods_per_year: float, risk_free_rate: float
) -> tuple[float, float, float]:
    """Return ``(sharpe, sortino, annualised_return_pct)``."""
    if len(equity) < 3:
        return 0.0, 0.0, 0.0

    returns = equity.pct_change().dropna()
    returns = returns[np.isfinite(returns)]
    if returns.empty:
        return 0.0, 0.0, 0.0

    periodic_rf = risk_free_rate / periods_per_year if periods_per_year > 0 else 0.0
    excess = returns - periodic_rf
    annualiser = math.sqrt(periods_per_year) if periods_per_year > 0 else 1.0

    std = float(excess.std(ddof=1))
    sharpe = safe_div(float(excess.mean()), std) * annualiser if std > 0 else 0.0

    # Sortino punishes only downside deviation; the denominator uses the full
    # sample length, not just the losing periods, which is the standard
    # definition and avoids inflating the ratio when losses are rare.
    downside = excess.clip(upper=0.0)
    downside_dev = math.sqrt(float((downside**2).mean()))
    sortino = safe_div(float(excess.mean()), downside_dev) * annualiser if downside_dev > 0 else 0.0

    start, end = float(equity.iloc[0]), float(equity.iloc[-1])
    if start > 0 and end > 0 and len(returns) > 0:
        years = len(returns) / periods_per_year if periods_per_year > 0 else 0.0
        annualised = ((end / start) ** (1.0 / years) - 1.0) * 100.0 if years > 0 else 0.0
    else:
        annualised = -100.0
    return sharpe, sortino, annualised


def compute_metrics(
    trades: list[Trade],
    equity_curve: list[tuple[datetime, float]],
    initial_equity: float,
    *,
    periods_per_year: float = 35_040.0,
    risk_free_rate: float = 0.0,
    bars_in_market: int = 0,
    total_bars: int = 0,
) -> PerformanceMetrics:
    """Compute the full metric set.

    Args:
        trades: closed trades, in chronological order.
        equity_curve: ``(timestamp, equity)`` samples, one per processed bar.
        initial_equity: starting equity.
        periods_per_year: bars per year, for annualisation. Derive it with
            :func:`utils.timeframes.bars_per_year`.
        risk_free_rate: annual risk-free rate as a decimal (``0.04`` = 4%).
        bars_in_market: bars during which at least one position was open.
        total_bars: bars processed, for the exposure figure.

    Returns:
        A populated :class:`PerformanceMetrics`, including any warnings about
        the reliability of the numbers.
    """
    frame = equity_dataframe(equity_curve)
    metrics = PerformanceMetrics(
        initial_equity=initial_equity,
        final_equity=float(frame["equity"].iloc[-1]) if not frame.empty else initial_equity,
        bars=total_bars or len(frame),
        period_start=frame.index[0].to_pydatetime() if not frame.empty else None,
        period_end=frame.index[-1].to_pydatetime() if not frame.empty else None,
    )

    metrics.net_profit = metrics.final_equity - initial_equity
    metrics.net_profit_pct = 100.0 * safe_div(metrics.net_profit, initial_equity)
    (
        metrics.max_drawdown_pct,
        metrics.max_drawdown_abs,
        metrics.max_drawdown_duration_bars,
    ) = _max_drawdown(frame)

    if not frame.empty:
        metrics.sharpe, metrics.sortino, metrics.annualised_return_pct = _ratios(
            frame["equity"], periods_per_year, risk_free_rate
        )
    metrics.calmar = safe_div(metrics.annualised_return_pct, metrics.max_drawdown_pct)
    metrics.recovery_factor = safe_div(metrics.net_profit, metrics.max_drawdown_abs)
    metrics.exposure_pct = 100.0 * safe_div(bars_in_market, total_bars)

    if not trades:
        metrics.warnings.append(
            "no trades were taken: either the gates are too strict for this data, or the "
            "warm-up consumed the whole sample"
        )
        return metrics

    pnls = np.array([t.pnl for t in trades], dtype=float)
    r_multiples = np.array([t.r_multiple for t in trades], dtype=float)
    winners = pnls[pnls > 0]
    losers = pnls[pnls < 0]

    metrics.total_trades = len(trades)
    metrics.wins = int(winners.size)
    metrics.losses = int(losers.size)
    metrics.breakeven = metrics.total_trades - metrics.wins - metrics.losses
    metrics.win_rate = 100.0 * safe_div(metrics.wins, metrics.total_trades)

    metrics.gross_profit = float(winners.sum()) if winners.size else 0.0
    metrics.gross_loss = float(-losers.sum()) if losers.size else 0.0
    metrics.total_fees = float(sum(t.fees for t in trades))
    metrics.average_win = float(winners.mean()) if winners.size else 0.0
    metrics.average_loss = float(losers.mean()) if losers.size else 0.0
    metrics.win_loss_ratio = safe_div(metrics.average_win, abs(metrics.average_loss))
    metrics.largest_win = float(winners.max()) if winners.size else 0.0
    metrics.largest_loss = float(losers.min()) if losers.size else 0.0

    # A zero gross loss makes the profit factor infinite, which is a red flag
    # (too few trades) rather than a triumph, so it is reported as inf and
    # flagged instead of being silently clipped.
    metrics.profit_factor = (
        safe_div(metrics.gross_profit, metrics.gross_loss)
        if metrics.gross_loss > 0
        else (float("inf") if metrics.gross_profit > 0 else 0.0)
    )

    metrics.expectancy = float(pnls.mean())
    finite_r = r_multiples[np.isfinite(r_multiples)]
    metrics.average_r = float(finite_r.mean()) if finite_r.size else 0.0
    metrics.expectancy_r = metrics.average_r
    winning_r = finite_r[finite_r > 0]
    metrics.average_rr_realised = float(winning_r.mean()) if winning_r.size else 0.0

    # System Quality Number: expectancy in R divided by its own standard error.
    if finite_r.size > 1:
        std_r = float(finite_r.std(ddof=1))
        metrics.sqn = safe_div(metrics.average_r, std_r) * math.sqrt(finite_r.size)

    metrics.max_consecutive_wins, metrics.max_consecutive_losses = _streaks(trades)
    metrics.average_bars_held = float(np.mean([t.bars_held for t in trades]))
    metrics.average_duration_hours = float(np.mean([t.duration_hours for t in trades]))

    for trade in trades:
        reason = trade.exit_reason.value
        metrics.by_exit_reason[reason] = metrics.by_exit_reason.get(reason, 0) + 1
        metrics.by_symbol[trade.symbol] = metrics.by_symbol.get(trade.symbol, 0.0) + trade.pnl

    # --- honesty checks ---
    if metrics.total_trades < MIN_MEANINGFUL_TRADES:
        metrics.warnings.append(
            f"only {metrics.total_trades} trades: profit factor, Sharpe and expectancy are "
            f"dominated by noise below ~{MIN_MEANINGFUL_TRADES}. Do not tune on this sample."
        )
    if metrics.gross_loss == 0 and metrics.wins > 0:
        metrics.warnings.append(
            "no losing trades at all - almost always a sign of too short a sample or a "
            "lookahead bug, not of a perfect system"
        )
    if metrics.max_drawdown_pct == 0 and metrics.total_trades > 5:
        metrics.warnings.append("zero drawdown over multiple trades is implausible; verify the run")
    if metrics.exposure_pct > 0 and metrics.exposure_pct < 1:
        metrics.warnings.append(
            f"exposure is only {metrics.exposure_pct:.2f}% of bars: the sample says very little "
            f"about how the system behaves in this market"
        )
    if metrics.expectancy_r <= 0:
        metrics.warnings.append(
            f"expectancy is {metrics.expectancy_r:+.3f}R per trade - this configuration loses "
            f"money net of costs. Do not trade it."
        )
    return metrics
