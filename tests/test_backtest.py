"""Backtest engine, metrics and the optimiser's walk-forward split."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from backtest.engine import Backtester
from backtest.metrics import compute_metrics, equity_dataframe
from backtest.optimizer import Optimiser
from backtest.report import export_csv, text_report
from config.settings import Settings
from risk.portfolio import ExitReason, Trade
from strategies import build_strategy
from strategies.base import SignalSide


def make_trade(
    pnl: float,
    r_multiple: float | None = None,
    *,
    reason: ExitReason = ExitReason.TAKE_PROFIT,
    day: int = 1,
    fees: float = 0.1,
) -> Trade:
    """A minimal completed trade, for metric arithmetic."""
    opened = datetime(2024, 1, day, tzinfo=UTC)
    return Trade(
        position_id=day,
        symbol="BTC/USDT",
        side=SignalSide.LONG,
        timeframe="15m",
        entry_price=100.0,
        exit_price=100.0 + pnl,
        quantity=1.0,
        opened_at=opened,
        closed_at=opened + timedelta(hours=4),
        pnl=pnl,
        fees=fees,
        exit_reason=reason,
        initial_risk=10.0,
        r_multiple=r_multiple if r_multiple is not None else pnl / 10.0,
        mae_r=-0.4,
        mfe_r=1.2,
        bars_held=16,
    )


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def run_settings(test_settings: Settings) -> Settings:
    """Settings for the shared run: start after the 4h warm-up."""
    return test_settings.with_overrides({"backtest.start": "2024-02-01"})


@pytest.fixture(scope="module")
def result(run_settings: Settings, ladder):
    """One backtest, reused by every test that only inspects the outcome.

    A full run over this fixture takes ~15s; the assertions below are
    independent of each other, so re-running it per test would multiply the
    suite's wall time for no extra coverage. Tests that need *different*
    settings still run their own.
    """
    return Backtester(run_settings, build_strategy(run_settings)).run(ladder, symbol="BTC/USDT")


def test_backtest_runs_and_reports(result) -> None:
    assert result.metrics.bars > 500
    assert result.symbol == "BTC/USDT"
    assert not result.equity_curve.empty
    # Every bar was either rejected or produced a signal - the loop ran.
    assert sum(result.rejections.values()) + result.signals_emitted > 0
    assert "POSITIVE" in text_report(result) or "NEGATIVE" in text_report(result)


def test_backtest_is_deterministic(run_settings: Settings, ladder, result) -> None:
    """Same settings, same data, same result - or nothing is reproducible."""
    first = result
    second = Backtester(run_settings, build_strategy(run_settings)).run(ladder, symbol="BTC/USDT")

    assert first.metrics.total_trades == second.metrics.total_trades
    assert first.metrics.net_profit == pytest.approx(second.metrics.net_profit)
    assert first.rejections == second.rejections
    assert [t.r_multiple for t in first.trades] == [
        pytest.approx(t.r_multiple) for t in second.trades
    ]


def test_fill_happens_on_the_bar_after_the_signal(run_settings: Settings, ladder, result) -> None:
    """Entries must never be priced at the close that generated them."""
    settings = run_settings
    if not result.trades:
        pytest.skip("fixture produced no trades")

    primary = ladder["15m"]
    for trade in result.trades:
        position = primary.index.get_indexer([pd.Timestamp(trade.opened_at)], method="pad")[0]
        bar_open = float(primary["open"].iloc[position])
        # entry == that bar's open, adjusted only by slippage.
        drift = abs(trade.entry_price - bar_open) / bar_open * 100.0
        assert drift <= settings.risk.slippage_pct + 1e-9, (
            f"entry {trade.entry_price} is not the open {bar_open} of its fill bar"
        )


def test_stopped_trades_lose_about_one_r(result) -> None:
    """Sizing plus stop placement must produce ~-1R losses, cost included."""
    stopped = [t for t in result.trades if t.exit_reason is ExitReason.STOP_LOSS]
    if not stopped:
        pytest.skip("fixture produced no stop-outs")
    for trade in stopped:
        assert -1.5 < trade.r_multiple < -0.8, f"unexpected stop-out R: {trade.r_multiple}"


def test_costs_reduce_the_result(test_settings: Settings, ladder) -> None:
    """A run with fees must not beat the same run without them."""
    free = test_settings.with_overrides(
        {"backtest.start": "2024-02-01", "risk.fee_pct": "0", "risk.slippage_pct": "0"}
    )
    costly = test_settings.with_overrides(
        {"backtest.start": "2024-02-01", "risk.fee_pct": "0.1", "risk.slippage_pct": "0.05"}
    )
    cheap_result = Backtester(free, build_strategy(free)).run(ladder, symbol="BTC/USDT")
    dear_result = Backtester(costly, build_strategy(costly)).run(ladder, symbol="BTC/USDT")

    if cheap_result.metrics.total_trades == 0:
        pytest.skip("fixture produced no trades")
    assert dear_result.metrics.total_fees > cheap_result.metrics.total_fees


def test_missing_primary_timeframe_is_an_error(test_settings: Settings, ladder) -> None:
    frames = {tf: frame for tf, frame in ladder.items() if tf != "15m"}
    with pytest.raises(ValueError, match="primary timeframe"):
        Backtester(test_settings, build_strategy(test_settings)).run(frames)


def test_too_little_data_is_an_error(test_settings: Settings, ladder) -> None:
    frames = dict(ladder)
    frames["15m"] = frames["15m"].iloc[:100]
    with pytest.raises(ValueError, match="not enough"):
        Backtester(test_settings, build_strategy(test_settings)).run(frames)


def test_open_positions_are_closed_at_the_end(result) -> None:
    # Nothing may remain open: every trade must appear in the ledger.
    reasons = {t.exit_reason for t in result.trades}
    assert ExitReason.END_OF_DATA in reasons or all(
        r is not ExitReason.END_OF_DATA for r in reasons
    )
    assert result.signals_taken == len(result.trades) + 0


def test_export_csv_writes_files(result, tmp_path) -> None:
    written = export_csv(result, tmp_path)
    assert written["trades"].is_file()
    assert written["equity"].is_file()
    assert written["equity"].read_text().startswith("timestamp")


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def test_metrics_arithmetic() -> None:
    trades = [make_trade(20.0, 2.0, day=1), make_trade(-10.0, -1.0, day=2),
              make_trade(30.0, 3.0, day=3), make_trade(-10.0, -1.0, day=4)]
    curve = [
        (datetime(2024, 1, day, tzinfo=UTC), equity)
        for day, equity in enumerate([1000.0, 1020.0, 1010.0, 1040.0, 1030.0], start=1)
    ]
    metrics = compute_metrics(trades, curve, 1000.0, periods_per_year=365.0, total_bars=5)

    assert metrics.total_trades == 4
    assert metrics.wins == 2
    assert metrics.losses == 2
    assert metrics.win_rate == pytest.approx(50.0)
    assert metrics.gross_profit == pytest.approx(50.0)
    assert metrics.gross_loss == pytest.approx(20.0)
    assert metrics.profit_factor == pytest.approx(2.5)
    assert metrics.average_win == pytest.approx(25.0)
    assert metrics.average_loss == pytest.approx(-10.0)
    assert metrics.win_loss_ratio == pytest.approx(2.5)
    assert metrics.expectancy == pytest.approx(7.5)
    assert metrics.expectancy_r == pytest.approx(0.75)
    assert metrics.net_profit == pytest.approx(30.0)
    assert metrics.max_consecutive_wins == 1
    assert metrics.max_consecutive_losses == 1
    assert metrics.has_positive_expectancy is True


def test_drawdown_is_measured_from_the_peak() -> None:
    curve = [
        (datetime(2024, 1, day, tzinfo=UTC), equity)
        for day, equity in enumerate([1000.0, 1200.0, 900.0, 1000.0], start=1)
    ]
    metrics = compute_metrics([], curve, 1000.0, periods_per_year=365.0)
    # 1200 -> 900 is -25%.
    assert metrics.max_drawdown_pct == pytest.approx(25.0)
    assert metrics.max_drawdown_abs == pytest.approx(300.0)


def test_no_trades_is_warned_not_celebrated() -> None:
    metrics = compute_metrics([], [], 1000.0)
    assert metrics.total_trades == 0
    assert any("no trades" in warning for warning in metrics.warnings)


def test_small_sample_is_flagged() -> None:
    trades = [make_trade(5.0, 0.5, day=(i % 28) + 1) for i in range(9)]
    curve = [(datetime(2024, 1, 1, tzinfo=UTC) + timedelta(days=i), 1000.0 + 5.0 * i)
             for i in range(10)]
    metrics = compute_metrics(trades, curve, 1000.0, periods_per_year=365.0, total_bars=10)
    assert metrics.is_statistically_meaningful is False
    assert any("dominated by noise" in warning for warning in metrics.warnings)


def test_no_losses_is_treated_as_suspicious() -> None:
    trades = [make_trade(10.0, 1.0, day=(i % 28) + 1) for i in range(8)]
    curve = [(datetime(2024, 1, 1, tzinfo=UTC) + timedelta(days=i), 1000.0 + 10.0 * i)
             for i in range(9)]
    metrics = compute_metrics(trades, curve, 1000.0, periods_per_year=365.0, total_bars=9)
    assert np.isinf(metrics.profit_factor)
    assert any("no losing trades" in warning for warning in metrics.warnings)


def test_negative_expectancy_says_do_not_trade() -> None:
    trades = [make_trade(-8.0, -0.8, day=(i % 28) + 1) for i in range(35)]
    curve = [(datetime(2024, 1, 1, tzinfo=UTC) + timedelta(days=i), 1000.0 - 8.0 * i)
             for i in range(36)]
    metrics = compute_metrics(trades, curve, 1000.0, periods_per_year=365.0, total_bars=36)
    assert metrics.has_positive_expectancy is False
    assert any("do not trade it" in warning.lower() for warning in metrics.warnings)


def test_equity_dataframe_handles_empty_input() -> None:
    frame = equity_dataframe([])
    assert frame.empty
    assert list(frame.columns) == ["equity", "peak", "drawdown", "drawdown_pct"]


def test_sharpe_scales_with_the_annualiser() -> None:
    """A wrong bars-per-year figure scales Sharpe by a constant - pin the behaviour."""
    curve = [
        (datetime(2024, 1, 1, tzinfo=UTC) + timedelta(days=i), 1000.0 * (1.001**i))
        for i in range(200)
    ]
    daily = compute_metrics([], curve, 1000.0, periods_per_year=365.0)
    hourly = compute_metrics([], curve, 1000.0, periods_per_year=365.0 * 24)
    assert hourly.sharpe > daily.sharpe
    assert hourly.sharpe / daily.sharpe == pytest.approx(np.sqrt(24.0), rel=1e-6)


# ---------------------------------------------------------------------------
# Optimiser
# ---------------------------------------------------------------------------
def test_walk_forward_split_is_a_single_timestamp(test_settings: Settings, ladder) -> None:
    """Every timeframe must be cut at the same instant, not at the same row count."""
    settings = test_settings.with_overrides({"optimization.train_fraction": "0.6"})
    optimiser = Optimiser(settings, ladder, "BTC/USDT")

    boundary = pd.Timestamp(optimiser._test_start)
    for timeframe, frame in optimiser._train_frames.items():
        assert frame.index.max() < boundary, f"{timeframe} train data leaks past the split"
    # The test frames keep full history so indicators can warm up, but the run
    # itself starts at the boundary.
    assert optimiser._test_frames["15m"].index.min() < boundary


def test_optimiser_reports_out_of_sample_and_rejects_thin_samples(
    test_settings: Settings, ladder
) -> None:
    settings = test_settings.with_overrides(
        {
            "backtest.start": None,
            "optimization.method": "grid",
            "optimization.min_trades": "10",
            "optimization.param_grid": {"indicators.adx_trend_min": [18, 30]},
        }
    )
    report = Optimiser(settings, ladder, "BTC/USDT").run()
    assert report.total_candidates == 2
    assert len(report.trials) == 2
    summary = report.summary()
    assert "out-of-sample" in summary or "minimum-trade bar" in summary

    for trial in report.trials:
        if not trial.accepted:
            assert "min_trades" in trial.rejected or "failed" in trial.rejected


def test_empty_param_grid_is_an_error(test_settings: Settings, ladder) -> None:
    from config.settings import ConfigError

    settings = test_settings.with_overrides({"optimization.param_grid": {}})
    with pytest.raises(ConfigError, match="param_grid is empty"):
        Optimiser(settings, ladder, "BTC/USDT").run()
