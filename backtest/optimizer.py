"""Parameter search with walk-forward validation.

The single most valuable thing an optimiser can do for a trading system is
**refuse to report the in-sample number**. A grid search over eight parameters
will always find a combination that looks excellent on the data it was fitted
to; that combination is usually worthless out of sample.

So every candidate here is fitted on the first ``train_fraction`` of the data and
**scored on the remainder**, which it never saw. The reported best is the best
out-of-sample result, and the in-sample figure is kept alongside it purely so the
gap between them can be inspected — a large gap is the signature of curve
fitting.

Optuna is optional. Grid search covers the default configuration and needs
nothing beyond the standard library.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from backtest.engine import Backtester, BacktestResult
from config.settings import ConfigError, Settings
from strategies import build_strategy
from utils.logger import get_logger

__all__ = ["OptimisationReport", "Optimiser", "TrialResult"]

_log = get_logger("backtest.optimizer")


@dataclass(slots=True)
class TrialResult:
    """One parameter combination, evaluated in and out of sample."""

    params: dict[str, Any]
    train_score: float
    test_score: float
    train_trades: int
    test_trades: int
    test_metrics: dict[str, object] = field(default_factory=dict)
    rejected: str = ""

    @property
    def accepted(self) -> bool:
        return not self.rejected

    @property
    def overfit_gap(self) -> float:
        """In-sample minus out-of-sample score.

        A large positive gap means the parameters describe the training data
        rather than the market.
        """
        return self.train_score - self.test_score

    def to_dict(self) -> dict[str, object]:
        return {
            "params": self.params,
            "train_score": round(self.train_score, 4),
            "test_score": round(self.test_score, 4),
            "overfit_gap": round(self.overfit_gap, 4),
            "train_trades": self.train_trades,
            "test_trades": self.test_trades,
            "rejected": self.rejected,
            "test_metrics": self.test_metrics,
        }


@dataclass(slots=True)
class OptimisationReport:
    """Outcome of a search."""

    objective: str
    method: str
    trials: list[TrialResult]
    best: TrialResult | None = None
    total_candidates: int = 0

    @property
    def accepted(self) -> list[TrialResult]:
        return [t for t in self.trials if t.accepted]

    def summary(self) -> str:
        """Human-readable summary, honest about what was and was not learnt."""
        lines = [
            "=" * 72,
            f"  OPTIMISATION  method={self.method}  objective={self.objective}",
            f"  {len(self.accepted)}/{self.total_candidates} candidates met the minimum-trade bar",
            "=" * 72,
        ]
        if self.best is None:
            lines.append(
                "\n  No candidate produced enough trades to be evaluated.\n"
                "  Either widen the parameter grid, lower optimization.min_trades, or - more\n"
                "  likely - fetch more history."
            )
            lines.append("=" * 72)
            return "\n".join(lines)

        best = self.best
        lines.append("\n-- Best out-of-sample candidate " + "-" * 40)
        for key, value in best.params.items():
            lines.append(f"  {key:<38} {value}")
        lines.append("")
        lines.append(f"  out-of-sample {self.objective:<24} {best.test_score:+.4f}")
        lines.append(f"  in-sample     {self.objective:<24} {best.train_score:+.4f}")
        lines.append(f"  overfit gap                            {best.overfit_gap:+.4f}")
        lines.append(f"  trades (train / test)                  {best.train_trades} / {best.test_trades}")

        if best.overfit_gap > abs(best.test_score) and best.test_score > 0:
            lines.append(
                "\n  ! The in-sample score is far above the out-of-sample score. These\n"
                "    parameters largely describe the training window, not the market."
            )
        if best.test_score <= 0:
            lines.append(
                "\n  ! The best candidate is still unprofitable out of sample. The edge is\n"
                "    not in these parameters - do not deploy this."
            )

        ranked = sorted(self.accepted, key=lambda t: t.test_score, reverse=True)[:5]
        if len(ranked) > 1:
            lines.append("\n-- Top 5 by out-of-sample score " + "-" * 40)
            for index, trial in enumerate(ranked, 1):
                changes = ", ".join(f"{k.split('.')[-1]}={v}" for k, v in trial.params.items())
                lines.append(
                    f"  {index}. {trial.test_score:+.4f} (train {trial.train_score:+.4f}, "
                    f"{trial.test_trades} trades)  {changes}"
                )
        lines.append(
            "\n  Reminder: a good out-of-sample score is a reason to paper-trade, not a\n"
            "  reason to trust. Re-run on a different period before committing."
        )
        lines.append("=" * 72)
        return "\n".join(lines)

    def to_dict(self) -> dict[str, object]:
        return {
            "objective": self.objective,
            "method": self.method,
            "total_candidates": self.total_candidates,
            "accepted": len(self.accepted),
            "best": self.best.to_dict() if self.best else None,
            "trials": [t.to_dict() for t in self.trials],
        }


class Optimiser:
    """Search the configured parameter space with a walk-forward split.

    Args:
        settings: base configuration; each trial is a copy with overrides.
        frames: raw OHLCV per timeframe, as passed to the backtester.
        symbol: symbol label.
    """

    def __init__(
        self, settings: Settings, frames: dict[str, pd.DataFrame], symbol: str | None = None
    ) -> None:
        self.settings = settings
        self.frames = frames
        self.symbol = symbol or settings.backtest.symbol
        self.cfg = settings.optimization
        self._train_frames, self._test_frames = self._split(frames)

    # -- data splitting -----------------------------------------------------
    def _split(
        self, frames: dict[str, pd.DataFrame]
    ) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
        """Split every timeframe at the same wall-clock instant.

        Splitting each timeframe by its own row count would put the boundary at
        a different date on each one, letting a higher timeframe see past the
        cut. The split is therefore chosen on the primary timeframe and applied
        as a timestamp everywhere.
        """
        primary_tf = self.settings.backtest.timeframe
        primary = frames.get(primary_tf)
        if primary is None or primary.empty:
            raise ConfigError(f"cannot split: no data for the primary timeframe {primary_tf!r}")

        cut_position = int(len(primary) * self.cfg.train_fraction)
        cut_position = max(1, min(cut_position, len(primary) - 1))
        boundary = primary.index[cut_position]

        train = {tf: frame[frame.index < boundary] for tf, frame in frames.items()}
        test = dict(frames)  # full history...
        # ...but the test *run* is restricted to bars after the boundary via
        # backtest.start, so the out-of-sample walk still has its indicator
        # warm-up available. Trimming the frames instead would force every
        # indicator to re-seed inside the test window and measure warm-up
        # artefacts rather than the strategy.
        self._test_start = boundary.isoformat()
        _log.info(
            "walk-forward split at %s: %d train bars, %d test bars of %s",
            boundary, cut_position, len(primary) - cut_position, primary_tf,
        )
        return train, test

    # -- candidate generation -----------------------------------------------
    def _grid(self) -> Iterator[dict[str, Any]]:
        """Full Cartesian product of ``optimization.param_grid``."""
        grid = self.cfg.param_grid
        if not grid:
            raise ConfigError("optimization.param_grid is empty; nothing to search")
        keys = list(grid)
        for combination in itertools.product(*(grid[key] for key in keys)):
            yield dict(zip(keys, combination, strict=True))

    @property
    def grid_size(self) -> int:
        total = 1
        for values in self.cfg.param_grid.values():
            total *= max(1, len(values))
        return total

    # -- evaluation ---------------------------------------------------------
    def _score(self, result: BacktestResult) -> float:
        value = result.objective_values.get(self.cfg.objective)
        if value is None or not math.isfinite(value):
            return float("-inf")
        return float(value)

    def _evaluate(self, params: dict[str, Any]) -> TrialResult:
        """Fit on the training window, score on the untouched remainder."""
        try:
            train_settings = self.settings.with_overrides({**params, "backtest.start": None})
        except ConfigError as exc:
            return TrialResult(params, float("-inf"), float("-inf"), 0, 0, rejected=str(exc))

        try:
            train_result = Backtester(train_settings, build_strategy(train_settings)).run(
                self._train_frames, symbol=self.symbol
            )
        except (ValueError, KeyError) as exc:
            return TrialResult(
                params, float("-inf"), float("-inf"), 0, 0, rejected=f"train run failed: {exc}"
            )

        if train_result.metrics.total_trades < self.cfg.min_trades:
            return TrialResult(
                params,
                self._score(train_result),
                float("-inf"),
                train_result.metrics.total_trades,
                0,
                rejected=(
                    f"only {train_result.metrics.total_trades} in-sample trades, "
                    f"min_trades={self.cfg.min_trades}"
                ),
            )

        test_settings = self.settings.with_overrides(
            {**params, "backtest.start": self._test_start}
        )
        try:
            test_result = Backtester(test_settings, build_strategy(test_settings)).run(
                self._test_frames, symbol=self.symbol
            )
        except (ValueError, KeyError) as exc:
            return TrialResult(
                params,
                self._score(train_result),
                float("-inf"),
                train_result.metrics.total_trades,
                0,
                rejected=f"test run failed: {exc}",
            )

        return TrialResult(
            params=params,
            train_score=self._score(train_result),
            test_score=self._score(test_result),
            train_trades=train_result.metrics.total_trades,
            test_trades=test_result.metrics.total_trades,
            test_metrics=test_result.metrics.to_dict(),
        )

    # -- public API ---------------------------------------------------------
    def run(self, *, max_candidates: int | None = None) -> OptimisationReport:
        """Execute the search configured in ``optimization.method``.

        Args:
            max_candidates: hard cap on evaluations, overriding the grid size.

        Returns:
            An :class:`OptimisationReport`.
        """
        if self.cfg.method == "optuna":
            return self._run_optuna()
        return self._run_grid(max_candidates)

    def _run_grid(self, max_candidates: int | None) -> OptimisationReport:
        candidates = list(self._grid())
        total = len(candidates)
        limit = max_candidates or total
        if total > limit:
            _log.warning(
                "grid has %d combinations, evaluating the first %d. Use method=optuna for a "
                "smarter search of a space this size.",
                total, limit,
            )
            candidates = candidates[:limit]

        trials: list[TrialResult] = []
        for index, params in enumerate(candidates, 1):
            trial = self._evaluate(params)
            trials.append(trial)
            status = trial.rejected or f"test={trial.test_score:+.4f}"
            _log.info("[%d/%d] %s -> %s", index, len(candidates), params, status)

        return self._report("grid", trials, total)

    def _run_optuna(self) -> OptimisationReport:
        """Bayesian search with Optuna, falling back to grid if it is missing."""
        try:
            import optuna
        except ImportError:
            _log.warning(
                "optuna is not installed; falling back to grid search. "
                "Install it with: pip install optuna"
            )
            return self._run_grid(None)

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        grid = self.cfg.param_grid
        if not grid:
            raise ConfigError("optimization.param_grid is empty; nothing to search")
        trials: list[TrialResult] = []

        def objective(trial: optuna.Trial) -> float:
            params = {
                key: trial.suggest_categorical(key, list(values)) for key, values in grid.items()
            }
            outcome = self._evaluate(params)
            trials.append(outcome)
            if not outcome.accepted:
                # Pruning tells Optuna this region is uninformative rather than
                # merely bad, so it does not chase the -inf.
                raise optuna.TrialPruned(outcome.rejected)
            return outcome.test_score

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=self.cfg.n_trials, catch=(ValueError,))
        return self._report("optuna", trials, self.grid_size)

    def _report(
        self, method: str, trials: list[TrialResult], total: int
    ) -> OptimisationReport:
        accepted = [t for t in trials if t.accepted and math.isfinite(t.test_score)]
        best = max(accepted, key=lambda t: t.test_score) if accepted else None
        return OptimisationReport(
            objective=self.cfg.objective,
            method=method,
            trials=trials,
            best=best,
            total_candidates=total,
        )
