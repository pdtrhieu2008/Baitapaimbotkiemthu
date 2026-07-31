"""Execution-quality guards: conditions under which we simply do not trade.

These are separate from the strategy's gates. A gate asks "is this a good
setup?"; a guard asks "is the *market* currently fit to trade in?". A perfect
setup on a 1% spread, in the middle of a funding-rate blow-off, is still a bad
trade — and it is one whose costs the backtest never modelled.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

from analysis.context import TimeframeContext
from analysis.regime import VolatilityState
from config.settings import RiskConfig
from data.models import ExternalContext
from utils.helpers import ensure_utc, utc_now
from utils.logger import get_logger

__all__ = ["GuardResult", "TradingGuards"]

_log = get_logger("risk.guards")


@dataclass(slots=True)
class GuardResult:
    """Outcome of the guard checks.

    Attributes:
        ok: whether trading is permitted.
        blocked_by: the first guard that objected.
        detail: explanation with the numbers involved.
        warnings: non-blocking concerns worth logging (e.g. an unmeasurable
            spread), so a degraded run is visible rather than silent.
    """

    ok: bool = True
    blocked_by: str = ""
    detail: str = ""
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "ok": self.ok,
            "blocked_by": self.blocked_by,
            "detail": self.detail,
            "warnings": self.warnings,
        }


class TradingGuards:
    """Pre-trade market-condition checks.

    Args:
        cfg: the ``risk`` configuration section.
    """

    def __init__(self, cfg: RiskConfig) -> None:
        self.cfg = cfg
        self._spread_warning_logged: set[str] = set()

    def check(
        self,
        symbol: str,
        context: TimeframeContext,
        external: ExternalContext,
        *,
        now: datetime | None = None,
    ) -> GuardResult:
        """Run every guard.

        Args:
            symbol: unified symbol.
            context: primary-timeframe context.
            external: ticker / funding / sentiment for the symbol.
            now: current time, for the trading-hours guard.

        Returns:
            A :class:`GuardResult`; check :attr:`GuardResult.ok`.
        """
        result = GuardResult()

        moment = ensure_utc(now or utc_now())
        if moment.hour in self.cfg.blocked_hours_utc:
            return GuardResult(
                False, "trading_hours", f"hour {moment.hour:02d}:00 UTC is blocked by configuration"
            )

        spread = self._spread_check(symbol, external, result)
        if spread is not None:
            return spread

        if context.regime.volatility is VolatilityState.EXTREME:
            return GuardResult(
                False,
                "volatility_anomaly",
                f"ATR is {context.regime.atr_ratio:.1f}x its median "
                f"(limit {self.cfg.volatility_anomaly_atr_ratio:.1f}x); stop distances and "
                f"slippage are unpredictable here",
            )

        if external.funding is not None and np.isfinite(external.funding.rate_pct):
            rate = external.funding.rate_pct
            if abs(rate) > self.cfg.max_funding_rate_abs:
                return GuardResult(
                    False,
                    "funding_extreme",
                    f"funding {rate:+.4f}% exceeds the ±{self.cfg.max_funding_rate_abs}% limit: "
                    f"positioning is crowded and squeeze risk is elevated",
                )

        return result

    def _spread_check(
        self, symbol: str, external: ExternalContext, result: GuardResult
    ) -> GuardResult | None:
        """Block on a wide spread; warn (once) when it cannot be measured."""
        ticker = external.ticker
        spread = ticker.spread_pct if ticker is not None else float("nan")

        if not np.isfinite(spread):
            # Unknown is not the same as zero. Trading is allowed to continue
            # because many venues omit the book from the ticker endpoint, but the
            # degradation is logged so it never passes unnoticed.
            message = "spread unmeasurable (no bid/ask in ticker); spread guard inactive"
            result.warnings.append(message)
            if symbol not in self._spread_warning_logged:
                self._spread_warning_logged.add(symbol)
                _log.warning("%s: %s", symbol, message)
            return None

        if spread > self.cfg.max_spread_pct:
            return GuardResult(
                False,
                "spread",
                f"spread {spread:.4f}% exceeds the {self.cfg.max_spread_pct}% limit",
            )
        return None
