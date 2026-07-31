"""Risk layer: stop placement, position sizing, guards and the ledger.

Split by concern so each half stays testable:

* :mod:`risk.planner` - pure maths: where the stop and targets go.
* :mod:`risk.manager` - stateful: how big, and whether to trade at all.
* :mod:`risk.guards` - market-condition vetoes (spread, volatility, funding).
* :mod:`risk.portfolio` - positions, trades and the equity curve, shared by
  paper trading and the backtester so their accounting cannot diverge.
"""

from risk.guards import GuardResult, TradingGuards
from risk.manager import HaltReason, RiskDecision, RiskManager, RiskState
from risk.planner import TradePlan, TradePlanner
from risk.portfolio import ExitReason, Fill, Portfolio, Position, Trade

__all__ = [
    "ExitReason",
    "Fill",
    "GuardResult",
    "HaltReason",
    "Portfolio",
    "Position",
    "RiskDecision",
    "RiskManager",
    "RiskState",
    "TradePlan",
    "TradePlanner",
    "Trade",
    "TradingGuards",
]
