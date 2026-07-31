"""Data layer: venue adapters, caching, sentiment feeds and snapshot assembly.

The rest of the project depends only on :class:`data.base.MarketDataProvider` and
the carriers in :mod:`data.models`, so adding a forex or equities venue means
writing one adapter and changing nothing else.

**Import-order note.** :mod:`analysis.context` depends on :mod:`data.models` (it
carries an :class:`~data.models.ExternalContext`), while
:class:`~data.collector.DataCollector` depends on :mod:`analysis.context`. Eagerly
importing the collector here would therefore make ``from data.models import ...``
re-enter a partially initialised :mod:`analysis.context`. The collector is instead
exposed lazily through :pep:`562` module ``__getattr__``, so ``data.DataCollector``
still works while the dependency stays one-directional at import time.
"""

from typing import TYPE_CHECKING, Any

from data.base import MarketDataProvider
from data.exchange import CCXTProvider, ExchangeUnavailable
from data.models import (
    ExternalContext,
    FundingSnapshot,
    OpenInterestSnapshot,
    SentimentSnapshot,
    Ticker,
)
from data.repository import OHLCVRepository
from data.sentiment import SentimentClient

if TYPE_CHECKING:  # for type checkers only; no runtime import
    from data.collector import DataCollector

__all__ = [
    "CCXTProvider",
    "DataCollector",
    "ExchangeUnavailable",
    "ExternalContext",
    "FundingSnapshot",
    "MarketDataProvider",
    "OHLCVRepository",
    "OpenInterestSnapshot",
    "SentimentClient",
    "SentimentSnapshot",
    "Ticker",
]

_LAZY = {"DataCollector": "data.collector"}


def __getattr__(name: str) -> Any:
    """Resolve lazily-exported names on first access (:pep:`562`)."""
    module_path = _LAZY.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    return getattr(importlib.import_module(module_path), name)
