"""Strategy layer: scoring, multi-timeframe agreement, triggers, signals."""

from config.settings import ConfigError, Settings
from strategies.base import (
    Rejection,
    Signal,
    SignalSide,
    SignalStrength,
    Strategy,
    TakeProfit,
)
from strategies.confluence import ConfluenceStrategy
from strategies.mtf import MTFVerdict, MultiTimeframeAnalyser
from strategies.scoring import ScoreCard, ScoreComponent, Scorer
from strategies.triggers import Trigger, TriggerDetector

__all__ = [
    "ConfluenceStrategy",
    "MTFVerdict",
    "MultiTimeframeAnalyser",
    "Rejection",
    "ScoreCard",
    "ScoreComponent",
    "Scorer",
    "Signal",
    "SignalSide",
    "SignalStrength",
    "Strategy",
    "TakeProfit",
    "Trigger",
    "TriggerDetector",
    "build_strategy",
]

#: Registry of available strategies, keyed by ``strategy.name`` in the config.
_REGISTRY: dict[str, type[Strategy]] = {
    "confluence": ConfluenceStrategy,
}


def build_strategy(settings: Settings) -> Strategy:
    """Instantiate the strategy named in the configuration.

    Args:
        settings: validated configuration.

    Returns:
        A ready :class:`Strategy`.

    Raises:
        ConfigError: when ``strategy.name`` is not registered.
    """
    try:
        factory = _REGISTRY[settings.strategy.name]
    except KeyError as exc:
        raise ConfigError(
            f"unknown strategy.name {settings.strategy.name!r}; "
            f"available: {sorted(_REGISTRY)}"
        ) from exc
    return factory(settings)  # type: ignore[call-arg]
