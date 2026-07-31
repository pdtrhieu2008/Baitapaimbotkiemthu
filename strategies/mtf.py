"""Multi-timeframe agreement.

Timeframes are not equal and they do not play the same role, so they are
assigned one:

* **context** (``1d``, ``4h``) — the macro backdrop. It may be neutral, but it
  must never *oppose* the trade. Fighting the daily is the fastest way to turn a
  good intraday setup into a loss.
* **confirm** (``1h``) — must actively agree. This is the trend the trade is
  supposed to be riding.
* **entry** (``15m``, ``5m``) — where the trigger fires. Its own bias matters
  least; it contributes to the weighted total but cannot veto on its own.

Alignment is computed as a *weighted* fraction, using ``strategy.mtf.weights``,
because "3 of 5 timeframes agree" is meaningless when the 2 that disagree are
the daily and the 4-hour.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from analysis.context import MarketSnapshot
from config.settings import MTFConfig
from strategies.base import SignalSide
from utils.helpers import safe_div

__all__ = ["MTFVerdict", "MultiTimeframeAnalyser"]


@dataclass(slots=True)
class MTFVerdict:
    """Outcome of the multi-timeframe check for one side.

    Attributes:
        side: the direction that was tested.
        agreement: weighted fraction of timeframes aligned with ``side``.
        required: the configured minimum.
        aligned / opposed / neutral: timeframe names by verdict.
        context_ok: no context timeframe opposes.
        confirm_ok: every confirm timeframe agrees.
        scores: per-timeframe bias score, for the notification.
        missing: configured timeframes absent from the snapshot.
    """

    side: SignalSide
    agreement: float
    required: float
    aligned: list[str] = field(default_factory=list)
    opposed: list[str] = field(default_factory=list)
    neutral: list[str] = field(default_factory=list)
    context_ok: bool = True
    confirm_ok: bool = True
    scores: dict[str, float] = field(default_factory=dict)
    missing: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        """Whether every multi-timeframe requirement is satisfied."""
        return (
            self.context_ok
            and self.confirm_ok
            and not self.missing
            and self.agreement >= self.required
        )

    @property
    def detail(self) -> str:
        """Human-readable explanation, used in rejections and notifications."""
        if self.missing:
            return f"missing timeframe data: {', '.join(self.missing)}"
        if not self.context_ok:
            return f"higher timeframe opposes: {', '.join(self.opposed)}"
        if not self.confirm_ok:
            return "confirmation timeframe does not agree"
        if self.agreement < self.required:
            return (
                f"agreement {self.agreement:.0%} below the required {self.required:.0%} "
                f"(aligned: {', '.join(self.aligned) or 'none'})"
            )
        return f"{self.agreement:.0%} weighted agreement ({', '.join(self.aligned)})"

    def to_dict(self) -> dict[str, object]:
        return {
            "side": self.side.name,
            "agreement": round(self.agreement, 3),
            "required": self.required,
            "aligned": self.aligned,
            "opposed": self.opposed,
            "neutral": self.neutral,
            "context_ok": self.context_ok,
            "confirm_ok": self.confirm_ok,
            "scores": {tf: round(v, 3) for tf, v in self.scores.items()},
            "passed": self.passed,
        }


class MultiTimeframeAnalyser:
    """Evaluate cross-timeframe agreement.

    Args:
        cfg: the ``strategy.mtf`` configuration section.
    """

    def __init__(self, cfg: MTFConfig) -> None:
        self.cfg = cfg

    def propose_side(self, snapshot: MarketSnapshot) -> SignalSide | None:
        """Derive the candidate direction from the higher timeframes.

        Only context and confirm timeframes vote: letting a 5-minute chart pick
        the direction is how a bot ends up counter-trending the daily.

        Args:
            snapshot: analysed market snapshot.

        Returns:
            The proposed side, or ``None`` when the higher timeframes are
            undecided (in which case no trade should be considered at all).
        """
        weighted = 0.0
        total = 0.0
        for timeframe in [*self.cfg.context, *self.cfg.confirm]:
            context = snapshot.get(timeframe)
            if context is None or not context.is_complete:
                continue
            weight = self.cfg.weight_of(timeframe)
            weighted += weight * context.bias_score
            total += weight
        if total == 0:
            return None
        average = safe_div(weighted, total)
        # A flat average means the higher timeframes disagree with each other.
        if abs(average) < 0.2:
            return None
        return SignalSide.LONG if average > 0 else SignalSide.SHORT

    def evaluate(self, snapshot: MarketSnapshot, side: SignalSide) -> MTFVerdict:
        """Test ``side`` against every configured timeframe.

        Args:
            snapshot: analysed market snapshot.
            side: direction under consideration.

        Returns:
            An :class:`MTFVerdict`; inspect :attr:`MTFVerdict.passed`.
        """
        verdict = MTFVerdict(side=side, agreement=0.0, required=self.cfg.min_agreement)

        aligned_weight = 0.0
        total_weight = 0.0

        for timeframe in self.cfg.all_timeframes:
            context = snapshot.get(timeframe)
            if context is None or not context.is_complete:
                verdict.missing.append(timeframe)
                continue

            weight = self.cfg.weight_of(timeframe)
            total_weight += weight
            verdict.scores[timeframe] = context.bias_score

            if context.bias == side.value:
                verdict.aligned.append(timeframe)
                aligned_weight += weight
            elif context.bias == 0:
                verdict.neutral.append(timeframe)
            else:
                verdict.opposed.append(timeframe)

        verdict.agreement = safe_div(aligned_weight, total_weight)
        verdict.context_ok = not any(tf in verdict.opposed for tf in self.cfg.context)
        verdict.confirm_ok = all(tf in verdict.aligned for tf in self.cfg.confirm)
        return verdict
