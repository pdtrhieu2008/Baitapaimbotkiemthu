"""Durable run state.

Persisted so that a restart — a deploy, a crash, a VPS reboot — does not reset
the things that must not be reset:

* the **daily loss counter** and any active halt (otherwise restarting the
  process becomes a way to bypass the risk limits);
* the **high-water mark**, so drawdown stays measured from the real peak;
* **open paper positions** with their moved stops and taken targets;
* the last evaluated bar per symbol, so a restart mid-bar does not re-emit a
  signal that was already sent.

Writes are atomic (temp file + replace): a power cut during a write leaves the
previous good state rather than a truncated file.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from risk.manager import RiskManager, RiskState
from risk.portfolio import Portfolio
from utils.helpers import utc_now
from utils.logger import get_logger

__all__ = ["StateStore"]

_log = get_logger("live.state")

#: Bumped when the on-disk shape changes so an old file is discarded loudly
#: rather than half-read.
STATE_VERSION = 1


class StateStore:
    """Load and save the bot's durable state.

    Args:
        path: JSON file location.
        enabled: when ``False``, every method is a no-op (useful for tests and
            for a deliberately stateless run).
    """

    def __init__(self, path: str | Path, *, enabled: bool = True) -> None:
        self.path = Path(path)
        self.enabled = enabled

    def save(
        self,
        portfolio: Portfolio,
        risk: RiskManager,
        last_bars: dict[str, str],
        extra: dict[str, Any] | None = None,
    ) -> bool:
        """Write the current state atomically.

        Returns:
            ``True`` on success. Failures are logged, never raised: losing a
            state write must not take down a running bot.
        """
        if not self.enabled:
            return False
        payload = {
            "version": STATE_VERSION,
            "saved_at": utc_now().isoformat(),
            "risk": risk.state.to_dict(),
            "portfolio": portfolio.export_state(),
            "last_bars": last_bars,
            **(extra or {}),
        }
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.path.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
            temporary.replace(self.path)
            return True
        except (OSError, TypeError, ValueError) as exc:
            _log.error("cannot save state to %s: %s", self.path, exc)
            return False

    def load(self) -> dict[str, Any] | None:
        """Read the state file, or ``None`` when absent or unusable."""
        if not self.enabled or not self.path.is_file():
            return None
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            _log.error("cannot read state %s: %s; starting fresh", self.path, exc)
            return None

        version = payload.get("version")
        if version != STATE_VERSION:
            _log.warning(
                "state file %s has version %s, expected %s; ignoring it and starting fresh",
                self.path, version, STATE_VERSION,
            )
            return None
        return payload

    def restore(
        self, portfolio: Portfolio, risk: RiskManager
    ) -> tuple[dict[str, str], bool]:
        """Apply a saved state to a fresh portfolio and risk manager.

        Args:
            portfolio: ledger to populate.
            risk: risk manager whose counters to restore.

        Returns:
            ``(last_bars, restored)`` where ``restored`` says whether anything
            was loaded.
        """
        payload = self.load()
        if payload is None:
            return {}, False

        try:
            risk.state = RiskState.from_dict(payload.get("risk", {}))
        except (KeyError, TypeError, ValueError) as exc:
            _log.error("saved risk state is unusable (%s); resetting the counters", exc)

        positions = portfolio.restore_state(payload.get("portfolio", {}))
        last_bars = {str(k): str(v) for k, v in (payload.get("last_bars") or {}).items()}

        _log.info(
            "restored state from %s (saved %s): equity %.4f, %d open position(s), halt=%s",
            self.path, payload.get("saved_at"), portfolio.equity(), positions,
            risk.state.halt.value,
        )
        if risk.state.halt.value != "none":
            _log.warning(
                "the restored state carries an active halt (%s). It is intentionally NOT cleared "
                "by restarting - clear it by editing or deleting %s once you have reviewed why "
                "it fired.",
                risk.state.halt.value, self.path,
            )
        return last_bars, True

    def clear(self) -> None:
        """Delete the state file."""
        try:
            self.path.unlink(missing_ok=True)
            _log.info("state file %s removed", self.path)
        except OSError as exc:
            _log.error("cannot remove state %s: %s", self.path, exc)
