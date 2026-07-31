"""Multi-channel logging.

Five independent channels are requested by the spec — ``signal``, ``trade``,
``error``, ``telegram`` and ``performance`` — so each gets its own rotating
file plus, optionally, a machine-readable JSON-Lines sibling that a dashboard
or a pandas notebook can read directly:

    logs/quantbot.log        everything, human readable
    logs/signal.log          one line per generated / rejected signal
    logs/trade.log           one line per position open / close
    logs/error.log           WARNING+ from anywhere in the process
    logs/telegram.log        outbound notification attempts
    logs/performance.log     periodic equity / metric snapshots
    logs/*.jsonl             structured events (json_lines: true)

Records logged to a channel also propagate to the root file and the console,
so nothing is hidden in a side file.
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import sys
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any

from config.settings import LoggingConfig

__all__ = ["CHANNELS", "EventLog", "get_logger", "setup_logging"]

#: Dedicated log channels. ``quantbot.<channel>`` is the logger name.
CHANNELS: tuple[str, ...] = ("signal", "trade", "error", "telegram", "performance")

_ROOT = "quantbot"
_CONSOLE_FMT = "%(asctime)s %(levelname)-7s %(name)-22s %(message)s"
_FILE_FMT = "%(asctime)s %(levelname)-7s %(name)-28s %(funcName)s:%(lineno)d %(message)s"
_DATE_FMT = "%Y-%m-%d %H:%M:%S"

_configured = False


def _json_default(obj: Any) -> Any:
    """Serialise the domain objects that end up inside structured events."""
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, datetime):
        return obj.isoformat()
    if is_dataclass(obj) and not isinstance(obj, type):
        return asdict(obj)
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    if hasattr(obj, "item"):  # numpy scalar
        return obj.item()
    return str(obj)


def _rotating(path: Path, cfg: LoggingConfig, level: int) -> logging.Handler:
    """Create a size-rotating file handler."""
    handler = logging.handlers.RotatingFileHandler(
        path, maxBytes=cfg.max_bytes, backupCount=cfg.backup_count, encoding="utf-8"
    )
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(_FILE_FMT, datefmt=_DATE_FMT))
    return handler


def setup_logging(cfg: LoggingConfig) -> logging.Logger:
    """Install handlers for the root logger and every dedicated channel.

    Idempotent: calling it twice (e.g. from a test and from ``main``) does not
    duplicate handlers.

    Args:
        cfg: the ``logging`` section of the configuration.

    Returns:
        The ``quantbot`` root logger.
    """
    global _configured
    root = logging.getLogger(_ROOT)
    if _configured:
        return root

    level = getattr(logging, cfg.level.upper(), logging.INFO)
    log_dir = Path(cfg.dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    root.setLevel(level)
    root.propagate = False
    root.handlers.clear()

    if cfg.console:
        console = logging.StreamHandler(sys.stdout)
        console.setLevel(level)
        console.setFormatter(logging.Formatter(_CONSOLE_FMT, datefmt=_DATE_FMT))
        root.addHandler(console)

    root.addHandler(_rotating(log_dir / "quantbot.log", cfg, level))

    # error.log collects WARNING+ from every channel, so a crash post-mortem
    # never requires grepping six files.
    root.addHandler(_rotating(log_dir / "error.log", cfg, logging.WARNING))

    for channel in CHANNELS:
        if channel == "error":
            continue  # already covered by the root WARNING handler
        logger = logging.getLogger(f"{_ROOT}.{channel}")
        logger.setLevel(level)
        logger.handlers.clear()
        logger.addHandler(_rotating(log_dir / f"{channel}.log", cfg, level))
        logger.propagate = True  # also reach quantbot.log / console

    # Third-party noise: keep warnings, drop the chatter.
    for noisy in ("ccxt", "urllib3", "asyncio", "aiohttp.access", "matplotlib"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _configured = True
    root.debug("logging initialised at level %s in %s", cfg.level.upper(), log_dir.resolve())
    return root


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced child logger.

    Args:
        name: either a bare channel name (``"signal"``) or a dotted module
            path (``"strategies.confluence"``).
    """
    return logging.getLogger(f"{_ROOT}.{name}")


class EventLog:
    """Append-only JSON-Lines writer for structured events.

    Human-readable logs are for operators; these files are for analysis. Every
    record carries a UTC timestamp and an event type, so a whole run can be
    loaded with ``pandas.read_json(path, lines=True)``.

    The writer never raises: a full disk or a permissions problem must not take
    down a running bot, so failures are downgraded to a single error log line.
    """

    def __init__(self, cfg: LoggingConfig) -> None:
        self._enabled = cfg.json_lines
        self._dir = Path(cfg.dir)
        self._log = get_logger("error")
        if self._enabled:
            self._dir.mkdir(parents=True, exist_ok=True)

    def write(self, stream: str, event: str, payload: dict[str, Any]) -> None:
        """Append one record to ``logs/<stream>.jsonl``.

        Args:
            stream: file stem, conventionally one of :data:`CHANNELS`.
            event: event type, e.g. ``"signal_emitted"``, ``"trade_closed"``.
            payload: JSON-serialisable body (dataclasses and enums are handled).
        """
        if not self._enabled:
            return
        record = {
            "ts": datetime.now(UTC).isoformat(timespec="seconds"),
            "event": event,
            **payload,
        }
        try:
            line = json.dumps(record, default=_json_default, ensure_ascii=False)
            with (self._dir / f"{stream}.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
        except OSError as exc:
            self._log.error("cannot write event log %s/%s.jsonl: %s", self._dir, stream, exc)
        except (TypeError, ValueError) as exc:
            self._log.error("event %s is not serialisable: %s", event, exc)
