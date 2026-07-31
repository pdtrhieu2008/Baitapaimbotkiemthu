"""Minimal Telegram Bot API client (send-only) over aiohttp.

Why not ``python-telegram-bot``: this bot only ever *sends*. The Bot API is a
plain HTTPS POST, so the whole transport is the ~100 lines below — no extra
dependency, no version churn, and full control over the two behaviours that
actually matter in production:

* **Rate limiting.** Telegram allows roughly one message per second per chat and
  answers 429 with a ``retry_after``. That header is honoured exactly, rather
  than guessed at with a fixed sleep.
* **Never crash the bot.** A notification failure is logged and swallowed. A
  trading process must not die because a chat is unreachable.

If you later want *inbound* commands (``/status``, ``/pause``), that is when
``python-telegram-bot`` earns its keep — add it as an extra and keep this class
for outbound.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from typing import Any

from config.settings import TelegramConfig
from utils.logger import get_logger

__all__ = ["TelegramClient"]

_log = get_logger("telegram")
_API_BASE = "https://api.telegram.org"

#: Telegram rejects messages longer than this.
MAX_MESSAGE_LENGTH = 4096


class TelegramClient:
    """Send-only Telegram client.

    Args:
        cfg: the ``telegram`` configuration section.
    """

    def __init__(self, cfg: TelegramConfig) -> None:
        self.cfg = cfg
        self._session: Any | None = None
        self._last_sent: float = 0.0
        self._lock = asyncio.Lock()
        self.sent = 0
        self.failed = 0

    # -- lifecycle ----------------------------------------------------------
    async def _get_session(self) -> Any | None:
        if self._session is not None:
            return self._session
        try:
            import aiohttp
        except ImportError:
            _log.error(
                "aiohttp is not installed, Telegram notifications are disabled. "
                "Install it with: pip install aiohttp"
            )
            return None
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.cfg.timeout_seconds)
        )
        return self._session

    async def close(self) -> None:
        """Close the HTTP session."""
        if self._session is not None:
            await self._session.close()
            self._session = None

    async def __aenter__(self) -> TelegramClient:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.close()

    # -- sending ------------------------------------------------------------
    async def _throttle(self) -> None:
        """Sleep just long enough to respect the configured minimum interval."""
        elapsed = time.monotonic() - self._last_sent
        wait = self.cfg.rate_limit_seconds - elapsed
        if wait > 0:
            await asyncio.sleep(wait)
        self._last_sent = time.monotonic()

    @staticmethod
    def _split(text: str) -> list[str]:
        """Split an over-long message on line boundaries.

        Truncating instead would drop exactly the part operators care about — the
        reasons and the risk numbers sit at the bottom of a signal message.
        """
        if len(text) <= MAX_MESSAGE_LENGTH:
            return [text]
        chunks: list[str] = []
        current: list[str] = []
        size = 0
        for line in text.splitlines(keepends=True):
            if size + len(line) > MAX_MESSAGE_LENGTH - 32 and current:
                chunks.append("".join(current))
                current, size = [], 0
            current.append(line)
            size += len(line)
        if current:
            chunks.append("".join(current))
        return chunks

    async def send_message(self, text: str, *, disable_notification: bool = False) -> bool:
        """Send a message to the configured chat.

        Args:
            text: message body, in the configured parse mode.
            disable_notification: deliver silently.

        Returns:
            ``True`` when every chunk was accepted. Never raises.
        """
        if not self.cfg.is_configured:
            return False
        session = await self._get_session()
        if session is None:
            return False

        url = f"{_API_BASE}/bot{self.cfg.bot_token}/sendMessage"
        ok = True
        async with self._lock:  # one in-flight message at a time, per chat
            for chunk in self._split(text):
                if not await self._post(session, url, chunk, disable_notification):
                    ok = False
        return ok

    async def _post(
        self, session: Any, url: str, text: str, silent: bool
    ) -> bool:
        """POST one chunk, retrying on 429 and transient 5xx."""
        payload = {
            "chat_id": self.cfg.chat_id,
            "text": text,
            "parse_mode": self.cfg.parse_mode,
            "disable_web_page_preview": True,
            "disable_notification": silent,
        }

        for attempt in range(1, self.cfg.max_retries + 1):
            await self._throttle()
            try:
                async with session.post(url, json=payload) as response:
                    if response.status == 200:
                        self.sent += 1
                        return True

                    body = await response.text()
                    if response.status == 429:
                        # Telegram tells us exactly how long to wait; use it.
                        retry_after = self.cfg.rate_limit_seconds
                        with contextlib.suppress(Exception):  # malformed body
                            retry_after = float(
                                (await response.json(content_type=None))
                                .get("parameters", {})
                                .get("retry_after", retry_after)
                            )
                        _log.warning("telegram rate limited, waiting %.1fs", retry_after)
                        await asyncio.sleep(retry_after + 0.25)
                        continue

                    if response.status in (400, 401, 403, 404):
                        # Permanent: bad token, wrong chat id, bot blocked, or
                        # malformed markup. Retrying cannot help, so say what to
                        # fix and stop.
                        self.failed += 1
                        _log.error(
                            "telegram rejected the message (HTTP %d): %s. Check "
                            "TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID, and that you have sent the "
                            "bot at least one message so it may reply.",
                            response.status, body[:300],
                        )
                        return False

                    _log.warning(
                        "telegram HTTP %d (attempt %d/%d): %s",
                        response.status, attempt, self.cfg.max_retries, body[:200],
                    )
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - network family is wide
                _log.warning(
                    "telegram send failed (attempt %d/%d): %s: %s",
                    attempt, self.cfg.max_retries, type(exc).__name__, exc,
                )
            await asyncio.sleep(min(2.0 * attempt, 10.0))

        self.failed += 1
        _log.error("telegram: giving up after %d attempts", self.cfg.max_retries)
        return False

    async def test_connection(self) -> tuple[bool, str]:
        """Verify credentials with ``getMe``.

        Returns:
            ``(ok, detail)`` — used by the ``telegram-test`` CLI command so a
            misconfiguration is found at setup time, not on the first signal.
        """
        if not self.cfg.is_configured:
            return False, "telegram is disabled or the token/chat id is empty"
        session = await self._get_session()
        if session is None:
            return False, "aiohttp is not installed"

        try:
            async with session.get(f"{_API_BASE}/bot{self.cfg.bot_token}/getMe") as response:
                payload = await response.json(content_type=None)
                if response.status != 200 or not payload.get("ok"):
                    return False, f"HTTP {response.status}: {payload}"
                username = payload["result"].get("username", "?")
                return True, f"authenticated as @{username}"
        except Exception as exc:  # noqa: BLE001
            return False, f"{type(exc).__name__}: {exc}"
