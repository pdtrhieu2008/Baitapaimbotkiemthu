"""Notification layer.

Named ``notify`` rather than ``telegram``: a top-level package called
``telegram`` would shadow the PyPI ``telegram`` module, so
``import telegram`` inside any dependency would resolve to this project instead
and fail. The specification's ``telegram/`` directory is honoured in spirit —
:mod:`notify.telegram_client` is the Telegram transport — without booby-trapping
the import path.
"""

from notify import formatter
from notify.notifier import Notifier
from notify.telegram_client import TelegramClient

__all__ = ["Notifier", "TelegramClient", "formatter"]
