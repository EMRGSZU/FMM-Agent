from __future__ import annotations

import logging
import os
import threading
import uuid
from collections.abc import Callable
from pathlib import Path, PurePath

from rich.logging import RichHandler
from rich.text import Text


_SET_UP_LOGGERS: set[str] = set()
_ADDITIONAL_HANDLERS: dict[str, logging.Handler] = {}
_LOG_LOCK = threading.Lock()

logging.TRACE = 5  # type: ignore[attr-defined]
logging.addLevelName(logging.TRACE, "TRACE")  # type: ignore[attr-defined]


def _interpret_level(level: int | str | None, *, default=logging.DEBUG) -> int:
    if level is None:
        return default
    if isinstance(level, int):
        return level
    if level.isnumeric():
        return int(level)
    return getattr(logging, level.upper())


_STREAM_LEVEL = _interpret_level(os.environ.get("FMM_AGENT_LOG_STREAM_LEVEL"))
_INCLUDE_LOGGER_NAME_IN_STREAM_HANDLER = False
_THREAD_NAME_TO_LOG_SUFFIX: dict[str, str] = {}


def register_thread_name(name: str) -> None:
    thread_name = threading.current_thread().name
    _THREAD_NAME_TO_LOG_SUFFIX[thread_name] = name


class _RichHandlerWithEmoji(RichHandler):
    def __init__(self, emoji: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if emoji and not emoji.endswith(" "):
            emoji += " "
        self.emoji = emoji

    def get_level_text(self, record: logging.LogRecord) -> Text:
        level_name = record.levelname.replace("WARNING", "WARN")
        return Text.styled((self.emoji + level_name).ljust(10), f"logging.level.{level_name.lower()}")


def get_logger(name: str, *, emoji: str = "") -> logging.Logger:
    """Return a project logger with the shared rich/file handler setup."""
    thread_name = threading.current_thread().name
    if thread_name != "MainThread":
        name = name + "-" + _THREAD_NAME_TO_LOG_SUFFIX.get(thread_name, thread_name)

    logger = logging.getLogger(name)
    if logger.hasHandlers():
        return logger

    handler = _RichHandlerWithEmoji(
        emoji=emoji,
        show_time=bool(os.environ.get("FMM_AGENT_LOG_SHOW_TIME", False)),
        show_path=False,
    )
    handler.setLevel(_STREAM_LEVEL)

    logger.setLevel(logging.TRACE)  # type: ignore[attr-defined]
    logger.addHandler(handler)
    logger.propagate = False
    _SET_UP_LOGGERS.add(name)

    with _LOG_LOCK:
        for extra_handler in _ADDITIONAL_HANDLERS.values():
            handler_filter = getattr(extra_handler, "my_filter", None)
            if handler_filter is None:
                logger.addHandler(extra_handler)
            elif isinstance(handler_filter, str) and handler_filter in name:
                logger.addHandler(extra_handler)
            elif callable(handler_filter) and handler_filter(name):
                logger.addHandler(extra_handler)

    if _INCLUDE_LOGGER_NAME_IN_STREAM_HANDLER:
        _add_logger_name_to_stream_handler(logger)
    return logger


def add_file_handler(
    path: PurePath | str,
    *,
    filter: str | Callable[[str], bool] | None = None,
    level: int | str = logging.TRACE,  # type: ignore[attr-defined]
    id_: str = "",
) -> str:
    """Attach one file handler to all existing and future project loggers."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s"))
    handler.setLevel(_interpret_level(level))

    with _LOG_LOCK:
        for name in _SET_UP_LOGGERS:
            if filter is not None:
                if isinstance(filter, str) and filter not in name:
                    continue
                if callable(filter) and not filter(name):
                    continue
            logging.getLogger(name).addHandler(handler)

    handler.my_filter = filter  # type: ignore[attr-defined]
    if not id_:
        id_ = str(uuid.uuid4())
    _ADDITIONAL_HANDLERS[id_] = handler
    return id_


def remove_file_handler(id_: str) -> None:
    """Remove a shared file handler by id."""
    handler = _ADDITIONAL_HANDLERS.pop(id_)
    with _LOG_LOCK:
        for log_name in _SET_UP_LOGGERS:
            logging.getLogger(log_name).removeHandler(handler)


def _add_logger_name_to_stream_handler(logger: logging.Logger) -> None:
    for handler in logger.handlers:
        if isinstance(handler, _RichHandlerWithEmoji):
            handler.setFormatter(logging.Formatter("[%(name)s] %(message)s"))


def add_logger_names_to_stream_handlers() -> None:
    """Show logger names in all configured rich stream handlers."""
    global _INCLUDE_LOGGER_NAME_IN_STREAM_HANDLER
    _INCLUDE_LOGGER_NAME_IN_STREAM_HANDLER = True
    with _LOG_LOCK:
        for logger_name in _SET_UP_LOGGERS:
            _add_logger_name_to_stream_handler(logging.getLogger(logger_name))


def set_stream_handler_levels(level: int) -> None:
    """Raise the minimum visible level for rich stream handlers."""
    global _STREAM_LEVEL
    _STREAM_LEVEL = level
    with _LOG_LOCK:
        for name in _SET_UP_LOGGERS:
            logger = logging.getLogger(name)
            for handler in logger.handlers:
                if isinstance(handler, _RichHandlerWithEmoji) and handler.level < level:
                    handler.setLevel(level)
