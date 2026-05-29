"""Structured logging configuration.

Uses ``structlog`` for typed, structured logs with environment-aware rendering:

- **Development**: human-readable, colourised output via ``rich``.
- **Test / Production**: JSON-formatted, one event per line, suitable for
  log aggregators and grep.

Set up once at application start via :func:`configure_logging`. After that,
obtain a logger anywhere with::

    from duediligence.logging import get_logger

    log = get_logger(__name__)
    log.info("fetched_filing", ticker="AAPL", form="10-K", size_bytes=42_119)

Bind contextual fields once per request/operation and they propagate through
every subsequent log call from that logger::

    log = log.bind(request_id=req_id, ticker=ticker)
    log.info("ingest_started")
    # ... later ...
    log.info("ingest_completed", duration_ms=123)
"""

from __future__ import annotations

import logging
import sys
from typing import Any

import structlog
from structlog.typing import Processor

from duediligence.config import Settings, get_settings


def configure_logging(settings: Settings | None = None) -> None:
    """Configure structlog and the stdlib logging module.

    Idempotent — safe to call multiple times (e.g. from tests). The stdlib
    ``logging`` module is configured to delegate to structlog so that logs
    emitted by libraries (httpx, uvicorn, etc.) end up in the same stream.
    """
    settings = settings or get_settings()
    level = logging.getLevelName(settings.log_level)

    # Stdlib logging — minimal config; structlog handles the formatting.
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=level,
    )

    shared_processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
    ]

    if settings.environment == "development":
        # Pretty console output. Tracebacks are rendered with rich.
        renderer: Processor = structlog.dev.ConsoleRenderer(
            colors=True,
            exception_formatter=structlog.dev.rich_traceback,
        )
    else:
        # JSON, one line per event. Add exception info as a structured field.
        shared_processors.append(structlog.processors.format_exc_info)
        renderer = structlog.processors.JSONRenderer()

    structlog.configure(
        processors=[*shared_processors, renderer],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str | None = None, **initial_values: Any) -> structlog.stdlib.BoundLogger:
    """Return a structured logger bound with any supplied initial fields.

    Parameters
    ----------
    name
        Logger name, typically ``__name__`` of the calling module.
    **initial_values
        Optional fields to pre-bind on the returned logger.
    """
    logger = structlog.get_logger(name)
    if initial_values:
        logger = logger.bind(**initial_values)
    return logger  # type: ignore[no-any-return]
