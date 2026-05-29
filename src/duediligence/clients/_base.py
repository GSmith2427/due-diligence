"""Shared infrastructure for HTTP clients.

This module owns:

- Domain exceptions raised by clients (one per upstream service).
- A shared retry policy applied to transient network errors.

Concrete clients live in sibling modules (``ollama.py``, ``qdrant.py``) and
should depend only on the types defined here, never on ``httpx`` exception
classes directly. This keeps the boundary between infrastructure and the rest
of the codebase clean: application code never imports ``httpx``.
"""

from __future__ import annotations

import httpx
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)


class ClientError(Exception):
    """Base class for all client errors raised by this package."""


class OllamaError(ClientError):
    """An Ollama operation failed.

    Wraps the underlying cause so callers can introspect it without depending
    on ``httpx``.
    """


class QdrantError(ClientError):
    """A Qdrant operation failed."""


# Errors that warrant a retry — broadly, "the network or the upstream had a
# bad moment." Application-level errors (4xx responses, validation failures)
# are not retried because retrying won't change the outcome.
_TRANSIENT_HTTPX_ERRORS = (
    httpx.ConnectError,
    httpx.ConnectTimeout,
    httpx.ReadTimeout,
    httpx.WriteTimeout,
    httpx.PoolTimeout,
    httpx.RemoteProtocolError,
)


def transient_network_retry(*, max_attempts: int = 3) -> AsyncRetrying:
    """Return a ``tenacity`` retry policy for transient network failures.

    Uses exponential backoff with jitter to avoid thundering-herd retries when
    a service has just come back online. Default of three attempts means the
    caller waits at most a few seconds in the worst case before getting a
    definitive failure.
    """
    return AsyncRetrying(
        retry=retry_if_exception_type(_TRANSIENT_HTTPX_ERRORS),
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential_jitter(initial=0.5, max=4.0),
        reraise=True,
    )
