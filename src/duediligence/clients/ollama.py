"""Async client for the Ollama HTTP API.

Only exposes the operations the application actually needs: text embedding,
chat completion, and a health check. The client is an async context manager
to ensure the underlying HTTP connection pool is closed cleanly.

Typical use::

    async with OllamaClient.from_settings(settings) as ollama:
        await ollama.health()
        vectors = await ollama.embed(["text one", "text two"])
        answer = await ollama.chat("You are helpful.", "What is XBRL?")
"""

from __future__ import annotations

from collections.abc import Sequence
from types import TracebackType
from typing import Protocol, Self

import httpx

from duediligence.clients._base import (
    OllamaError,
    transient_network_retry,
)
from duediligence.config import OllamaSettings
from duediligence.logging import get_logger

log = get_logger(__name__)


class OllamaClientProtocol(Protocol):
    """Surface the application depends on. See :class:`OllamaClient`."""

    async def health(self) -> None: ...
    async def embed(self, texts: Sequence[str]) -> list[list[float]]: ...
    async def chat(self, system: str, user: str) -> str: ...


class OllamaClient:
    """Concrete Ollama HTTP client.

    Use :meth:`from_settings` to construct one from application settings, and
    use the instance as an async context manager so the HTTP pool is closed
    on exit.
    """

    def __init__(
        self,
        *,
        base_url: str,
        chat_model: str,
        embedding_model: str,
        timeout_seconds: float,
    ) -> None:
        self._chat_model = chat_model
        self._embedding_model = embedding_model
        self._client = httpx.AsyncClient(
            base_url=base_url,
            timeout=timeout_seconds,
        )

    @classmethod
    def from_settings(cls, settings: OllamaSettings) -> Self:
        return cls(
            base_url=str(settings.host),
            chat_model=settings.chat_model,
            embedding_model=settings.embedding_model,
            timeout_seconds=settings.request_timeout_seconds,
        )

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self._client.aclose()

    # ----- Public operations ------------------------------------------------

    async def health(self) -> None:
        """Verify Ollama is reachable and the configured models are pulled.

        Raises :class:`OllamaError` if the server is unreachable or either of
        the configured models is missing locally.
        """
        try:
            async for attempt in transient_network_retry():
                with attempt:
                    response = await self._client.get("/api/tags")
                    response.raise_for_status()
        except httpx.HTTPError as e:
            raise OllamaError(f"ollama unreachable at {self._client.base_url}") from e

        installed = {m["name"] for m in response.json().get("models", [])}
        missing = {self._chat_model, self._embedding_model} - installed
        if missing:
            raise OllamaError(
                f"required model(s) not pulled in ollama: {sorted(missing)}. "
                f"run `ollama pull <model>` to install."
            )

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed a batch of texts into dense vectors.

        Ollama's ``/api/embed`` endpoint accepts a list of inputs and returns
        one vector per input in the same order.
        """
        if not texts:
            return []

        payload = {"model": self._embedding_model, "input": list(texts)}
        try:
            async for attempt in transient_network_retry():
                with attempt:
                    response = await self._client.post("/api/embed", json=payload)
                    response.raise_for_status()
        except httpx.HTTPError as e:
            raise OllamaError(f"embedding request failed: {e}") from e

        embeddings = response.json().get("embeddings")
        if not isinstance(embeddings, list) or len(embeddings) != len(texts):
            raise OllamaError(
                f"unexpected embedding response shape: got "
                f"{len(embeddings) if isinstance(embeddings, list) else type(embeddings).__name__} "
                f"vectors for {len(texts)} inputs"
            )
        return embeddings

    async def chat(self, system: str, user: str) -> str:
        """Complete a single-turn chat with the configured model.

        Returns the assistant's full response as a string. Streaming will be
        added when the FastAPI layer needs it.
        """
        payload = {
            "model": self._chat_model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
        }
        try:
            async for attempt in transient_network_retry():
                with attempt:
                    response = await self._client.post("/api/chat", json=payload)
                    response.raise_for_status()
        except httpx.HTTPError as e:
            raise OllamaError(f"chat request failed: {e}") from e

        message = response.json().get("message", {})
        content = message.get("content")
        if not isinstance(content, str):
            raise OllamaError(f"unexpected chat response shape: {response.json()}")
        return content
