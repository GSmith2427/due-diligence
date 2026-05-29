"""Tests for the Ollama and Qdrant clients.

Unit tests use ``httpx.MockTransport`` to substitute fake HTTP responses, so
they run fast and offline. Integration tests are marked and require the real
services to be running locally; they are skipped in CI.
"""

from __future__ import annotations

from collections.abc import Callable

import httpx
import pytest

from duediligence.clients import (
    OllamaClient,
    OllamaError,
    QdrantClient,
    QdrantError,
    VectorRecord,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_ollama(
    handler: Callable[[httpx.Request], httpx.Response],
) -> OllamaClient:
    """Construct an OllamaClient whose HTTP transport is mocked."""
    client = OllamaClient(
        base_url="http://test.invalid",
        chat_model="test-chat",
        embedding_model="test-embed",
        timeout_seconds=5.0,
    )
    # Substitute the internal AsyncClient with one driven by MockTransport.
    client._client = httpx.AsyncClient(
        base_url="http://test.invalid",
        transport=httpx.MockTransport(handler),
    )
    return client


def _mock_qdrant(
    handler: Callable[[httpx.Request], httpx.Response],
) -> QdrantClient:
    client = QdrantClient(
        base_url="http://test.invalid",
        collection_name="test_collection",
        vector_size=4,
    )
    client._client = httpx.AsyncClient(
        base_url="http://test.invalid",
        transport=httpx.MockTransport(handler),
    )
    return client


# ---------------------------------------------------------------------------
# Ollama unit tests
# ---------------------------------------------------------------------------


async def test_ollama_health_passes_when_models_installed() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/tags"
        return httpx.Response(
            200,
            json={"models": [{"name": "test-chat"}, {"name": "test-embed"}]},
        )

    async with _mock_ollama(handler) as ollama:
        await ollama.health()  # does not raise


async def test_ollama_health_reports_missing_models() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"models": [{"name": "test-chat"}]})

    async with _mock_ollama(handler) as ollama:
        with pytest.raises(OllamaError, match="test-embed"):
            await ollama.health()


async def test_ollama_embed_returns_one_vector_per_input() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/embed"
        return httpx.Response(
            200,
            json={"embeddings": [[0.1, 0.2], [0.3, 0.4]]},
        )

    async with _mock_ollama(handler) as ollama:
        vectors = await ollama.embed(["hello", "world"])

    assert vectors == [[0.1, 0.2], [0.3, 0.4]]


async def test_ollama_embed_empty_input_short_circuits() -> None:
    """Embedding an empty list should not make an HTTP call."""

    def handler(_: httpx.Request) -> httpx.Response:  # pragma: no cover
        raise AssertionError("should not have been called")

    async with _mock_ollama(handler) as ollama:
        assert await ollama.embed([]) == []


async def test_ollama_chat_returns_message_content() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/chat"
        return httpx.Response(
            200,
            json={"message": {"role": "assistant", "content": "the answer"}},
        )

    async with _mock_ollama(handler) as ollama:
        answer = await ollama.chat("system", "user")

    assert answer == "the answer"


# ---------------------------------------------------------------------------
# Qdrant unit tests
# ---------------------------------------------------------------------------


async def test_qdrant_ensure_collection_creates_when_missing() -> None:
    seen_paths: list[tuple[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_paths.append((request.method, request.url.path))
        if request.method == "GET":
            return httpx.Response(404, json={"status": {"error": "not found"}})
        # PUT creates the collection
        return httpx.Response(200, json={"result": True, "status": "ok"})

    async with _mock_qdrant(handler) as qdrant:
        await qdrant.ensure_collection()

    assert ("GET", "/collections/test_collection") in seen_paths
    assert ("PUT", "/collections/test_collection") in seen_paths


async def test_qdrant_ensure_collection_rejects_size_mismatch() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "result": {
                    "config": {"params": {"vectors": {"size": 99, "distance": "Cosine"}}},
                },
            },
        )

    async with _mock_qdrant(handler) as qdrant:
        with pytest.raises(QdrantError, match="vector size"):
            await qdrant.ensure_collection()


async def test_qdrant_search_parses_hits() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "result": [
                    {"id": "abc", "score": 0.91, "payload": {"ticker": "AAPL"}},
                    {"id": "def", "score": 0.88, "payload": {"ticker": "MSFT"}},
                ]
            },
        )

    async with _mock_qdrant(handler) as qdrant:
        hits = await qdrant.search([0.1] * 4, limit=2)

    assert [h.id for h in hits] == ["abc", "def"]
    assert hits[0].payload["ticker"] == "AAPL"


async def test_qdrant_upsert_empty_is_no_op() -> None:
    def handler(_: httpx.Request) -> httpx.Response:  # pragma: no cover
        raise AssertionError("should not have been called")

    async with _mock_qdrant(handler) as qdrant:
        await qdrant.upsert([])


async def test_qdrant_upsert_coerces_non_uuid_ids() -> None:
    """String ids that aren't UUIDs are coerced to deterministic UUID5s."""
    import json
    import uuid

    captured_ids: list[object] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        captured_ids.append(body["points"][0]["id"])
        return httpx.Response(200, json={"result": {"status": "ok"}})

    record = VectorRecord(id="not-a-uuid", vector=[0.1, 0.2, 0.3, 0.4], payload={})
    async with _mock_qdrant(handler) as qdrant:
        await qdrant.upsert([record])

    sent_id = captured_ids[0]
    assert isinstance(sent_id, str)
    uuid.UUID(sent_id)  # raises if not a valid UUID


# ---------------------------------------------------------------------------
# Integration tests — require the real services
# ---------------------------------------------------------------------------


@pytest.mark.integration
async def test_qdrant_real_roundtrip() -> None:
    """Live test: create collection, upsert a vector, search, get it back."""
    from duediligence.config import get_settings

    get_settings.cache_clear()
    settings = get_settings()
    qdrant = QdrantClient(
        base_url=str(settings.qdrant.host),
        collection_name="dd_integration_test",
        vector_size=4,
    )
    async with qdrant:
        await qdrant.health()
        await qdrant.ensure_collection()
        await qdrant.upsert(
            [
                VectorRecord(
                    id="test-1",
                    vector=[0.1, 0.2, 0.3, 0.4],
                    payload={"ticker": "TEST"},
                )
            ]
        )
        hits = await qdrant.search([0.1, 0.2, 0.3, 0.4], limit=1)
        assert hits, "search returned no hits"
        assert hits[0].payload["ticker"] == "TEST"


@pytest.mark.integration
async def test_ollama_real_health() -> None:
    """Live test: confirm Ollama is up and the configured models are installed."""
    from duediligence.config import get_settings

    get_settings.cache_clear()
    settings = get_settings()
    async with OllamaClient.from_settings(settings.ollama) as ollama:
        await ollama.health()
