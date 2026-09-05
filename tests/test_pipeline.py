"""Tests for the ingestion pipeline.

Uses in-memory fakes for the Ollama and Qdrant clients — see the protocols
in ``duediligence.clients`` for the contracts being implemented. This is
the practical benefit of having defined those protocols: tests don't need
mock libraries or HTTP transports, just small classes that fulfil the
interface.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Any

import pytest

from duediligence.clients import SearchHit, VectorRecord
from duediligence.ingestion import ChunkConfig, Chunker, IngestionPipeline
from duediligence.sources.edgar.models import (
    CompanyIdentity,
    Filing,
    FilingMetadata,
    Provenance,
)

# ---------------------------------------------------------------------------
# In-memory fakes for the client protocols
# ---------------------------------------------------------------------------


class FakeOllama:
    """Embeds by hashing — deterministic, fast, dimensionally consistent."""

    def __init__(self, dim: int = 8) -> None:
        self._dim = dim
        self.embed_calls: list[list[str]] = []

    async def health(self) -> None: ...

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        self.embed_calls.append(list(texts))
        return [[float((hash(t) >> i) & 1) for i in range(self._dim)] for t in texts]

    async def chat(self, system: str, user: str) -> str:
        return ""


class FakeQdrant:
    """In-memory store recording everything upserted."""

    def __init__(self) -> None:
        self.upserts: list[VectorRecord] = []
        self.ensure_called = False

    async def health(self) -> None: ...

    async def ensure_collection(self) -> None:
        self.ensure_called = True

    async def upsert(self, records: Sequence[VectorRecord]) -> None:
        self.upserts.extend(records)

    async def search(
        self,
        vector: Sequence[float],
        *,
        limit: int = 5,
        filter_payload: Mapping[str, Any] | None = None,
    ) -> list[SearchHit]:
        return []


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def filing() -> Filing:
    text = "Apple reported strong results. " * 200  # ~1200 tokens
    return Filing(
        company=CompanyIdentity(cik="0000320193", ticker="AAPL", name="Apple Inc."),
        metadata=FilingMetadata(
            accession_number="0000320193-24-000123",
            form="10-K",
            filing_date=datetime(2024, 11, 1, tzinfo=UTC),
            primary_document="aapl-20240928.htm",
        ),
        text=text,
        provenance=Provenance.for_content(
            source="sec-edgar",
            url="https://www.sec.gov/example.htm",
            content=text.encode("utf-8"),
            fetched_at=datetime(2024, 11, 2, tzinfo=UTC),
        ),
    )


@pytest.fixture
def pipeline() -> tuple[IngestionPipeline, FakeOllama, FakeQdrant]:
    ollama = FakeOllama(dim=8)
    qdrant = FakeQdrant()
    chunker = Chunker(ChunkConfig(target_tokens=100, max_tokens=150, overlap_tokens=20))
    return IngestionPipeline(chunker=chunker, ollama=ollama, qdrant=qdrant), ollama, qdrant


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


async def test_ingest_chunks_embeds_and_upserts(
    pipeline: tuple[IngestionPipeline, FakeOllama, FakeQdrant],
    filing: Filing,
) -> None:
    pipe, ollama, qdrant = pipeline

    result = await pipe.ingest(filing)

    assert result.ticker == "AAPL"
    assert result.chunks_indexed > 1
    assert len(qdrant.upserts) == result.chunks_indexed
    # Every chunk's text should have gone through the embedder
    embedded_texts = [t for batch in ollama.embed_calls for t in batch]
    assert len(embedded_texts) == result.chunks_indexed


async def test_ingest_records_carry_full_payload(
    pipeline: tuple[IngestionPipeline, FakeOllama, FakeQdrant],
    filing: Filing,
) -> None:
    pipe, _, qdrant = pipeline
    await pipe.ingest(filing)

    record = qdrant.upserts[0]
    payload = record.payload

    assert payload["ticker"] == "AAPL"
    assert payload["cik"] == "0000320193"
    assert payload["form"] == "10-K"
    assert payload["accession_number"] == "0000320193-24-000123"
    assert "text" in payload
    assert "chunk_index" in payload
    assert "char_start" in payload
    assert "char_end" in payload
    assert payload["source_url"].startswith("https://www.sec.gov/")


async def test_ingest_is_idempotent(
    pipeline: tuple[IngestionPipeline, FakeOllama, FakeQdrant],
    filing: Filing,
) -> None:
    """Re-ingesting the same filing produces the same chunk ids."""
    pipe, _, qdrant = pipeline

    await pipe.ingest(filing)
    first_ids = [r.id for r in qdrant.upserts]
    qdrant.upserts.clear()

    await pipe.ingest(filing)
    second_ids = [r.id for r in qdrant.upserts]

    assert first_ids == second_ids


async def test_ingest_batches_embedding_calls(
    pipeline: tuple[IngestionPipeline, FakeOllama, FakeQdrant],
    filing: Filing,
) -> None:
    """Embeddings should be requested in batches, not one-at-a-time."""
    pipe, ollama, _ = pipeline

    await pipe.ingest(filing)

    # Several chunks should have come through, but the number of embed calls
    # should be much smaller than the number of chunks.
    assert len(ollama.embed_calls) >= 1
    total_chunks = sum(len(batch) for batch in ollama.embed_calls)
    assert len(ollama.embed_calls) < total_chunks


async def test_ingest_handles_empty_filing(
    pipeline: tuple[IngestionPipeline, FakeOllama, FakeQdrant],
) -> None:
    pipe, ollama, qdrant = pipeline
    empty_filing = Filing(
        company=CompanyIdentity(cik="0000000001", ticker="ZZZ", name="Empty Co."),
        metadata=FilingMetadata(
            accession_number="0000000001-24-000001",
            form="10-K",
            filing_date=datetime(2024, 1, 1, tzinfo=UTC),
            primary_document="empty.htm",
        ),
        text="",
        provenance=Provenance.for_content(
            source="sec-edgar",
            url="https://www.sec.gov/empty.htm",
            content=b"",
            fetched_at=datetime(2024, 1, 1, tzinfo=UTC),
        ),
    )

    result = await pipe.ingest(empty_filing)

    assert result.chunks_indexed == 0
    assert ollama.embed_calls == []
    assert qdrant.upserts == []
