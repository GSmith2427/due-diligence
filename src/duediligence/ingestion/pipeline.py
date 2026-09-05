"""Ingestion pipeline: filing → chunks → embeddings → Qdrant.

This is the orchestration layer that ties the chunker, embedder, and vector
store together. Kept deliberately thin — each constituent is doing the
non-trivial work; this module is mostly plumbing and observability.

Idempotency
-----------
Chunk ids are deterministic SHA-256 hashes derived from
``(accession_number, chunk_index, chunk_text)``. Ingesting the same filing
twice produces the same ids, and Qdrant's upsert overwrites rather than
duplicates. Re-running ingest is therefore safe and the right way to refresh
parsing logic against an already-fetched filing.

Batching
--------
Embedding calls go out in batches of ``EMBED_BATCH_SIZE`` chunks. A typical
10-K produces 100-300 chunks; batching keeps the round-trip overhead small
without blowing through Ollama's context limits.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Final

from duediligence.clients import (
    OllamaClientProtocol,
    QdrantClientProtocol,
    VectorRecord,
)
from duediligence.ingestion.chunker import Chunker
from duediligence.ingestion.models import Chunk
from duediligence.logging import get_logger
from duediligence.sources.edgar.models import Filing

log = get_logger(__name__)

EMBED_BATCH_SIZE: Final[int] = 32


@dataclass(frozen=True)
class IngestResult:
    """Summary of a single ingest run."""

    ticker: str
    accession_number: str
    chunks_indexed: int


class IngestionPipeline:
    """Coordinates chunking, embedding, and indexing."""

    def __init__(
        self,
        *,
        chunker: Chunker,
        ollama: OllamaClientProtocol,
        qdrant: QdrantClientProtocol,
    ) -> None:
        self._chunker = chunker
        self._ollama = ollama
        self._qdrant = qdrant

    async def ingest(self, filing: Filing) -> IngestResult:
        """Chunk, embed, and index a filing.

        The Qdrant collection must already exist; the caller is responsible
        for calling ``QdrantClient.ensure_collection()`` before the first
        ingest (typically once at application start).
        """
        bound_log = log.bind(
            ticker=filing.company.ticker,
            accession=filing.metadata.accession_number,
        )
        bound_log.info("ingest_started", chars=len(filing.text))

        chunks = self._chunker.chunk(filing.text)
        if not chunks:
            bound_log.warning("ingest_no_chunks")
            return IngestResult(
                ticker=filing.company.ticker,
                accession_number=filing.metadata.accession_number,
                chunks_indexed=0,
            )

        bound_log.info("ingest_chunked", chunk_count=len(chunks))

        records = await self._embed_in_batches(filing, chunks, bound_log)
        await self._qdrant.upsert(records)

        bound_log.info("ingest_completed", chunks_indexed=len(records))
        return IngestResult(
            ticker=filing.company.ticker,
            accession_number=filing.metadata.accession_number,
            chunks_indexed=len(records),
        )

    async def _embed_in_batches(
        self,
        filing: Filing,
        chunks: list[Chunk],
        bound_log: object,  # structlog's BoundLogger
    ) -> list[VectorRecord]:
        records: list[VectorRecord] = []
        for batch_start in range(0, len(chunks), EMBED_BATCH_SIZE):
            batch = chunks[batch_start : batch_start + EMBED_BATCH_SIZE]
            vectors = await self._ollama.embed([c.text for c in batch])
            for chunk, vector in zip(batch, vectors, strict=True):
                records.append(_to_record(filing, chunk, vector))
        return records


def _to_record(filing: Filing, chunk: Chunk, vector: list[float]) -> VectorRecord:
    """Construct the Qdrant record for one chunk of one filing.

    The payload contains everything a citation needs: how to find the
    document, where in it the chunk lives, and the chunk's text for
    inline display.
    """
    return VectorRecord(
        id=_chunk_id(filing.metadata.accession_number, chunk),
        vector=vector,
        payload={
            "ticker": filing.company.ticker,
            "cik": filing.company.cik,
            "company_name": filing.company.name,
            "accession_number": filing.metadata.accession_number,
            "form": filing.metadata.form,
            "filing_date": filing.metadata.filing_date.isoformat(),
            "source_url": str(filing.provenance.url),
            "chunk_index": chunk.index,
            "char_start": chunk.char_start,
            "char_end": chunk.char_end,
            "text": chunk.text,
        },
    )


def _chunk_id(accession_number: str, chunk: Chunk) -> str:
    """Deterministic id for a chunk within a filing.

    Includes the chunk text so a re-chunk with different parameters produces
    new ids (rather than silently overwriting old data with new offsets).
    """
    digest = hashlib.sha256()
    digest.update(accession_number.encode("utf-8"))
    digest.update(f":{chunk.index}:".encode())
    digest.update(chunk.text.encode("utf-8"))
    return digest.hexdigest()
