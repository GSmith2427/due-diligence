"""Async client for the Qdrant HTTP API.

Exposes only the surface needed by the application: ensure a collection exists,
upsert vectors with payloads, and query for nearest neighbours.

Why not the official ``qdrant-client`` library?
-----------------------------------------------
The official client is fine, but it bundles synchronous and async support
behind a single class with subtle behavioural differences, pulls in gRPC
dependencies we don't need, and would force us to learn its idioms in
addition to Qdrant's HTTP API. A thin async ``httpx`` wrapper is ~100 lines,
makes the HTTP shape obvious, and means every dependency is one we already use.
"""

from __future__ import annotations

import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import TracebackType
from typing import Any, Protocol, Self

import httpx

from duediligence.clients._base import (
    QdrantError,
    transient_network_retry,
)
from duediligence.config import QdrantSettings
from duediligence.logging import get_logger

log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class VectorRecord:
    """One vector + its payload, ready to upsert into Qdrant."""

    id: str
    vector: list[float]
    payload: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class SearchHit:
    """One nearest-neighbour result from a vector query."""

    id: str
    score: float
    payload: Mapping[str, Any]


class QdrantClientProtocol(Protocol):
    """Surface the application depends on. See :class:`QdrantClient`."""

    async def health(self) -> None: ...
    async def ensure_collection(self) -> None: ...
    async def upsert(self, records: Sequence[VectorRecord]) -> None: ...
    async def search(
        self,
        vector: Sequence[float],
        *,
        limit: int = 5,
        filter_payload: Mapping[str, Any] | None = None,
    ) -> list[SearchHit]: ...


class QdrantClient:
    """Concrete Qdrant HTTP client.

    Use :meth:`from_settings` to construct one from application settings, and
    use the instance as an async context manager so the HTTP pool is closed
    on exit.
    """

    def __init__(
        self,
        *,
        base_url: str,
        collection_name: str,
        vector_size: int,
    ) -> None:
        self._collection_name = collection_name
        self._vector_size = vector_size
        self._client = httpx.AsyncClient(base_url=base_url, timeout=30.0)

    @classmethod
    def from_settings(cls, settings: QdrantSettings) -> Self:
        return cls(
            base_url=str(settings.host),
            collection_name=settings.collection_name,
            vector_size=settings.vector_size,
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
        """Verify Qdrant is reachable."""
        try:
            async for attempt in transient_network_retry():
                with attempt:
                    response = await self._client.get("/readyz")
                    response.raise_for_status()
        except httpx.HTTPError as e:
            raise QdrantError(f"qdrant unreachable at {self._client.base_url}") from e

    async def ensure_collection(self) -> None:
        """Create the configured collection if it does not already exist.

        Idempotent: a collection that already exists with matching vector size
        is left untouched. A mismatched vector size is a fatal configuration
        error and is surfaced as a :class:`QdrantError`.
        """
        existing = await self._get_collection()
        if existing is None:
            await self._create_collection()
            log.info(
                "qdrant_collection_created",
                collection=self._collection_name,
                vector_size=self._vector_size,
            )
            return

        actual_size = existing["config"]["params"]["vectors"]["size"]
        if actual_size != self._vector_size:
            raise QdrantError(
                f"collection '{self._collection_name}' has vector size "
                f"{actual_size}, but settings expect {self._vector_size}. "
                "drop the collection or update the configuration."
            )

    async def upsert(self, records: Sequence[VectorRecord]) -> None:
        """Insert or update a batch of vector records."""
        if not records:
            return

        points = [
            {
                "id": _coerce_point_id(r.id),
                "vector": r.vector,
                "payload": dict(r.payload),
            }
            for r in records
        ]
        try:
            async for attempt in transient_network_retry():
                with attempt:
                    response = await self._client.put(
                        f"/collections/{self._collection_name}/points",
                        params={"wait": "true"},
                        json={"points": points},
                    )
                    response.raise_for_status()
        except httpx.HTTPError as e:
            raise QdrantError(f"upsert of {len(records)} points failed: {e}") from e

    async def search(
        self,
        vector: Sequence[float],
        *,
        limit: int = 5,
        filter_payload: Mapping[str, Any] | None = None,
    ) -> list[SearchHit]:
        """Return the top-``limit`` nearest neighbours to ``vector``.

        ``filter_payload`` is an optional payload-equality filter applied
        server-side — useful for scoping searches to a single ticker or
        filing.
        """
        body: dict[str, Any] = {
            "vector": list(vector),
            "limit": limit,
            "with_payload": True,
        }
        if filter_payload:
            body["filter"] = {
                "must": [{"key": k, "match": {"value": v}} for k, v in filter_payload.items()]
            }

        try:
            async for attempt in transient_network_retry():
                with attempt:
                    response = await self._client.post(
                        f"/collections/{self._collection_name}/points/search",
                        json=body,
                    )
                    response.raise_for_status()
        except httpx.HTTPError as e:
            raise QdrantError(f"search failed: {e}") from e

        hits = response.json().get("result", [])
        return [
            SearchHit(id=str(h["id"]), score=float(h["score"]), payload=h.get("payload") or {})
            for h in hits
        ]

    # ----- Private helpers --------------------------------------------------

    async def _get_collection(self) -> dict[str, Any] | None:
        """Return the collection config if it exists, else None."""
        try:
            response = await self._client.get(f"/collections/{self._collection_name}")
            if response.status_code == httpx.codes.NOT_FOUND:
                return None
            response.raise_for_status()
        except httpx.HTTPError as e:
            raise QdrantError(f"failed to read collection metadata: {e}") from e

        result = response.json().get("result")
        if not isinstance(result, dict):
            raise QdrantError(f"unexpected collection metadata shape: {response.json()}")
        return result

    async def _create_collection(self) -> None:
        body = {
            "vectors": {
                "size": self._vector_size,
                "distance": "Cosine",
            }
        }
        try:
            response = await self._client.put(
                f"/collections/{self._collection_name}",
                json=body,
            )
            response.raise_for_status()
        except httpx.HTTPError as e:
            raise QdrantError(f"failed to create collection: {e}") from e


def _coerce_point_id(raw: str) -> str | int:
    """Qdrant accepts integer or UUID point ids only.

    The application uses string ids derived from content hashes. If the string
    happens to parse as a UUID, pass it through; otherwise derive a stable
    UUID5 from it. This keeps the application free to use opaque string ids
    while satisfying Qdrant's id constraints.
    """
    try:
        return str(uuid.UUID(raw))
    except ValueError:
        return str(uuid.uuid5(uuid.NAMESPACE_URL, raw))
