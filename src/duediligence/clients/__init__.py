"""HTTP clients for external services used by the due-diligence system."""

from duediligence.clients._base import (
    ClientError,
    OllamaError,
    QdrantError,
)
from duediligence.clients.ollama import OllamaClient, OllamaClientProtocol
from duediligence.clients.qdrant import (
    QdrantClient,
    QdrantClientProtocol,
    SearchHit,
    VectorRecord,
)

__all__ = [
    "ClientError",
    "OllamaClient",
    "OllamaClientProtocol",
    "OllamaError",
    "QdrantClient",
    "QdrantClientProtocol",
    "QdrantError",
    "SearchHit",
    "VectorRecord",
]
