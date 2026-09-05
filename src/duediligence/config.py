"""Application configuration.

Settings are loaded from environment variables (with `DD_` prefix) and an
optional `.env` file in the project root. The `Settings` class is the single
source of truth for runtime configuration — there should be no `os.getenv()`
calls anywhere else in the codebase.

Access pattern:

    from duediligence.config import get_settings

    settings = get_settings()
    print(settings.ollama.host)

The `get_settings()` accessor is cached, so settings are constructed exactly
once per process. Tests can override individual values via
`Settings.model_copy(update={...})` or by clearing the cache.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, HttpUrl, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Project root resolved relative to this file — `src/duediligence/config.py`
# lives two directories below the project root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class OllamaSettings(BaseSettings):
    """Configuration for the local Ollama LLM runtime."""

    host: HttpUrl = Field(
        default=HttpUrl("http://localhost:11434"),
        description="Base URL of the Ollama HTTP API.",
    )
    chat_model: str = Field(
        default="qwen2.5:14b-instruct",
        description="Model identifier used for chat/completion calls.",
    )
    embedding_model: str = Field(
        default="bge-m3:latest",
        description="Model identifier used for embedding generation.",
    )
    request_timeout_seconds: float = Field(
        default=120.0,
        ge=1.0,
        description="HTTP timeout for individual LLM calls. Long-running "
        "generation needs generous timeouts; embedding calls are fast.",
    )


class QdrantSettings(BaseSettings):
    """Configuration for the Qdrant vector store."""

    host: HttpUrl = Field(
        default=HttpUrl("http://localhost:6333"),
        description="Base URL of the Qdrant HTTP API.",
    )
    collection_name: str = Field(
        default="filings",
        description="Default Qdrant collection used for filing chunks.",
    )
    vector_size: int = Field(
        default=1024,
        ge=1,
        description="Dimensionality of the embedding vectors. Must match the "
        "configured embedding model — bge-m3 produces 1024-dim vectors.",
    )


class SecEdgarSettings(BaseSettings):
    """Configuration for SEC EDGAR API access.

    The SEC requires a User-Agent header identifying the caller. Their fair-use
    policy caps requests at 10/sec; we enforce a lower rate ourselves.
    """

    base_url: HttpUrl = Field(default=HttpUrl("https://data.sec.gov"))
    user_agent: str = Field(
        default="duediligence research-project contact@example.com",
        description="Identifies the project to SEC EDGAR. Per SEC fair-use "
        "guidance, this should include a contact email.",
    )
    requests_per_second: float = Field(default=5.0, gt=0.0, le=10.0)


# Top-level field names of `Settings` that are populated from `DD_<NAME>`.
# Used by the unknown-variable check below.
_TOP_LEVEL_FIELDS: frozenset[str] = frozenset(
    {
        "environment",
        "log_level",
        "ollama",
        "qdrant",
        "sec_edgar",
    }
)


class Settings(BaseSettings):
    """Root application settings.

    Composed of subsections for each external dependency, plus top-level fields
    for service-wide concerns. Loaded from environment variables prefixed with
    ``DD_`` and from a ``.env`` file if present.

    Examples
    --------
    Environment-variable equivalents of nested fields use double underscores::

        DD_OLLAMA__CHAT_MODEL=llama3.1:8b-instruct
        DD_QDRANT__COLLECTION_NAME=my_filings
    """

    model_config = SettingsConfigDict(
        env_prefix="DD_",
        env_nested_delimiter="__",
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="forbid",
    )

    environment: Literal["development", "test", "production"] = Field(
        default="development",
        description="Deployment environment. Drives log formatting and other "
        "environment-dependent behaviour.",
    )
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
    )

    ollama: OllamaSettings = Field(default_factory=OllamaSettings)
    qdrant: QdrantSettings = Field(default_factory=QdrantSettings)
    sec_edgar: SecEdgarSettings = Field(default_factory=SecEdgarSettings)

    @model_validator(mode="before")
    @classmethod
    def _reject_unknown_dd_env_vars(cls, values: Any) -> Any:
        """Reject any ``DD_*`` environment variable that doesn't map to a known field.

        Without this, a typo such as ``DD_LOG_LEVL=DEBUG`` would silently do
        nothing — the service would start with the default log level and the
        misconfiguration would go undetected. By scanning ``os.environ`` at
        construction time we surface the typo immediately as a validation error.

        Recognised forms are::

            DD_<top_level_field>                  e.g. DD_LOG_LEVEL
            DD_<subsection>__<sub_field>          e.g. DD_OLLAMA__CHAT_MODEL
        """
        prefix = "DD_"
        delimiter = "__"
        unknown: list[str] = []

        for raw_key in os.environ:
            if not raw_key.startswith(prefix):
                continue
            key = raw_key[len(prefix) :].lower()
            head, _, _ = key.partition(delimiter)
            if head not in _TOP_LEVEL_FIELDS:
                unknown.append(raw_key)

        if unknown:
            joined = ", ".join(sorted(unknown))
            raise ValueError(
                f"Unknown DD_-prefixed environment variable(s): {joined}. "
                "Check for typos or remove the variable."
            )
        return values


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings instance.

    Cached so settings are constructed exactly once per process. Call
    ``get_settings.cache_clear()`` in tests when overriding environment.
    """
    return Settings()
