"""Typed records for the ingestion pipeline."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class Chunk(BaseModel):
    """A unit of retrievable text from a source document.

    Carries enough metadata to reconstruct citations: the index within the
    source document and the character offsets back to the original text.
    """

    model_config = ConfigDict(frozen=True)

    index: int = Field(
        ge=0,
        description="Zero-based ordinal of this chunk within its source document.",
    )
    text: str = Field(min_length=1, description="The chunk's text content.")
    token_count: int = Field(
        ge=1,
        description="Approximate token count, used for sizing during chunking.",
    )
    char_start: int = Field(
        ge=0,
        description="Inclusive character offset of this chunk in the source document.",
    )
    char_end: int = Field(
        gt=0,
        description="Exclusive character offset of this chunk in the source document.",
    )
