"""Typed records for SEC EDGAR data.

Every record carries a ``Provenance`` block so downstream code can cite the
exact source and detect duplicates by content hash.
"""

from __future__ import annotations

import hashlib
from datetime import datetime
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, HttpUrl


class Provenance(BaseModel):
    """Where a piece of data came from.

    Attached to every record so the final report can cite primary sources.
    """

    model_config = ConfigDict(frozen=True)

    source: str = Field(description="Stable identifier of the source system (e.g. 'sec-edgar').")
    url: HttpUrl = Field(description="Canonical URL of the resource.")
    fetched_at: datetime = Field(description="Timestamp the resource was retrieved (UTC).")
    content_sha256: str = Field(
        description="SHA-256 of the raw bytes as fetched. Used for dedup and integrity.",
        min_length=64,
        max_length=64,
    )

    @classmethod
    def for_content(cls, *, source: str, url: str, content: bytes, fetched_at: datetime) -> Self:
        return cls(
            source=source,
            url=HttpUrl(url),
            fetched_at=fetched_at,
            content_sha256=hashlib.sha256(content).hexdigest(),
        )


class CompanyIdentity(BaseModel):
    """Minimal company identity used throughout the system."""

    model_config = ConfigDict(frozen=True)

    cik: str = Field(
        description="Ten-digit zero-padded SEC Central Index Key.", pattern=r"^\d{10}$"
    )
    ticker: str = Field(description="Primary trading ticker (uppercase).")
    name: str


class FilingMetadata(BaseModel):
    """Header info for a single SEC filing."""

    model_config = ConfigDict(frozen=True)

    accession_number: str = Field(
        description="SEC accession number in dashed form (e.g. '0000320193-24-000123').",
        pattern=r"^\d{10}-\d{2}-\d{6}$",
    )
    form: str = Field(description="Form type, e.g. '10-K', '10-Q', 'DEF 14A'.")
    filing_date: datetime
    report_date: datetime | None = Field(
        default=None,
        description="Period the filing covers, if applicable.",
    )
    primary_document: str = Field(description="Filename of the primary document inside the filing.")


class Filing(BaseModel):
    """A fully-retrieved filing: metadata, clean text, and provenance."""

    company: CompanyIdentity
    metadata: FilingMetadata
    text: str = Field(description="Cleaned plain text of the primary document.")
    provenance: Provenance
