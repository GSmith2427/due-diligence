"""Async client for SEC EDGAR.

Provides one high-level operation: ``fetch_latest_10k(ticker)`` returns a
:class:`Filing` with cleaned text and full provenance. Lower-level pieces
(ticker resolution, submissions index, raw document download) are exposed
on the client for completeness but most callers won't need them.

The client enforces SEC's fair-use policy:
    * Sets a descriptive User-Agent header on every request.
    * Limits itself to a configurable requests-per-second rate (default 5/s).

Usage::

    async with SecEdgarClient.from_settings(settings.sec_edgar) as edgar:
        filing = await edgar.fetch_latest_10k("AAPL")
"""

from __future__ import annotations

from datetime import UTC, datetime
from types import TracebackType
from typing import Any, Self

import httpx
from aiolimiter import AsyncLimiter

from duediligence.config import SecEdgarSettings
from duediligence.logging import get_logger
from duediligence.sources.edgar._ticker_index import TickerIndex
from duediligence.sources.edgar.models import (
    CompanyIdentity,
    Filing,
    FilingMetadata,
    Provenance,
)
from duediligence.sources.edgar.parser import html_to_text

log = get_logger(__name__)


class SecEdgarError(Exception):
    """A SEC EDGAR operation failed."""


class SecEdgarClient:
    """High-level client for SEC EDGAR."""

    def __init__(
        self,
        *,
        user_agent: str,
        requests_per_second: float,
    ) -> None:
        self._http = httpx.AsyncClient(
            timeout=30.0,
            headers={"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"},
            follow_redirects=True,
        )
        # aiolimiter rate-limits to `max_rate` events per `time_period` seconds.
        self._limiter = AsyncLimiter(max_rate=requests_per_second, time_period=1.0)
        self._ticker_index = TickerIndex(self._http)

    @classmethod
    def from_settings(cls, settings: SecEdgarSettings) -> Self:
        return cls(
            user_agent=settings.user_agent,
            requests_per_second=settings.requests_per_second,
        )

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self._http.aclose()

    # ----- Public operations ------------------------------------------------

    async def fetch_latest_10k(self, ticker: str) -> Filing:
        """Fetch the most recent 10-K filing for ``ticker``.

        Raises :class:`SecEdgarError` if the ticker is unknown or the company
        has no 10-K on file.
        """
        try:
            cik, name = await self._ticker_index.resolve(ticker)
        except KeyError as e:
            raise SecEdgarError(str(e)) from e

        company = CompanyIdentity(cik=cik, ticker=ticker.upper(), name=name)
        submissions = await self._fetch_submissions(cik)
        metadata = _pick_latest_10k(submissions, cik=cik)

        doc_url = _build_document_url(cik=cik, metadata=metadata)
        content, fetched_at = await self._download(doc_url)
        text = html_to_text(content)

        return Filing(
            company=company,
            metadata=metadata,
            text=text,
            provenance=Provenance.for_content(
                source="sec-edgar",
                url=doc_url,
                content=content,
                fetched_at=fetched_at,
            ),
        )

    # ----- Internals --------------------------------------------------------

    async def _fetch_submissions(self, cik: str) -> dict[str, Any]:
        url = f"https://data.sec.gov/submissions/CIK{cik}.json"
        async with self._limiter:
            try:
                response = await self._http.get(url)
                response.raise_for_status()
            except httpx.HTTPError as e:
                raise SecEdgarError(f"failed to fetch submissions for CIK {cik}: {e}") from e
        payload: dict[str, Any] = response.json()
        return payload

    async def _download(self, url: str) -> tuple[bytes, datetime]:
        async with self._limiter:
            try:
                response = await self._http.get(url)
                response.raise_for_status()
            except httpx.HTTPError as e:
                raise SecEdgarError(f"failed to download {url}: {e}") from e
        return response.content, datetime.now(UTC)


def _pick_latest_10k(submissions: dict[str, Any], *, cik: str) -> FilingMetadata:
    """Extract metadata for the most recent 10-K from a submissions payload."""
    recent = submissions.get("filings", {}).get("recent", {})
    forms: list[str] = recent.get("form", [])
    accession_numbers: list[str] = recent.get("accessionNumber", [])
    filing_dates: list[str] = recent.get("filingDate", [])
    report_dates: list[str] = recent.get("reportDate", [])
    primary_docs: list[str] = recent.get("primaryDocument", [])

    for i, form in enumerate(forms):
        if form == "10-K":
            return FilingMetadata(
                accession_number=accession_numbers[i],
                form=form,
                filing_date=datetime.fromisoformat(filing_dates[i]),
                report_date=(datetime.fromisoformat(report_dates[i]) if report_dates[i] else None),
                primary_document=primary_docs[i],
            )
    raise SecEdgarError(f"no 10-K filings found for CIK {cik}")


def _build_document_url(*, cik: str, metadata: FilingMetadata) -> str:
    """Construct the URL for the primary document of a filing.

    EDGAR document URLs follow the pattern
    ``/Archives/edgar/data/{cik_no_pad}/{accession_no_dashes}/{primary_doc}``.
    """
    accession_no_dashes = metadata.accession_number.replace("-", "")
    cik_no_pad = str(int(cik))  # strip leading zeros
    return (
        "https://www.sec.gov/Archives/edgar/data/"
        f"{cik_no_pad}/{accession_no_dashes}/{metadata.primary_document}"
    )
