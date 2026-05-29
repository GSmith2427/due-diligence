"""Tests for the SEC EDGAR collector."""

from __future__ import annotations

from collections.abc import Callable

import httpx
import pytest

from duediligence.sources.edgar import (
    Filing,
    SecEdgarClient,
    SecEdgarError,
)
from duediligence.sources.edgar.parser import html_to_text

# ---------------------------------------------------------------------------
# Parser unit tests
# ---------------------------------------------------------------------------


def test_parser_drops_script_and_style() -> None:
    html = b"""
    <html><head><style>body{color:red}</style></head>
    <body>
      <script>alert(1)</script>
      <p>Real content here.</p>
    </body></html>
    """
    text = html_to_text(html)
    assert "alert" not in text
    assert "color:red" not in text
    assert "Real content here." in text


def test_parser_renders_tables_as_tsv() -> None:
    html = b"""
    <html><body>
      <table>
        <tr><th>Year</th><th>Revenue</th></tr>
        <tr><td>2023</td><td>$100M</td></tr>
        <tr><td>2024</td><td>$120M</td></tr>
      </table>
    </body></html>
    """
    text = html_to_text(html)
    assert "Year\tRevenue" in text
    assert "2023\t$100M" in text
    assert "2024\t$120M" in text


def test_parser_collapses_whitespace_but_preserves_paragraphs() -> None:
    html = b"<html><body><p>One.</p><p>Two.</p><p>Three.</p></body></html>"
    text = html_to_text(html)
    # Three paragraphs should produce three lines of content
    non_empty = [line for line in text.splitlines() if line.strip()]
    assert non_empty == ["One.", "Two.", "Three."]


# ---------------------------------------------------------------------------
# Client unit tests — mocked HTTP transport
# ---------------------------------------------------------------------------


_TICKERS_PAYLOAD = {
    "0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."},
    "1": {"cik_str": 789019, "ticker": "MSFT", "title": "Microsoft Corporation"},
}


_SUBMISSIONS_PAYLOAD = {
    "filings": {
        "recent": {
            "form": ["8-K", "10-K", "10-Q"],
            "accessionNumber": [
                "0000320193-24-000100",
                "0000320193-24-000123",
                "0000320193-24-000130",
            ],
            "filingDate": ["2024-10-01", "2024-11-01", "2024-11-15"],
            "reportDate": ["2024-09-30", "2024-09-30", "2024-09-30"],
            "primaryDocument": ["8k.htm", "aapl-20240928.htm", "10q.htm"],
        }
    }
}


_FILING_HTML = b"""
<html><body>
  <h1>ANNUAL REPORT ON FORM 10-K</h1>
  <p>This is the body of the filing.</p>
  <table><tr><th>Year</th><th>Revenue</th></tr><tr><td>2024</td><td>$391B</td></tr></table>
</body></html>
"""


def _build_handler() -> Callable[[httpx.Request], httpx.Response]:
    """Construct a MockTransport handler that serves all three expected URLs."""

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if url.endswith("/files/company_tickers.json"):
            return httpx.Response(200, json=_TICKERS_PAYLOAD)
        if "submissions/CIK" in url:
            return httpx.Response(200, json=_SUBMISSIONS_PAYLOAD)
        if "Archives/edgar/data" in url and url.endswith("aapl-20240928.htm"):
            return httpx.Response(200, content=_FILING_HTML)
        return httpx.Response(404, json={"error": f"unexpected url {url}"})

    return handler


def _mock_client() -> SecEdgarClient:
    client = SecEdgarClient(
        user_agent="duediligence-test test@example.com",
        requests_per_second=100.0,  # effectively disable rate limiting in tests
    )
    client._http = httpx.AsyncClient(
        timeout=5.0,
        transport=httpx.MockTransport(_build_handler()),
        follow_redirects=True,
        headers={"User-Agent": "duediligence-test"},
    )
    # The ticker index holds a reference to the original client; swap it.
    from duediligence.sources.edgar._ticker_index import TickerIndex

    client._ticker_index = TickerIndex(client._http)
    return client


async def test_fetch_latest_10k_returns_parsed_filing() -> None:
    async with _mock_client() as edgar:
        filing = await edgar.fetch_latest_10k("AAPL")

    assert isinstance(filing, Filing)
    assert filing.company.ticker == "AAPL"
    assert filing.company.cik == "0000320193"
    assert filing.metadata.form == "10-K"
    assert filing.metadata.accession_number == "0000320193-24-000123"
    assert "ANNUAL REPORT" in filing.text
    assert "2024\t$391B" in filing.text
    assert filing.provenance.source == "sec-edgar"
    assert len(filing.provenance.content_sha256) == 64


async def test_unknown_ticker_raises() -> None:
    async with _mock_client() as edgar:
        with pytest.raises(SecEdgarError, match="NOPE"):
            await edgar.fetch_latest_10k("NOPE")


async def test_lowercase_ticker_normalises_to_uppercase() -> None:
    async with _mock_client() as edgar:
        filing = await edgar.fetch_latest_10k("aapl")

    assert filing.company.ticker == "AAPL"


# ---------------------------------------------------------------------------
# Integration test — hits the real SEC
# ---------------------------------------------------------------------------


@pytest.mark.integration
async def test_real_apple_10k() -> None:
    """Live test: fetch Apple's most recent 10-K from the real SEC.

    Skipped in CI. Run locally with: pytest -m integration
    """
    from duediligence.config import get_settings

    get_settings.cache_clear()
    settings = get_settings()
    async with SecEdgarClient.from_settings(settings.sec_edgar) as edgar:
        filing = await edgar.fetch_latest_10k("AAPL")

    assert filing.company.ticker == "AAPL"
    assert filing.metadata.form == "10-K"
    assert len(filing.text) > 100_000  # a 10-K is many tens of thousands of words
