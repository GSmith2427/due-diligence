"""Ticker → CIK lookup.

SEC identifies every filer by a ten-digit Central Index Key. The public
ticker-to-CIK map lives at
``https://www.sec.gov/files/company_tickers.json`` and is updated nightly.
We fetch it once per process and cache the parsed map.
"""

from __future__ import annotations

import asyncio
from typing import Any

import httpx

from duediligence.logging import get_logger

log = get_logger(__name__)

_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"


class TickerIndex:
    """Lazily-loaded ticker → CIK lookup.

    The first call to :meth:`resolve` fetches the SEC's ticker map and caches
    it for the lifetime of the instance. Subsequent calls hit the cache.
    """

    def __init__(self, http: httpx.AsyncClient) -> None:
        self._http = http
        self._map: dict[str, tuple[str, str]] | None = None
        self._lock = asyncio.Lock()

    async def resolve(self, ticker: str) -> tuple[str, str]:
        """Return ``(cik_10_digit, company_name)`` for a ticker.

        Raises :class:`KeyError` if the ticker is not in SEC's index.
        """
        index = await self._load()
        try:
            return index[ticker.upper()]
        except KeyError as e:
            raise KeyError(f"ticker {ticker!r} not found in SEC EDGAR index") from e

    async def _load(self) -> dict[str, tuple[str, str]]:
        cached = self._map
        if cached is not None:
            return cached
        async with self._lock:
            # Re-check after acquiring the lock; another coroutine may have
            # populated the cache while we were waiting.
            cached = self._map
            if cached is not None:
                return cached

            log.info("ticker_index_loading", url=_TICKERS_URL)
            response = await self._http.get(_TICKERS_URL)
            response.raise_for_status()
            raw: dict[str, dict[str, Any]] = response.json()

            new_map = {
                row["ticker"].upper(): (str(row["cik_str"]).zfill(10), row["title"])
                for row in raw.values()
            }
            self._map = new_map
            log.info("ticker_index_loaded", entries=len(new_map))
            return new_map
