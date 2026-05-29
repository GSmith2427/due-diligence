"""SEC EDGAR data source."""

from duediligence.sources.edgar.client import SecEdgarClient, SecEdgarError
from duediligence.sources.edgar.models import (
    CompanyIdentity,
    Filing,
    FilingMetadata,
    Provenance,
)

__all__ = [
    "CompanyIdentity",
    "Filing",
    "FilingMetadata",
    "Provenance",
    "SecEdgarClient",
    "SecEdgarError",
]
