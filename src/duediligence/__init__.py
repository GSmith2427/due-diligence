"""AI-powered due diligence system for company and leadership analysis."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("duediligence")
except PackageNotFoundError:  # pragma: no cover
    # Package is not installed (e.g. running from a source checkout)
    __version__ = "0.0.0+unknown"

__all__ = ["__version__"]
