"""Smoke tests verifying the package is importable and well-formed."""

from __future__ import annotations

import duediligence


def test_package_imports() -> None:
    """The package can be imported without side effects."""
    assert duediligence is not None


def test_package_has_version() -> None:
    """The package exposes its version via __version__."""
    assert hasattr(duediligence, "__version__")
    assert isinstance(duediligence.__version__, str)
    assert len(duediligence.__version__) > 0
