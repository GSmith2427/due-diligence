"""Tests for the configuration module."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from duediligence.config import Settings, get_settings


def test_defaults_load_without_environment() -> None:
    """Settings can be constructed with no environment variables set."""
    settings = Settings()

    assert settings.environment == "development"
    assert settings.log_level == "INFO"
    assert str(settings.ollama.host).rstrip("/") == "http://localhost:11434"
    assert settings.qdrant.collection_name == "filings"
    assert settings.qdrant.vector_size == 1024


def test_nested_env_var_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """Nested settings are populated via the double-underscore delimiter."""
    monkeypatch.setenv("DD_OLLAMA__CHAT_MODEL", "llama3.1:8b-instruct")
    monkeypatch.setenv("DD_QDRANT__VECTOR_SIZE", "768")

    settings = Settings()

    assert settings.ollama.chat_model == "llama3.1:8b-instruct"
    assert settings.qdrant.vector_size == 768


def test_top_level_env_var_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """Top-level settings respect the DD_ prefix."""
    monkeypatch.setenv("DD_LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("DD_ENVIRONMENT", "production")

    settings = Settings()

    assert settings.log_level == "DEBUG"
    assert settings.environment == "production"


def test_invalid_log_level_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    """Invalid log levels raise a validation error at construction time."""
    monkeypatch.setenv("DD_LOG_LEVEL", "VERBOSE")

    with pytest.raises(ValidationError):
        Settings()


def test_unknown_field_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unknown environment variables with DD_ prefix are rejected.

    This guards against typos like DD_LOG_LEVL silently doing nothing.
    """
    monkeypatch.setenv("DD_NONEXISTENT_FIELD", "value")

    with pytest.raises(ValidationError):
        Settings()


def test_get_settings_is_cached() -> None:
    """Repeat calls to get_settings return the same instance."""
    get_settings.cache_clear()
    a = get_settings()
    b = get_settings()

    assert a is b
