"""Tests for config.py — property getters with fallback logic.

These properties have conditional logic (fallbacks, warnings) that
is not exercised by simply instantiating the config.

NOTE: The project loads .env via dotenv, so tests that need to verify
fallback behavior must explicitly override or remove env vars.
"""

from unittest.mock import patch

import pytest
from pydantic import ValidationError

from config import DatabaseConfig, LLMConfig


class TestLLMConfigApiKey:
    """Tests for LLMConfig.api_key property with fallback chain."""

    def test_returns_open_router_api_key_when_set(self) -> None:
        """Should return OPEN_ROUTER_API_KEY when it is explicitly set."""
        config = LLMConfig(OPEN_ROUTER_API_KEY="sk-or-key-123")
        assert config.api_key == "sk-or-key-123"

    def test_falls_back_to_openrouter_api_key(self) -> None:
        """Should fall back to OPENROUTER_API_KEY when OPEN_ROUTER_API_KEY is None."""
        # Arrange
        config = LLMConfig(
            OPEN_ROUTER_API_KEY=None, OPENROUTER_API_KEY="sk-fallback-key"
        )
        # Assert
        assert config.api_key == "sk-fallback-key"

    def test_prefers_open_router_api_key_over_fallback(self) -> None:
        """Should prefer OPEN_ROUTER_API_KEY over OPENROUTER_API_KEY."""
        config = LLMConfig(
            OPEN_ROUTER_API_KEY="sk-primary",
            OPENROUTER_API_KEY="sk-secondary",
        )
        assert config.api_key == "sk-primary"

    def test_returns_empty_string_when_both_none(self) -> None:
        """Should return empty string and log warning when both keys are None."""
        # Arrange
        config = LLMConfig(OPEN_ROUTER_API_KEY=None, OPENROUTER_API_KEY=None)
        with patch("config.logger.warning") as mock_warning:
            result = config.api_key
            # Assert
            mock_warning.assert_called_once_with(
                "OPENROUTER_API_KEY not set. API calls will fail."
            )
        # Assert
        assert result == ""


class TestLLMConfigApiBaseUrl:
    """Tests for LLMConfig.api_base_url property with fallback."""

    def test_returns_base_url_when_set(self) -> None:
        """Should return BASE_URL when it is set."""
        config = LLMConfig(BASE_URL="https://custom.api.com/v1")
        assert config.api_base_url == "https://custom.api.com/v1"

    def test_falls_back_to_open_router_api_url(self) -> None:
        """Should fall back to OPEN_ROUTER_API_URL when BASE_URL is None."""
        # Arrange
        config = LLMConfig(BASE_URL=None)
        # Assert
        assert config.api_base_url == "https://openrouter.ai/api/v1"

    def test_prefers_base_url_over_default(self) -> None:
        """Should prefer BASE_URL over OPEN_ROUTER_API_URL."""
        config = LLMConfig(
            BASE_URL="https://custom.api.com/v1",
            OPEN_ROUTER_API_URL="https://other.api.com/v1",
        )
        assert config.api_base_url == "https://custom.api.com/v1"


class TestDatabaseConfigURL:
    """Tests for DatabaseConfig DATABASE_URL normalization."""

    def test_postgresql_scheme_normalized_to_asyncpg(self) -> None:
        """Should normalize postgresql:// to postgresql+asyncpg://."""
        config = DatabaseConfig(
            DATABASE_URL="postgresql://myuser:secret@db.example.com:5433/mydb"
        )
        expected = "postgresql+asyncpg://myuser:secret@db.example.com:5433/mydb"
        assert expected == config.DATABASE_URL

    def test_postgres_scheme_normalized_to_asyncpg(self) -> None:
        """Should normalize postgres:// to postgresql+asyncpg://."""
        config = DatabaseConfig(DATABASE_URL="postgres://user:pass@host:5432/db")
        assert config.DATABASE_URL == "postgresql+asyncpg://user:pass@host:5432/db"

    def test_already_asyncpg_scheme_unchanged(self) -> None:
        """Should leave postgresql+asyncpg:// URLs unchanged."""
        url = "postgresql+asyncpg://user:pass@host:5432/db"
        config = DatabaseConfig(DATABASE_URL=url)
        assert url == config.DATABASE_URL

    def test_default_url_uses_asyncpg(self) -> None:
        """Should construct default URL with asyncpg driver."""
        config = DatabaseConfig(
            DATABASE_URL="postgresql://raganything:raganything@localhost:5432/raganything"
        )
        expected = (
            "postgresql+asyncpg://raganything:raganything@localhost:5432/raganything"
        )
        assert expected == config.DATABASE_URL

    def test_sslmode_and_channel_binding_stripped_from_url(self) -> None:
        """Should strip sslmode and channel_binding from URL (asyncpg doesn't accept them)."""
        config = DatabaseConfig(
            DATABASE_URL="postgresql://neondb_owner:password@ep-xxx.neon.tech/neondb?sslmode=require&channel_binding=require"
        )
        assert config.DATABASE_URL == (
            "postgresql+asyncpg://neondb_owner:password@ep-xxx.neon.tech/neondb"
        )

    def test_sslmode_extracted_to_property(self) -> None:
        """Should extract sslmode to ssl_mode property for connect_args usage."""
        config = DatabaseConfig(
            DATABASE_URL="postgresql://user:pass@host:5432/db?sslmode=require"
        )
        assert config.ssl_mode == "require"

    def test_no_sslmode_returns_none(self) -> None:
        """Should return None for ssl_mode when no sslmode in URL."""
        config = DatabaseConfig(DATABASE_URL="postgresql://user:pass@host:5432/db")
        assert config.ssl_mode is None

    def test_other_query_params_preserved(self) -> None:
        """Should preserve query params other than sslmode/channel_binding."""
        config = DatabaseConfig(
            DATABASE_URL="postgresql://user:pass@host:5432/db?sslmode=require&application_name=myapp"
        )
        assert "application_name=myapp" in config.DATABASE_URL
        assert "sslmode" not in config.DATABASE_URL

    def test_special_characters_in_password_preserved(self) -> None:
        """Should include special characters verbatim in the URL."""
        config = DatabaseConfig(
            DATABASE_URL="postgresql://admin:p@ss:w0rd/complex@host:5432/db"
        )
        assert "p@ss:w0rd/complex" in config.DATABASE_URL

    def test_asyncpg_url_strips_driver_suffix(self) -> None:
        """asyncpg_url property should strip +asyncpg for raw asyncpg usage."""
        config = DatabaseConfig(DATABASE_URL="postgresql://user:pass@host:5432/db")
        assert config.asyncpg_url == "postgresql://user:pass@host:5432/db"

    def test_asyncpg_url_preserves_plus_asyncpg_in_password(self) -> None:
        """asyncpg_url should only strip the scheme's +asyncpg suffix, not occurrences in the password.

        The naive str.replace strips ALL "+asyncpg" occurrences, corrupting passwords
        that legitimately contain "+asyncpg" (e.g. generated credentials).
        """
        # Arrange — password is "foo+asyncpg"; validator prepends scheme suffix
        config = DatabaseConfig(
            DATABASE_URL="postgresql://user:foo+asyncpg@host:5432/db"
        )
        # Assert — only the scheme's "+asyncpg" is removed; password keeps its "+asyncpg"
        assert config.asyncpg_url == "postgresql://user:foo+asyncpg@host:5432/db"

    def test_invalid_scheme_rejected(self) -> None:
        """Should reject URLs with an unsupported scheme at startup."""
        with pytest.raises(ValidationError):
            DatabaseConfig(DATABASE_URL="mysql://user:pass@host:3306/db")

    def test_missing_host_rejected(self) -> None:
        """Should reject URLs with no host component (e.g. typo with missing //)."""
        with pytest.raises(ValidationError):
            DatabaseConfig(DATABASE_URL="postgresql:user:pass@/db")

    def test_not_a_url_rejected(self) -> None:
        """Should reject non-URL strings that urlsplit would silently accept."""
        with pytest.raises(ValidationError):
            DatabaseConfig(DATABASE_URL="not a url")


class TestDatabaseConfigStatementCacheSize:
    """Tests for POSTGRES_STATEMENT_CACHE_SIZE config."""

    def test_default_is_none(self) -> None:
        config = DatabaseConfig()
        assert config.POSTGRES_STATEMENT_CACHE_SIZE is None

    def test_can_set_to_zero_for_poolers(self) -> None:
        config = DatabaseConfig(POSTGRES_STATEMENT_CACHE_SIZE=0)
        assert config.POSTGRES_STATEMENT_CACHE_SIZE == 0

    def test_can_set_to_custom_value(self) -> None:
        config = DatabaseConfig(POSTGRES_STATEMENT_CACHE_SIZE=50)
        assert config.POSTGRES_STATEMENT_CACHE_SIZE == 50
