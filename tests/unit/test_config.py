"""Tests for config.py — property getters with fallback logic.

These properties have conditional logic (fallbacks, warnings) that
is not exercised by simply instantiating the config.

NOTE: The project loads .env via dotenv, so tests that need to verify
fallback behavior must explicitly override or remove env vars.
"""

from unittest.mock import patch

from config import DatabaseConfig, LLMConfig


class TestLLMConfigApiKey:
    """Tests for LLMConfig.api_key property with fallback chain."""

    def test_returns_open_router_api_key_when_set(self) -> None:
        """Should return OPEN_ROUTER_API_KEY when it is explicitly set."""
        config = LLMConfig(OPEN_ROUTER_API_KEY="sk-or-key-123")
        assert config.api_key == "sk-or-key-123"

    def test_falls_back_to_openrouter_api_key(self) -> None:
        """Should fall back to OPENROUTER_API_KEY when OPEN_ROUTER_API_KEY is None."""
        config = LLMConfig(
            OPEN_ROUTER_API_KEY=None, OPENROUTER_API_KEY="sk-fallback-key"
        )
        assert config.api_key == "sk-fallback-key"

    def test_prefers_open_router_api_key_over_fallback(self) -> None:
        """Should prefer OPEN_ROUTER_API_KEY over OPENROUTER_API_KEY."""
        config = LLMConfig(
            OPEN_ROUTER_API_KEY="sk-primary",
            OPENROUTER_API_KEY="sk-secondary",
        )
        assert config.api_key == "sk-primary"

    def test_returns_empty_string_when_both_none(self) -> None:
        """Should return empty string and print warning when both keys are None."""
        config = LLMConfig(OPEN_ROUTER_API_KEY=None, OPENROUTER_API_KEY=None)
        with patch("builtins.print") as mock_print:
            result = config.api_key
            mock_print.assert_called_once_with(
                "WARNING: OPENROUTER_API_KEY not set. API calls will fail."
            )
        assert result == ""


class TestLLMConfigApiBaseUrl:
    """Tests for LLMConfig.api_base_url property with fallback."""

    def test_returns_base_url_when_set(self) -> None:
        """Should return BASE_URL when it is set."""
        config = LLMConfig(BASE_URL="https://custom.api.com/v1")
        assert config.api_base_url == "https://custom.api.com/v1"

    def test_falls_back_to_open_router_api_url(self) -> None:
        """Should fall back to OPEN_ROUTER_API_URL when BASE_URL is None."""
        config = LLMConfig(BASE_URL=None)
        assert config.api_base_url == "https://openrouter.ai/api/v1"

    def test_prefers_base_url_over_default(self) -> None:
        """Should prefer BASE_URL over OPEN_ROUTER_API_URL."""
        config = LLMConfig(
            BASE_URL="https://custom.api.com/v1",
            OPEN_ROUTER_API_URL="https://other.api.com/v1",
        )
        assert config.api_base_url == "https://custom.api.com/v1"


class TestDatabaseConfigURL:
    """Tests for DatabaseConfig.DATABASE_URL property."""

    def test_constructs_valid_postgres_url(self) -> None:
        """Should construct a proper asyncpg PostgreSQL URL."""
        config = DatabaseConfig(
            POSTGRES_USER="myuser",
            POSTGRES_PASSWORD="secret",
            POSTGRES_HOST="db.example.com",
            POSTGRES_PORT="5433",
            POSTGRES_DATABASE="mydb",
        )
        expected = "postgresql+asyncpg://myuser:secret@db.example.com:5433/mydb"
        assert expected == config.DATABASE_URL

    def test_uses_explicit_default_values(self) -> None:
        """Should construct URL with the default field values from the model."""
        config = DatabaseConfig(
            POSTGRES_USER="raganything",
            POSTGRES_PASSWORD="raganything",
            POSTGRES_HOST="localhost",
            POSTGRES_PORT="5432",
            POSTGRES_DATABASE="raganything",
        )
        expected = (
            "postgresql+asyncpg://raganything:raganything@localhost:5432/raganything"
        )
        assert expected == config.DATABASE_URL

    def test_handles_special_characters_in_password(self) -> None:
        """Should include special characters verbatim in the URL."""
        config = DatabaseConfig(
            POSTGRES_USER="admin",
            POSTGRES_PASSWORD="p@ss:w0rd/complex",
        )
        assert "p@ss:w0rd/complex" in config.DATABASE_URL
