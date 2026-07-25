"""Tests for the env-var resolution helpers (infrastructure.env_utils).

``resolve_env_vars`` replaces ``${VAR_NAME}`` patterns in a string using
``os.environ``. Unknown placeholders are kept verbatim.
``resolve_env_vars_in_dict`` applies the same resolution to every string value
of a dict and passes non-string values through unchanged.

These are pure helpers with no external dependencies, so they are exercised
directly. Environment variables are controlled via ``monkeypatch``.
"""

import pytest

from infrastructure.env_utils import resolve_env_vars, resolve_env_vars_in_dict


# -- resolve_env_vars ----------------------------------------------------------


class TestResolveEnvVars:
    """Tests for resolve_env_vars."""

    def test_resolve_env_vars_replaces_known_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Arrange
        monkeypatch.setenv("MY_VAR", "my-value")

        # Act
        result = resolve_env_vars("token-${MY_VAR}-suffix")

        # Assert
        assert result == "token-my-value-suffix"

    def test_resolve_env_vars_keeps_unknown_placeholder(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Arrange — ensure the var is unset
        monkeypatch.delenv("MISSING_VAR", raising=False)

        # Act
        result = resolve_env_vars("prefix-${MISSING_VAR}-suffix")

        # Assert
        assert result == "prefix-${MISSING_VAR}-suffix"

    def test_resolve_env_vars_no_placeholder_returns_unchanged(self) -> None:
        # Act / Assert
        assert resolve_env_vars("plain-string-no-placeholder") == "plain-string-no-placeholder"

    def test_resolve_env_vars_multiple_placeholders(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Arrange
        monkeypatch.setenv("A", "alpha")
        monkeypatch.setenv("B", "beta")

        # Act
        result = resolve_env_vars("${A} and ${B}")

        # Assert
        assert result == "alpha and beta"


# -- resolve_env_vars_in_dict --------------------------------------------------


class TestResolveEnvVarsInDict:
    """Tests for resolve_env_vars_in_dict."""

    def test_resolve_env_vars_in_dict_resolves_string_values(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Arrange
        monkeypatch.setenv("TOK", "secret-token")
        mapping = {"Authorization": "Bearer ${TOK}", "X-Other": "static"}

        # Act
        result = resolve_env_vars_in_dict(mapping)

        # Assert
        assert result == {"Authorization": "Bearer secret-token", "X-Other": "static"}

    def test_resolve_env_vars_in_dict_passes_non_string_values(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Arrange — an int value and a list value must pass through unchanged
        monkeypatch.setenv("X", "ignored")
        mapping = {"count": 42, "enabled": True, "items": [1, 2, 3]}

        # Act
        result = resolve_env_vars_in_dict(mapping)

        # Assert
        assert result == {"count": 42, "enabled": True, "items": [1, 2, 3]}

    def test_resolve_env_vars_in_dict_empty_dict_returns_empty(self) -> None:
        # Act / Assert
        assert resolve_env_vars_in_dict({}) == {}
