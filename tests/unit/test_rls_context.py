"""Tests for the RLS contextvars module.

The module exposes ``current_user_id``, ``current_credential``,
``current_auth_method`` and ``bypass_rls`` contextvars plus a
``system_rls_context`` async context manager that temporarily enables RLS
bypass for system/migration queries.

It also exposes ``set_rls_context(conn, user_id)`` — an async helper that
sets the PostgreSQL GUC ``app.user_id`` on a raw asyncpg connection (used by
``McpRegistryStore``) so Row-Level Security policies on ``mcp_servers``
filter rows per authenticated user.
"""

from unittest.mock import AsyncMock

import pytest

from infrastructure.database.rls_context import (
    bypass_rls,
    current_auth_method,
    current_credential,
    current_user_id,
    set_rls_context,
    system_rls_context,
)


class TestRlsContextDefaults:
    """Tests for contextvar default values."""

    def test_current_user_id_defaults_to_none(self) -> None:
        # Act & Assert
        assert current_user_id.get() is None

    def test_current_credential_defaults_to_none(self) -> None:
        # Act & Assert
        assert current_credential.get() is None

    def test_current_auth_method_defaults_to_none(self) -> None:
        # Act & Assert
        assert current_auth_method.get() is None

    def test_bypass_rls_defaults_to_false(self) -> None:
        # Act & Assert
        assert bypass_rls.get() is False


class TestRlsContextSetGet:
    """Tests for setting/getting contextvars within a test."""

    def test_current_user_id_set_then_get_returns_value(self) -> None:
        # Arrange
        token = current_user_id.set("user-123")
        try:
            # Act & Assert
            assert current_user_id.get() == "user-123"
        finally:
            current_user_id.reset(token)
            assert current_user_id.get() is None

    def test_current_credential_set_then_get_returns_value(self) -> None:
        # Arrange
        token = current_credential.set("raw-jwt")
        try:
            # Act & Assert
            assert current_credential.get() == "raw-jwt"
        finally:
            current_credential.reset(token)
            assert current_credential.get() is None

    def test_current_auth_method_set_then_get_returns_value(self) -> None:
        # Arrange
        token = current_auth_method.set("jwt")
        try:
            # Act & Assert
            assert current_auth_method.get() == "jwt"
        finally:
            current_auth_method.reset(token)
            assert current_auth_method.get() is None


class TestSystemRlsContext:
    """Tests for the ``system_rls_context`` async context manager."""

    async def test_system_rls_context_sets_bypass_true_inside(self) -> None:
        # Act & Assert
        async with system_rls_context():
            assert bypass_rls.get() is True

    async def test_system_rls_context_resets_bypass_after_exit(self) -> None:
        # Arrange
        assert bypass_rls.get() is False

        # Act
        async with system_rls_context():
            assert bypass_rls.get() is True

        # Assert
        assert bypass_rls.get() is False

    async def test_system_rls_context_resets_even_on_exception(self) -> None:
        # Arrange
        assert bypass_rls.get() is False

        # Act & Assert
        with pytest.raises(RuntimeError, match="boom"):
            async with system_rls_context():
                assert bypass_rls.get() is True
                raise RuntimeError("boom")

        # Assert — bypass reset after the exception propagated
        assert bypass_rls.get() is False


class TestSetRlsContext:
    """Tests for the ``set_rls_context(conn, user_id)`` asyncpg helper."""

    async def test_sets_app_user_id_guc_when_user_id_is_set(self) -> None:
        # Arrange
        conn = AsyncMock()
        # Act
        await set_rls_context(conn, user_id="user-abc")

        # Assert — set_config('app.user_id', 'user-abc', true) emitted
        calls = conn.execute.await_args_list
        assert any(
            "set_config" in c.args[0].lower() and "app.user_id" in c.args[0].lower()
            for c in calls
        ), f"app.user_id GUC not set; calls={calls}"

    async def test_noop_when_user_id_is_none(self) -> None:
        # Arrange
        conn = AsyncMock()
        # Act
        await set_rls_context(conn, user_id=None)

        # Assert — no GUC issued
        conn.execute.assert_not_awaited()

    async def test_disables_row_security_when_bypass_rls_is_true(self) -> None:
        # Arrange — set bypass_rls contextvar
        token = bypass_rls.set(True)
        try:
            conn = AsyncMock()
            # Act
            await set_rls_context(conn, user_id="user-abc")

            # Assert — SET LOCAL row_security = off emitted
            calls = conn.execute.await_args_list
            assert any(
                "row_security" in c.args[0].lower() and "off" in c.args[0].lower()
                for c in calls
            ), f"row_security=off not emitted; calls={calls}"
        finally:
            bypass_rls.reset(token)
