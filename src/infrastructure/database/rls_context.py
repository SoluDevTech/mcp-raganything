"""Context variables for per-request RLS isolation (asyncpg-based).

These contextvars are set by ``ComposableAgentsSecurity.verify_credentials``
after a successful authentication and consumed by the ``set_rls_context``
helper (and ``McpRegistryStore``) to set PostgreSQL GUCs (``app.user_id``)
on raw asyncpg connections so that Row-Level Security policies can filter rows
per authenticated user.

``current_credential`` holds the raw credential (JWT token or API key) for
audit logging without re-reading the request headers.

``current_auth_method`` records which authentication method produced the
context (``"jwt"`` or ``"api_key"``).

``bypass_rls`` is set to ``True`` by the ``system_rls_context`` async context
manager so that background jobs (cron, migrations) and privileged auth reads
(e.g. the ``AsyncpgApiKeyReader`` reading the FORCE RLS-protected ``api_keys``
table before the user is known) can read across all users without an
authenticated principal.
"""

import contextvars
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

current_user_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "current_user_id", default=None
)
current_credential: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "current_credential", default=None
)
# Authentication method that produced the current context ("jwt" or "api_key").
current_auth_method: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "current_auth_method", default=None
)
# When True, the RLS helper emits ``SET LOCAL row_security = off`` so that
# system/migration queries and privileged auth reads can read across all users.
bypass_rls: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "bypass_rls", default=False
)


@asynccontextmanager
async def system_rls_context() -> AsyncIterator[None]:
    """Temporarily disable RLS for the duration of a system / migration block.

    Background jobs and migrations run without an authenticated user and
    therefore have no ``current_user_id``. Without this context, RLS policies
    would filter out every row (``user_id = NULL`` is always FALSE).

    Usage::

        async with system_rls_context():
            await run_migrations()

    The ``bypass_rls`` contextvar is reset on exit, including when an exception
    propagates out of the ``with`` block.
    """
    token = bypass_rls.set(True)
    try:
        yield
    finally:
        bypass_rls.reset(token)


async def set_rls_context(conn: Any, user_id: str | None) -> None:
    """Set the PostgreSQL RLS GUC ``app.user_id`` on a raw asyncpg connection.

    Called by ``McpRegistryStore`` (and any other asyncpg usage) right after
    acquiring a connection so that Row-Level Security policies on
    ``mcp_servers`` filter rows per authenticated user.

    Behaviour:
    - When ``user_id`` is set and ``bypass_rls`` is False → emit
      ``SELECT set_config('app.user_id', $user_id, true)`` so RLS policies
      using ``current_setting('app.user_id', true)`` see the user.
    - When ``user_id`` is set and ``bypass_rls`` is True → emit
      ``SET LOCAL row_security = off`` instead (privileged/system mode).
    - When ``user_id`` is None → no-op (legacy/tests path; RLS not engaged).

    Args:
        conn: A raw asyncpg connection (already acquired from the pool).
        user_id: The authenticated user_id (from the RLS contextvar). None
            disables the GUC emission.
    """
    if user_id is None:
        return

    if bypass_rls.get():
        # Privileged/system mode: bypass RLS entirely for this connection.
        await conn.execute("SET LOCAL row_security = off")
    else:
        # Per-user mode: set the GUC so RLS policies filter rows.
        # ``true`` = local (transaction-scoped) setting.
        await conn.execute("SELECT set_config('app.user_id', $1, true)", user_id)
