"""Enable Row-Level Security on mcp_servers.

Revision ID: 003
Revises: 002
Create Date: 2026-07-27

Enables and FORCES Row-Level Security on the ``mcp_servers`` table so that
PostgreSQL itself filters rows by ``current_setting('app.user_id', true)``.
The application sets the ``app.user_id`` GUC on each acquired asyncpg
connection (via ``set_rls_context`` in ``McpRegistryStore``) after a successful
dual-auth.

FORCE is used so that even the table owner (the role mcp-raganything connects
as) is subject to the policy — this is required because mcp-raganything shares
the database with composable-agents and the connecting role is the owner of
its own table. Without FORCE, owners bypass RLS by default.

A bypass is provided for system/migration queries via
``system_rls_context`` (which sets the ``bypass_rls`` contextvar to True) and
the ``set_rls_context`` helper (which then emits ``SET LOCAL row_security =
off``). The Alembic migration itself runs as the table owner and would bypass
RLS naturally, but the policy is written to allow ``user_id = ''`` (the legacy
default) so dev/test rows without a user are still visible.
"""

from collections.abc import Sequence

from alembic import op

revision: str = "003"
down_revision: str | None = "002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("ALTER TABLE mcp_servers ENABLE ROW LEVEL SECURITY;")
    op.execute("ALTER TABLE mcp_servers FORCE ROW LEVEL SECURITY;")
    op.execute(
        """
        CREATE POLICY mcp_servers_user_isolation
            ON mcp_servers
            USING (user_id = current_setting('app.user_id', true))
            WITH CHECK (user_id = current_setting('app.user_id', true));
        """
    )


def downgrade() -> None:
    op.execute("DROP POLICY IF EXISTS mcp_servers_user_isolation ON mcp_servers;")
    op.execute("ALTER TABLE mcp_servers NO FORCE ROW LEVEL SECURITY;")
    op.execute("ALTER TABLE mcp_servers DISABLE ROW LEVEL SECURITY;")
