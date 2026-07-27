"""Add user_id column to mcp_servers.

Revision ID: 002
Revises: 001
Create Date: 2026-07-27

Adds a ``user_id`` column to the ``mcp_servers`` table so each registered MCP
server is owned by the user who created it. The column is backfilled with an
empty string for legacy rows (the dev/test path where no user is set), and a
btree index is created on ``user_id`` to speed up per-user listings.

This is the first half of the per-user isolation story for mcp-raganything:
the second half (``003_enable_rls_mcp_servers``) enables Row-Level Security on
the table so PostgreSQL itself filters rows by ``current_setting('app.user_id')``.
"""

from collections.abc import Sequence

from alembic import op

revision: str = "002"
down_revision: str | None = "001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute(
        """
        ALTER TABLE mcp_servers
            ADD COLUMN IF NOT EXISTS user_id VARCHAR(255) NOT NULL DEFAULT '';
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_mcp_servers_user_id
            ON mcp_servers (user_id);
        """
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_mcp_servers_user_id;")
    op.execute("ALTER TABLE mcp_servers DROP COLUMN IF EXISTS user_id;")
