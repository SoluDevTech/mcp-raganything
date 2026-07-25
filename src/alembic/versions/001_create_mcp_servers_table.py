"""Create mcp_servers table.

Revision ID: 001
Revises:
Create Date: 2026-07-25

Stores registered MCP servers keyed by name. The ``headers``, ``env`` and
``auth_token`` columns store Fernet-encrypted ciphertext (the
:class:`McpRegistryStore` adapter encrypts on write and decrypts on read via the
injected :class:`SecretCipher`).

This is the single source of truth for the ``mcp_servers`` schema — mcp-raganything
owns the registry service and its table. ``CREATE TABLE IF NOT EXISTS`` makes the
migration safe on databases where the table was previously created by
composable-agents' legacy migrations 008/009 (now removed from that brick).
"""

from collections.abc import Sequence

from alembic import op

revision: str = "001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS mcp_servers (
            name                  VARCHAR(100) PRIMARY KEY,
            transport             VARCHAR(20)  NOT NULL DEFAULT 'http',
            url                   VARCHAR(500) NOT NULL,
            headers_encrypted     TEXT         NOT NULL DEFAULT '{}',
            env_encrypted         TEXT         NOT NULL DEFAULT '{}',
            auth_token_encrypted  TEXT,
            tool_count            INTEGER      NOT NULL DEFAULT 0,
            created_at            TIMESTAMPTZ  NOT NULL DEFAULT now(),
            updated_at            TIMESTAMPTZ  NOT NULL DEFAULT now(),
            source_type           VARCHAR(20)  NOT NULL DEFAULT 'external',
            openapi_url           VARCHAR(500) NULL
        );
        """
    )


def downgrade() -> None:
    op.drop_table("mcp_servers")
