"""Add BM25 support via pg_textsearch

Revision ID: 001
Revises:
Create Date: 2026-04-07

"""

from collections.abc import Sequence

from alembic import op

revision: str = "001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add BM25 chunks table with tsvector column, indexes, and trigger."""
    # Create pg_textsearch extension (requires shared_preload_libraries in postgresql.conf)
    op.execute("CREATE EXTENSION IF NOT EXISTS pg_textsearch")

    # Create chunks table (used by BM25 adapter for full-text search)
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS chunks (
            chunk_id VARCHAR(255) PRIMARY KEY,
            content TEXT NOT NULL,
            file_path TEXT NOT NULL,
            working_dir VARCHAR(512) NOT NULL,
            metadata JSONB DEFAULT '{}',
            content_tsv tsvector
        )
    """
    )

    # Create GIN index for tsvector
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_chunks_content_tsv ON chunks USING GIN(content_tsv)"
    )

    # Create index on working_dir for filtering
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_chunks_working_dir ON chunks(working_dir)"
    )

    # Create BM25 index using pg_textsearch
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_chunks_bm25
        ON chunks USING bm25(content)
        WITH (text_config='english')
    """
    )

    # Create auto-update trigger function
    op.execute(
        """
        CREATE OR REPLACE FUNCTION update_chunks_tsv()
        RETURNS TRIGGER AS $$
        BEGIN
            NEW.content_tsv := to_tsvector('english', COALESCE(NEW.content, ''));
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql;
    """
    )

    # Create trigger
    op.execute("DROP TRIGGER IF EXISTS trg_chunks_content_tsv ON chunks")
    op.execute(
        """
        CREATE TRIGGER trg_chunks_content_tsv
        BEFORE INSERT OR UPDATE ON chunks
        FOR EACH ROW EXECUTE FUNCTION update_chunks_tsv();
    """
    )


def downgrade() -> None:
    """Remove BM25 support."""
    op.execute("DROP TABLE IF EXISTS chunks")
    op.execute("DROP FUNCTION IF EXISTS update_chunks_tsv()")
    op.execute("DROP EXTENSION IF EXISTS pg_textsearch")
