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
    """Add tsvector column, indexes, and trigger for BM25 search."""
    # Add tsvector column
    op.execute("ALTER TABLE chunks ADD COLUMN IF NOT EXISTS content_tsv tsvector")

    # Create GIN index for tsvector
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_chunks_content_tsv ON chunks USING GIN(content_tsv)"
    )

    # Create BM25 index (conditional on pg_textsearch extension)
    op.execute(
        """
        DO $$
        BEGIN
            IF EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'pg_textsearch') THEN
                CREATE INDEX IF NOT EXISTS idx_chunks_bm25
                ON chunks USING bm25(content)
                WITH (text_config='english');
            END IF;
        END $$;
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

    # Backfill existing documents
    op.execute(
        "UPDATE chunks SET content_tsv = to_tsvector('english', COALESCE(content, '')) WHERE content_tsv IS NULL"
    )


def downgrade() -> None:
    """Remove BM25 support."""
    op.execute("DROP TRIGGER IF EXISTS trg_chunks_content_tsv ON chunks")
    op.execute("DROP FUNCTION IF EXISTS update_chunks_tsv()")
    op.execute("DROP INDEX IF EXISTS idx_chunks_bm25")
    op.execute("DROP INDEX IF EXISTS idx_chunks_content_tsv")
    op.execute("ALTER TABLE chunks DROP COLUMN IF EXISTS content_tsv")