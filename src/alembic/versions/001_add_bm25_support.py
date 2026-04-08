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
    """Add BM25 full-text search to lightrag_doc_chunks."""
    op.execute("CREATE EXTENSION IF NOT EXISTS pg_textsearch")

    op.execute(
        "ALTER TABLE lightrag_doc_chunks ADD COLUMN IF NOT EXISTS content_tsv tsvector"
    )

    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_lightrag_chunks_content_tsv ON lightrag_doc_chunks USING GIN(content_tsv)"
    )

    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_lightrag_chunks_bm25
        ON lightrag_doc_chunks USING bm25(content)
        WITH (text_config='english')
    """
    )

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

    op.execute("DROP TRIGGER IF EXISTS trg_chunks_content_tsv ON lightrag_doc_chunks")
    op.execute(
        """
        CREATE TRIGGER trg_chunks_content_tsv
        BEFORE INSERT OR UPDATE ON lightrag_doc_chunks
        FOR EACH ROW EXECUTE FUNCTION update_chunks_tsv();
    """
    )

    # WARNING: This UPDATE scans the entire table. For tables with >100K rows,
    # consider running as a separate manual batch operation instead.
    op.execute(
        "UPDATE lightrag_doc_chunks SET content_tsv = to_tsvector('english', COALESCE(content, '')) WHERE content_tsv IS NULL"
    )

    op.execute("DROP TABLE IF EXISTS chunks")


def downgrade() -> None:
    """Remove BM25 support from lightrag_doc_chunks."""
    op.execute("DROP TRIGGER IF EXISTS trg_chunks_content_tsv ON lightrag_doc_chunks")
    op.execute("DROP FUNCTION IF EXISTS update_chunks_tsv()")
    op.execute("DROP INDEX IF EXISTS idx_lightrag_chunks_bm25")
    op.execute("DROP INDEX IF EXISTS idx_lightrag_chunks_content_tsv")
    op.execute("ALTER TABLE lightrag_doc_chunks DROP COLUMN IF EXISTS content_tsv")
    op.execute("DROP EXTENSION IF EXISTS pg_textsearch")
