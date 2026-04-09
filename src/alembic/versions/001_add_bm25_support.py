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

    # Guard: LightRAG creates lightrag_doc_chunks lazily on first use.
    # On a fresh database the table does not exist yet, so skip the
    # column/index/trigger steps.  They will be applied on next run
    # after LightRAG has created the table.
    op.execute(
        """
        DO $$
        BEGIN
            IF EXISTS (
                SELECT 1 FROM information_schema.tables
                WHERE table_name = 'lightrag_doc_chunks'
            ) THEN
                ALTER TABLE lightrag_doc_chunks
                    ADD COLUMN IF NOT EXISTS content_tsv tsvector;

                CREATE INDEX IF NOT EXISTS idx_lightrag_chunks_content_tsv
                    ON lightrag_doc_chunks USING GIN(content_tsv);

                DROP TRIGGER IF EXISTS trg_chunks_content_tsv
                    ON lightrag_doc_chunks;

                CREATE TRIGGER trg_chunks_content_tsv
                    BEFORE INSERT OR UPDATE ON lightrag_doc_chunks
                    FOR EACH ROW EXECUTE FUNCTION update_chunks_tsv();

                UPDATE lightrag_doc_chunks
                    SET content_tsv = to_tsvector('english', COALESCE(content, ''))
                    WHERE content_tsv IS NULL;
            END IF;
        END;
        $$
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

    op.execute("DROP TABLE IF EXISTS chunks")


def downgrade() -> None:
    """Remove BM25 support from lightrag_doc_chunks."""
    op.execute("DROP TRIGGER IF EXISTS trg_chunks_content_tsv ON lightrag_doc_chunks")
    op.execute("DROP FUNCTION IF EXISTS update_chunks_tsv()")
    for suffix in ("english", "french"):
        op.execute(f"DROP INDEX IF EXISTS idx_lightrag_chunks_bm25_{suffix}")
    op.execute("DROP INDEX IF EXISTS idx_lightrag_chunks_content_tsv")
    op.execute("ALTER TABLE lightrag_doc_chunks DROP COLUMN IF EXISTS content_tsv")
    op.execute("DROP EXTENSION IF EXISTS pg_textsearch")
