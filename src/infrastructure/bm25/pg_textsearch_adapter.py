"""PostgreSQL BM25 adapter using pg_textsearch extension."""

import asyncio
import hashlib
import logging
from typing import Any

import asyncpg

from domain.ports.bm25_engine import BM25EnginePort, BM25SearchResult

logger = logging.getLogger(__name__)


class PostgresBM25Adapter(BM25EnginePort):
    """PostgreSQL BM25 implementation using pg_textsearch.

    Queries the lightrag_doc_chunks table directly, using the same
    workspace mapping as LightRAGAdapter (_make_workspace).
    """

    _BM25_INDEX_PREFIX = "idx_lightrag_chunks_bm25"

    def __init__(self, db_url: str, text_config: str = "english"):
        self.db_url = db_url
        self.text_config = text_config
        self._pool: asyncpg.Pool | None = None
        self._pool_lock = asyncio.Lock()

    @property
    def bm25_index_name(self) -> str:
        return f"{self._BM25_INDEX_PREFIX}_{self.text_config}"

    @staticmethod
    def _make_workspace(working_dir: str) -> str:
        """Map working_dir to lightrag_doc_chunks.workspace (same as LightRAGAdapter)."""
        digest = hashlib.sha256(working_dir.encode()).hexdigest()[:16]
        return f"ws_{digest}"

    async def _get_pool(self) -> asyncpg.Pool:
        """Get or create database connection pool with double-checked locking."""
        if self._pool is not None:
            return self._pool
        async with self._pool_lock:
            if self._pool is not None:
                return self._pool
            self._pool = await asyncpg.create_pool(self.db_url)
            await self._check_extension()
        return self._pool

    async def _check_extension(self) -> None:
        """Warn if pg_textsearch extension is not installed."""
        async with self._pool.acquire() as conn:
            try:
                result = await conn.fetchval(
                    "SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname='pg_textsearch')"
                )
                if not result:
                    logger.warning(
                        "pg_textsearch extension not installed. "
                        "BM25 ranking <@> operator will not work. "
                        "Run: CREATE EXTENSION pg_textsearch;"
                    )
                    return
            except Exception as e:
                logger.warning("Could not check pg_textsearch extension: %s", e)
                return

            await self._ensure_bm25_index(conn)
            await self._rebuild_tsv_if_config_changed(conn)

    async def _ensure_bm25_index(self, conn) -> None:
        """Create or recreate the BM25 index for the configured text_config.

        Drops any stale BM25 index from a different text_config.
        """
        index_name = self.bm25_index_name
        try:
            existing = await conn.fetchval(
                "SELECT indexname FROM pg_indexes WHERE indexname = $1",
                index_name,
            )
            if existing:
                logger.info(
                    "BM25 index '%s' already exists for text_config='%s'",
                    index_name,
                    self.text_config,
                )
                return

            for suffix in ("english", "french"):
                stale = f"{self._BM25_INDEX_PREFIX}_{suffix}"
                if stale != index_name:
                    await conn.execute(f"DROP INDEX IF EXISTS {stale}")

            await conn.execute(
                f"""
                CREATE INDEX {index_name}
                ON lightrag_doc_chunks USING bm25(content)
                WITH (text_config='{self.text_config}')
                """
            )
            logger.info(
                "Created BM25 index '%s' with text_config='%s'",
                index_name,
                self.text_config,
            )
        except Exception as e:
            logger.error("Failed to ensure BM25 index: %s", e)

    async def _rebuild_tsv_if_config_changed(self, conn) -> None:
        """Rebuild content_tsv if trigger function uses a different text_config."""
        try:
            func_def = await conn.fetchval(
                "SELECT prosrc FROM pg_proc WHERE proname = 'update_chunks_tsv'"
            )
            if func_def and f"'{self.text_config}'" not in func_def:
                logger.info(
                    "Updating trigger function from old text_config to '%s'",
                    self.text_config,
                )
                await conn.execute(
                    f"""
                    CREATE OR REPLACE FUNCTION update_chunks_tsv()
                    RETURNS TRIGGER AS $$
                    BEGIN
                        NEW.content_tsv := to_tsvector('{self.text_config}', COALESCE(NEW.content, ''));
                        RETURN NEW;
                    END;
                    $$ LANGUAGE plpgsql;
                    """
                )
                status = await conn.execute(
                    f"""
                    UPDATE lightrag_doc_chunks
                    SET content_tsv = to_tsvector('{self.text_config}', COALESCE(content, ''))
                    WHERE content_tsv IS NOT NULL
                    """
                )
                logger.info("Rebuilt content_tsv: %s with text_config='%s'", status, self.text_config)
        except Exception as e:
            logger.warning("Could not check/rebuild trigger function: %s", e)

    async def close(self) -> None:
        """Close connection pool on shutdown."""
        if self._pool:
            await self._pool.close()
            self._pool = None

    async def search(
        self,
        query: str,
        working_dir: str,
        top_k: int = 10,
    ) -> list[BM25SearchResult]:
        """Search using BM25 ranking on lightrag_doc_chunks."""
        pool = await self._get_pool()
        workspace = self._make_workspace(working_dir)
        bm25_index = f"idx_lightrag_chunks_bm25_{self.text_config}"

        try:
            async with pool.acquire() as conn:
                sql = """
                    SELECT
                        id AS chunk_id,
                        content,
                        file_path,
                        content <@> to_bm25query($1, $3) as score
                    FROM lightrag_doc_chunks
                    WHERE workspace = $2
                      AND content <@> to_bm25query($1, $3) < 0
                    ORDER BY score
                    LIMIT $4
                """
                results = await conn.fetch(
                    sql, query, workspace, bm25_index, top_k
                )

                return [
                    BM25SearchResult(
                        chunk_id=row["chunk_id"],
                        content=row["content"],
                        file_path=row["file_path"],
                        score=abs(row["score"]),
                        metadata={},
                    )
                    for row in results
                ]
        except Exception as e:
            logger.error(
                "BM25 search failed: %s",
                e,
                extra={"query": query, "working_dir": working_dir},
            )
            raise

    async def index_document(
        self,
        chunk_id: str,
        content: str,
        file_path: str,
        working_dir: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """No-op: LightRAG owns the lightrag_doc_chunks table."""
        pass

    async def create_index(self, working_dir: str) -> None:
        """Re-index tsvector for workspace chunks."""
        pool = await self._get_pool()
        workspace = self._make_workspace(working_dir)

        try:
            async with pool.acquire() as conn:
                await conn.execute(
                    f"""
                    UPDATE lightrag_doc_chunks
                    SET content_tsv = to_tsvector('{self.text_config}', COALESCE(content, ''))
                    WHERE workspace = $1 AND content_tsv IS NULL
                    """,
                    workspace,
                )
        except Exception as e:
            logger.error(
                "BM25 index creation failed: %s", e, extra={"working_dir": working_dir}
            )
            raise

    async def drop_index(self, working_dir: str) -> None:
        """Clear tsvector for workspace chunks."""
        pool = await self._get_pool()
        workspace = self._make_workspace(working_dir)

        try:
            async with pool.acquire() as conn:
                await conn.execute(
                    "UPDATE lightrag_doc_chunks SET content_tsv = NULL WHERE workspace = $1",
                    workspace,
                )
        except Exception as e:
            logger.error(
                "BM25 index drop failed: %s", e, extra={"working_dir": working_dir}
            )
            raise
