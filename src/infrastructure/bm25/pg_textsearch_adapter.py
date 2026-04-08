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

    def __init__(self, db_url: str):
        self.db_url = db_url
        self._pool: asyncpg.Pool | None = None
        self._pool_lock = asyncio.Lock()

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
            except Exception as e:
                logger.warning("Could not check pg_textsearch extension: %s", e)

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

        try:
            async with pool.acquire() as conn:
                sql = """
                    SELECT
                        id AS chunk_id,
                        content,
                        file_path,
                        content <@> to_bm25query($1, 'idx_lightrag_chunks_bm25') as score
                    FROM lightrag_doc_chunks
                    WHERE workspace = $2
                      AND content_tsv @@ plainto_tsquery('english', $1)
                      AND content <@> to_bm25query($1, 'idx_lightrag_chunks_bm25') < 0
                    ORDER BY score
                    LIMIT $3
                """
                results = await conn.fetch(sql, query, workspace, top_k)

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
                    """
                    UPDATE lightrag_doc_chunks
                    SET content_tsv = to_tsvector('english', COALESCE(content, ''))
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
