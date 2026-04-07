"""PostgreSQL BM25 adapter using pg_textsearch extension."""

import asyncio
import logging
from typing import Any

import asyncpg

from domain.ports.bm25_engine import BM25EnginePort, BM25SearchResult

logger = logging.getLogger(__name__)


class PostgresBM25Adapter(BM25EnginePort):
    """PostgreSQL BM25 implementation using pg_textsearch.

    Uses PostgreSQL native full-text search with tsvector/tsquery
    and pg_textsearch extension for BM25-style ranking.

    The <@> operator returns negative scores (lower is better),
    so we convert to positive for consistency.
    """

    def __init__(self, db_url: str):
        self.db_url = db_url
        self._pool: asyncpg.Pool | None = None
        self._pool_lock = asyncio.Lock()

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
        """Search using BM25 ranking.

        Uses pg_textsearch <@> operator for BM25 scoring.
        Scores are negative (lower is better), converted to positive.

        Args:
            query: Search query string
            working_dir: Project/workspace directory
            top_k: Number of results to return

        Returns:
            List of BM25SearchResult ordered by relevance
        """
        pool = await self._get_pool()

        try:
            async with pool.acquire() as conn:
                sql = """
                    SELECT
                        chunk_id,
                        content,
                        file_path,
                        content <@> websearch_to_tsquery('english', $1) as score,
                        metadata
                    FROM chunks
                    WHERE working_dir = $2
                      AND content_tsv @@ websearch_to_tsquery('english', $1)
                    ORDER BY score
                    LIMIT $3
                """
                results = await conn.fetch(sql, query, working_dir, top_k)

                return [
                    BM25SearchResult(
                        chunk_id=row["chunk_id"],
                        content=row["content"],
                        file_path=row["file_path"],
                        score=abs(row["score"]),
                        metadata=row["metadata"] or {},
                    )
                    for row in results
                ]
        except Exception as e:
            logger.error("BM25 search failed: %s", e, extra={"query": query, "working_dir": working_dir})
            raise

    async def index_document(
        self,
        chunk_id: str,
        content: str,
        file_path: str,
        working_dir: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Index document chunk.

        The tsvector column is auto-updated via trigger,
        so we only need to INSERT/UPDATE the row.

        Args:
            chunk_id: Unique chunk identifier
            content: Text content to index
            file_path: Path to source file
            working_dir: Project/workspace directory
            metadata: Optional metadata dictionary
        """
        pool = await self._get_pool()

        try:
            async with pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO chunks (chunk_id, content, file_path, working_dir, metadata)
                    VALUES ($1, $2, $3, $4, $5)
                    ON CONFLICT (chunk_id) DO UPDATE SET
                        content = EXCLUDED.content,
                        file_path = EXCLUDED.file_path,
                        metadata = EXCLUDED.metadata
                    """,
                    chunk_id,
                    content,
                    file_path,
                    working_dir,
                    metadata or {},
                )
        except Exception as e:
            logger.error("BM25 document indexing failed: %s", e, extra={"chunk_id": chunk_id})
            raise

    async def create_index(self, working_dir: str) -> None:
        """Create BM25 index for workspace.

        The index is auto-updated via trigger; this method is for explicit re-indexing.

        Args:
            working_dir: Project/workspace directory
        """
        pool = await self._get_pool()

        try:
            async with pool.acquire() as conn:
                await conn.execute(
                    """
                    UPDATE chunks
                    SET content_tsv = to_tsvector('english', content)
                    WHERE working_dir = $1 AND content_tsv IS NULL
                    """,
                    working_dir,
                )
        except Exception as e:
            logger.error("BM25 index creation failed: %s", e, extra={"working_dir": working_dir})
            raise

    async def drop_index(self, working_dir: str) -> None:
        """Drop BM25 index for workspace."""
        pool = await self._get_pool()

        try:
            async with pool.acquire() as conn:
                await conn.execute(
                    "UPDATE chunks SET content_tsv = NULL WHERE working_dir = $1",
                    working_dir,
                )
        except Exception as e:
            logger.error("BM25 index drop failed: %s", e, extra={"working_dir": working_dir})
            raise
