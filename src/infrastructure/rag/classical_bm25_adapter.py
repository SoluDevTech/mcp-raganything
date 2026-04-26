"""PostgreSQL BM25 adapter for classical RAG tables.

Targets the classical_rag_* tables managed by LangchainPgvectorAdapter.
Uses pg_textsearch extension for BM25 ranking, same as PostgresBM25Adapter
but with a different table resolution strategy.
"""

import asyncio
import hashlib
import logging
import re as re_module
from typing import Any

import asyncpg

from domain.ports.bm25_engine import BM25EnginePort, BM25SearchResult

logger = logging.getLogger(__name__)

_SQL_IDENTIFIER_RE = re_module.compile(r"^[a-zA-Z0-9_]+$")
_TEXT_CONFIG_RE = re_module.compile(r"^[a-zA-Z0-9_]+$")


class ClassicalBM25Adapter(BM25EnginePort):
    """BM25 adapter for classical_rag_* tables (LangchainPgvector schema).

    Uses the same table name hashing as LangchainPgvectorAdapter so the
    BM25 search queries the same table that the vector store uses.
    """

    def __init__(self, db_url: str, table_prefix: str, text_config: str = "english"):
        if not _SQL_IDENTIFIER_RE.match(table_prefix):
            raise ValueError(f"Invalid table_prefix: {table_prefix!r}")
        if not _TEXT_CONFIG_RE.match(text_config):
            raise ValueError(f"Invalid text_config: {text_config!r}")
        self.db_url = db_url
        self.table_prefix = table_prefix
        self.text_config = text_config
        self._pool: asyncpg.Pool | None = None
        self._pool_lock = asyncio.Lock()
        self._ensured_tables: set[str] = set()

    def _get_table_name(self, working_dir: str) -> str:
        hashed = hashlib.sha256(working_dir.encode()).hexdigest()[:16]
        return f"{self.table_prefix}{hashed}"

    @staticmethod
    def _bm25_index_name(table_name: str, text_config: str) -> str:
        safe = table_name.replace("_", "")
        return f"idx_{safe}_bm25_{text_config}"

    async def _get_pool(self) -> asyncpg.Pool:
        if self._pool is not None:
            return self._pool
        async with self._pool_lock:
            if self._pool is not None:
                return self._pool
            self._pool = await asyncpg.create_pool(self.db_url)
            await self._check_extension()
        return self._pool

    async def _check_extension(self) -> None:
        async with self._pool.acquire() as conn:
            try:
                result = await conn.fetchval(
                    "SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname='pg_textsearch')"
                )
                if not result:
                    logger.warning(
                        "pg_textsearch extension not installed. "
                        "BM25 ranking will not work. "
                        "Run: CREATE EXTENSION pg_textsearch;"
                    )
            except Exception as e:
                logger.warning("Could not check pg_textsearch extension: %s", e)

    async def _ensure_bm25_index(self, table_name: str) -> None:
        if table_name in self._ensured_tables:
            return

        pool = await self._get_pool()
        index_name = self._bm25_index_name(table_name, self.text_config)

        try:
            async with pool.acquire() as conn:
                exists = await conn.fetchval(
                    "SELECT EXISTS(SELECT 1 FROM pg_tables WHERE tablename = $1)",
                    table_name,
                )
                if not exists:
                    logger.debug(
                        "Table %s does not exist yet, skipping BM25 index", table_name
                    )
                    return

                existing = await conn.fetchval(
                    "SELECT indexname FROM pg_indexes WHERE indexname = $1",
                    index_name,
                )
                if existing:
                    self._ensured_tables.add(table_name)
                    return

                await conn.execute(
                    f"""
                    CREATE INDEX {index_name}
                    ON {table_name} USING bm25(content)
                    WITH (text_config='{self.text_config}')
                    """
                )
                logger.info(
                    "Created BM25 index '%s' on %s with text_config='%s'",
                    index_name,
                    table_name,
                    self.text_config,
                )
                self._ensured_tables.add(table_name)
        except Exception as e:
            if "already exists" in str(e).lower():
                self._ensured_tables.add(table_name)
            else:
                logger.error("Failed to ensure BM25 index on %s: %s", table_name, e)

    async def search(
        self,
        query: str,
        working_dir: str,
        top_k: int = 10,
    ) -> list[BM25SearchResult]:
        table_name = self._get_table_name(working_dir)
        await self._ensure_bm25_index(table_name)

        pool = await self._get_pool()
        index_name = self._bm25_index_name(table_name, self.text_config)

        try:
            async with pool.acquire() as conn:
                exists = await conn.fetchval(
                    "SELECT EXISTS(SELECT 1 FROM pg_tables WHERE tablename = $1)",
                    table_name,
                )
                if not exists:
                    return []

                sql = f"""
                    SELECT
                        langchain_id AS chunk_id,
                        content,
                        langchain_metadata->>'file_path' AS file_path,
                        content <@> ag_catalog.to_bm25query($1, $3) as score
                    FROM {table_name}
                    WHERE content <@> ag_catalog.to_bm25query($1, $3) < 0
                    ORDER BY score
                    LIMIT $2
                """
                results = await conn.fetch(sql, query, top_k, index_name)

                return [
                    BM25SearchResult(
                        chunk_id=str(row["chunk_id"]),
                        content=row["content"],
                        file_path=row["file_path"] or "",
                        score=abs(row["score"]),
                        metadata={},
                    )
                    for row in results
                ]
        except Exception as e:
            logger.error(
                "BM25 search failed on %s: %s",
                table_name,
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
        pass

    async def create_index(self, working_dir: str) -> None:
        table_name = self._get_table_name(working_dir)
        await self._ensure_bm25_index(table_name)

    async def drop_index(self, working_dir: str) -> None:
        pool = await self._get_pool()
        table_name = self._get_table_name(working_dir)
        index_name = self._bm25_index_name(table_name, self.text_config)

        try:
            async with pool.acquire() as conn:
                await conn.execute(f"DROP INDEX IF EXISTS {index_name}")
            self._ensured_tables.discard(table_name)
        except Exception as e:
            logger.error("BM25 index drop failed on %s: %s", table_name, e)
            raise

    async def close(self) -> None:
        if self._pool:
            await self._pool.close()
            self._pool = None
