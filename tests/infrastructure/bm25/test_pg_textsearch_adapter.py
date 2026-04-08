"""Tests for PostgresBM25Adapter implementation."""

from unittest.mock import AsyncMock

import asyncpg
import pytest

from infrastructure.bm25.pg_textsearch_adapter import PostgresBM25Adapter


@pytest.fixture
def mock_pool():
    """Create mock asyncpg pool."""
    pool = AsyncMock(spec=asyncpg.Pool)
    return pool


@pytest.fixture
def mock_connection():
    """Create mock asyncpg connection."""
    conn = AsyncMock(spec=asyncpg.Connection)
    return conn


def _acquire_mock(conn):
    """Helper to create an async context manager for pool.acquire()."""
    cm = AsyncMock()
    cm.__aenter__ = AsyncMock(return_value=conn)
    cm.__aexit__ = AsyncMock(return_value=None)
    return cm


class TestMakeWorkspace:
    """Tests for _make_workspace static method."""

    def test_make_workspace_produces_ws_prefix(self):
        """Should produce workspace with ws_ prefix."""
        result = PostgresBM25Adapter._make_workspace(
            "36ecc1eb-dead-4000-beef-1234567890ab"
        )
        assert result.startswith("ws_")

    def test_make_workspace_is_deterministic(self):
        """Same input should always produce same workspace."""
        result1 = PostgresBM25Adapter._make_workspace("test-working-dir")
        result2 = PostgresBM25Adapter._make_workspace("test-working-dir")
        assert result1 == result2

    def test_make_workspace_different_inputs_different_outputs(self):
        """Different inputs should produce different workspaces."""
        result1 = PostgresBM25Adapter._make_workspace("dir-a")
        result2 = PostgresBM25Adapter._make_workspace("dir-b")
        assert result1 != result2


class TestTextConfig:
    """Tests for text_config and BM25 index naming."""

    def test_default_text_config_is_english(self):
        adapter = PostgresBM25Adapter(db_url="postgresql://test")
        assert adapter.text_config == "english"

    def test_custom_text_config(self):
        adapter = PostgresBM25Adapter(db_url="postgresql://test", text_config="french")
        assert adapter.text_config == "french"

    def test_bm25_index_name_includes_text_config(self):
        adapter = PostgresBM25Adapter(db_url="postgresql://test", text_config="french")
        assert adapter.bm25_index_name == "idx_lightrag_chunks_bm25_french"

    def test_bm25_index_name_english(self):
        adapter = PostgresBM25Adapter(db_url="postgresql://test", text_config="english")
        assert adapter.bm25_index_name == "idx_lightrag_chunks_bm25_english"


class TestSearch:
    @pytest.mark.asyncio
    async def test_search_returns_results(self, mock_pool, mock_connection):
        """Search should return BM25SearchResult list."""
        adapter = PostgresBM25Adapter(db_url="postgresql://test")
        adapter._pool = mock_pool
        mock_pool.acquire.return_value = _acquire_mock(mock_connection)

        mock_connection.fetch.return_value = [
            {
                "chunk_id": "123",
                "content": "PostgreSQL database system",
                "file_path": "/doc.pdf",
                "score": -2.345,
            }
        ]

        results = await adapter.search("PostgreSQL", "workspace1", top_k=5)

        assert len(results) == 1
        assert results[0].chunk_id == "123"
        assert results[0].content == "PostgreSQL database system"
        assert results[0].file_path == "/doc.pdf"
        assert results[0].score == 2.345

    @pytest.mark.asyncio
    async def test_search_converts_negative_scores(self, mock_pool, mock_connection):
        """Search should convert negative BM25 scores to positive."""
        adapter = PostgresBM25Adapter(db_url="postgresql://test")
        adapter._pool = mock_pool
        mock_pool.acquire.return_value = _acquire_mock(mock_connection)

        mock_connection.fetch.return_value = [
            {
                "chunk_id": "1",
                "content": "test",
                "file_path": "/t.pdf",
                "score": -5.0,
            }
        ]

        results = await adapter.search("test", "ws", top_k=10)
        assert results[0].score == 5.0

    @pytest.mark.asyncio
    async def test_search_with_no_results(self, mock_pool, mock_connection):
        """Search should return empty list when no matches."""
        adapter = PostgresBM25Adapter(db_url="postgresql://test")
        adapter._pool = mock_pool
        mock_pool.acquire.return_value = _acquire_mock(mock_connection)

        mock_connection.fetch.return_value = []

        results = await adapter.search("nonexistent", "workspace1", top_k=10)
        assert results == []

    @pytest.mark.asyncio
    async def test_search_queries_lightrag_doc_chunks(self, mock_pool, mock_connection):
        """Search should query lightrag_doc_chunks with workspace mapping."""
        adapter = PostgresBM25Adapter(db_url="postgresql://test")
        adapter._pool = mock_pool
        mock_pool.acquire.return_value = _acquire_mock(mock_connection)

        mock_connection.fetch.return_value = []

        await adapter.search("test query", "some-working-dir", top_k=5)

        sql = mock_connection.fetch.call_args[0][0]
        assert "lightrag_doc_chunks" in sql
        assert "workspace" in sql

        workspace_arg = mock_connection.fetch.call_args[0][2]
        assert workspace_arg == PostgresBM25Adapter._make_workspace("some-working-dir")

    @pytest.mark.asyncio
    async def test_search_uses_bm25_index_with_text_config(self, mock_pool, mock_connection):
        """Search should use text_config-specific BM25 index."""
        adapter = PostgresBM25Adapter(db_url="postgresql://test", text_config="french")
        adapter._pool = mock_pool
        mock_pool.acquire.return_value = _acquire_mock(mock_connection)

        mock_connection.fetch.return_value = []

        await adapter.search("test query", "some-working-dir", top_k=5)

        bm25_index_arg = mock_connection.fetch.call_args[0][3]
        assert bm25_index_arg == "idx_lightrag_chunks_bm25_french"


class TestIndexDocument:
    @pytest.mark.asyncio
    async def test_index_document_is_noop(self, mock_pool, mock_connection):
        """Index document should be a no-op since LightRAG owns the table."""
        adapter = PostgresBM25Adapter(db_url="postgresql://test")
        adapter._pool = mock_pool

        await adapter.index_document(
            chunk_id="123",
            content="test content",
            file_path="/doc.pdf",
            working_dir="workspace1",
            metadata={"page": 1},
        )

        mock_pool.acquire.assert_not_called()
        mock_connection.execute.assert_not_called()


class TestCreateIndex:
    @pytest.mark.asyncio
    async def test_create_index_updates_lightrag_doc_chunks(
        self, mock_pool, mock_connection
    ):
        """Create index should update lightrag_doc_chunks tsvector."""
        adapter = PostgresBM25Adapter(db_url="postgresql://test", text_config="french")
        adapter._pool = mock_pool
        mock_pool.acquire.return_value = _acquire_mock(mock_connection)

        await adapter.create_index("some-working-dir")

        mock_connection.execute.assert_called_once()
        sql = mock_connection.execute.call_args[0][0]
        assert "lightrag_doc_chunks" in sql
        assert "content_tsv" in sql
        assert "french" in sql

        workspace_arg = mock_connection.execute.call_args[0][1]
        assert workspace_arg == PostgresBM25Adapter._make_workspace("some-working-dir")


class TestDropIndex:
    @pytest.mark.asyncio
    async def test_drop_index_clears_tsvector(self, mock_pool, mock_connection):
        """Drop index should clear tsvector for workspace."""
        adapter = PostgresBM25Adapter(db_url="postgresql://test")
        adapter._pool = mock_pool
        mock_pool.acquire.return_value = _acquire_mock(mock_connection)

        await adapter.drop_index("workspace1")

        mock_connection.execute.assert_called_once()
        sql = mock_connection.execute.call_args[0][0]
        assert "lightrag_doc_chunks" in sql
        assert "content_tsv = NULL" in sql

    @pytest.mark.asyncio
    async def test_drop_index_uses_workspace_mapping(self, mock_pool, mock_connection):
        """Drop index should map working_dir to workspace."""
        adapter = PostgresBM25Adapter(db_url="postgresql://test")
        adapter._pool = mock_pool
        mock_pool.acquire.return_value = _acquire_mock(mock_connection)

        await adapter.drop_index("my-working-dir")

        workspace_arg = mock_connection.execute.call_args[0][1]
        assert workspace_arg == PostgresBM25Adapter._make_workspace("my-working-dir")


class TestClose:
    @pytest.mark.asyncio
    async def test_close_closes_pool(self, mock_pool):
        """Close should close connection pool."""
        adapter = PostgresBM25Adapter(db_url="postgresql://test")
        adapter._pool = mock_pool

        await adapter.close()

        mock_pool.close.assert_called_once()
        assert adapter._pool is None

    @pytest.mark.asyncio
    async def test_close_with_no_pool(self):
        """Close should handle None pool gracefully."""
        adapter = PostgresBM25Adapter(db_url="postgresql://test")
        adapter._pool = None

        await adapter.close()

        assert adapter._pool is None
