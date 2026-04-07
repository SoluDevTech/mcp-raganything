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


@pytest.mark.asyncio
async def test_search_returns_results(mock_pool, mock_connection):
    """Search should return BM25SearchResult list."""
    adapter = PostgresBM25Adapter(db_url="postgresql://test")
    adapter._pool = mock_pool

    # Mock the pool.acquire context manager
    mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_connection)
    mock_pool.acquire.return_value.__exit__ = AsyncMock(return_value=None)

    # Mock database response with negative scores (pg_textsearch returns negative)
    mock_connection.fetch.return_value = [
        {
            "chunk_id": "123",
            "content": "PostgreSQL database system",
            "file_path": "/doc.pdf",
            "score": -2.345,
            "metadata": {"page": 1},
        }
    ]

    results = await adapter.search("PostgreSQL", "workspace1", top_k=5)

    assert len(results) == 1
    assert results[0].chunk_id == "123"
    assert results[0].content == "PostgreSQL database system"
    assert results[0].file_path == "/doc.pdf"
    assert results[0].score == 2.345  # Negative converted to positive
    assert results[0].metadata == {"page": 1}


@pytest.mark.asyncio
async def test_search_converts_negative_scores(mock_pool, mock_connection):
    """Search should convert negative BM25 scores to positive."""
    adapter = PostgresBM25Adapter(db_url="postgresql://test")
    adapter._pool = mock_pool
    mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_connection)
    mock_pool.acquire.return_value.__exit__ = AsyncMock(return_value=None)

    mock_connection.fetch.return_value = [
        {
            "chunk_id": "1",
            "content": "test",
            "file_path": "/t.pdf",
            "score": -5.0,
            "metadata": {},
        }
    ]

    results = await adapter.search("test", "ws", top_k=10)

    assert results[0].score == 5.0  # Negative converted to positive


@pytest.mark.asyncio
async def test_search_with_no_results(mock_pool, mock_connection):
    """Search should return empty list when no matches."""
    adapter = PostgresBM25Adapter(db_url="postgresql://test")
    adapter._pool = mock_pool
    mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_connection)
    mock_pool.acquire.return_value.__exit__ = AsyncMock(return_value=None)

    mock_connection.fetch.return_value = []

    results = await adapter.search("nonexistent", "workspace1", top_k=10)

    assert results == []


@pytest.mark.asyncio
async def test_index_document_executes_correct_sql(mock_pool, mock_connection):
    """Index document should execute correct INSERT/UPDATE SQL."""
    adapter = PostgresBM25Adapter(db_url="postgresql://test")
    adapter._pool = mock_pool
    mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_connection)
    mock_pool.acquire.return_value.__exit__ = AsyncMock(return_value=None)

    await adapter.index_document(
        chunk_id="123",
        content="test content",
        file_path="/doc.pdf",
        working_dir="workspace1",
        metadata={"page": 1},
    )

    # Verify SQL was executed
    mock_connection.execute.assert_called_once()
    call_args = mock_connection.execute.call_args[0]
    sql = call_args[0]
    assert "INSERT INTO chunks" in sql or "UPDATE chunks" in sql


@pytest.mark.asyncio
async def test_create_index_executes_correct_sql(mock_pool, mock_connection):
    """Create index should execute correct SQL."""
    adapter = PostgresBM25Adapter(db_url="postgresql://test")
    adapter._pool = mock_pool
    mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_connection)
    mock_pool.acquire.return_value.__exit__ = AsyncMock(return_value=None)

    await adapter.create_index("workspace1")

    mock_connection.execute.assert_called()


@pytest.mark.asyncio
async def test_drop_index_clears_tsvector(mock_pool, mock_connection):
    """Drop index should clear tsvector for workspace."""
    adapter = PostgresBM25Adapter(db_url="postgresql://test")
    adapter._pool = mock_pool
    mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_connection)
    mock_pool.acquire.return_value.__exit__ = AsyncMock(return_value=None)

    await adapter.drop_index("workspace1")

    # Verify SQL was executed
    mock_connection.execute.assert_called_once()
    call_args = mock_connection.execute.call_args[0]
    assert "UPDATE chunks" in call_args[0]
    assert "content_tsv = NULL" in call_args[0]
