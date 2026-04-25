"""Tests for ClassicalBM25Adapter."""

import hashlib
from unittest.mock import AsyncMock, patch

import pytest

from infrastructure.rag.classical_bm25_adapter import ClassicalBM25Adapter


class TestClassicalBM25Adapter:
    def test_get_table_name_hashes_working_dir(self) -> None:
        adapter = ClassicalBM25Adapter(
            db_url="postgresql://u:p@localhost/db",
            table_prefix="classical_rag_",
        )
        hashed = hashlib.sha256(b"/tmp/rag/project").hexdigest()[:16]
        assert adapter._get_table_name("/tmp/rag/project") == f"classical_rag_{hashed}"

    def test_get_table_name_different_prefix(self) -> None:
        adapter = ClassicalBM25Adapter(
            db_url="postgresql://u:p@localhost/db",
            table_prefix="my_prefix_",
        )
        hashed = hashlib.sha256(b"/tmp/test").hexdigest()[:16]
        assert adapter._get_table_name("/tmp/test") == f"my_prefix_{hashed}"

    def test_bm25_index_name(self) -> None:
        adapter = ClassicalBM25Adapter(
            db_url="postgresql://u:p@localhost/db",
            table_prefix="classical_rag_",
            text_config="english",
        )
        table_name = adapter._get_table_name("/tmp/rag/project")
        index_name = ClassicalBM25Adapter._bm25_index_name(table_name, "english")
        assert "classicalrag" in index_name
        assert "english" in index_name
        assert index_name.startswith("idx_")

    @pytest.mark.asyncio
    async def test_search_returns_empty_when_table_not_exists(self) -> None:
        adapter = ClassicalBM25Adapter(
            db_url="postgresql://u:p@localhost/db",
            table_prefix="classical_rag_",
        )
        from contextlib import asynccontextmanager

        mock_conn = AsyncMock()
        mock_conn.fetchval.return_value = False

        @asynccontextmanager
        async def mock_acquire():
            yield mock_conn

        mock_pool = AsyncMock()
        mock_pool.acquire = mock_acquire

        with (
            patch.object(adapter, "_get_pool", return_value=mock_pool),
            patch.object(adapter, "_ensure_bm25_index", return_value=None),
        ):
            results = await adapter.search(
                query="test", working_dir="/tmp/rag/project", top_k=5
            )
            assert results == []

    @pytest.mark.asyncio
    async def test_index_document_is_noop(self) -> None:
        adapter = ClassicalBM25Adapter(
            db_url="postgresql://u:p@localhost/db",
            table_prefix="classical_rag_",
        )
        result = await adapter.index_document(
            chunk_id="test-id",
            content="test content",
            file_path="/test.pdf",
            working_dir="/tmp/rag/project",
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_close_cleans_up_pool(self) -> None:
        adapter = ClassicalBM25Adapter(
            db_url="postgresql://u:p@localhost/db",
            table_prefix="classical_rag_",
        )
        mock_pool = AsyncMock()
        adapter._pool = mock_pool

        await adapter.close()
        mock_pool.close.assert_called_once()
        assert adapter._pool is None

    @pytest.mark.asyncio
    async def test_close_handles_none_pool(self) -> None:
        adapter = ClassicalBM25Adapter(
            db_url="postgresql://u:p@localhost/db",
            table_prefix="classical_rag_",
        )
        adapter._pool = None
        await adapter.close()
