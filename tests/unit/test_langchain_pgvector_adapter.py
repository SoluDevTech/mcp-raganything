"""Tests for LangchainPgvectorAdapter — TDD Red phase.

These tests will FAIL until the production code is implemented.
This adapter implements VectorStorePort using langchain-postgres PGVectorStore.
All langchain internals are mocked since they are external dependencies.
"""

import hashlib
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from domain.ports.vector_store_port import SearchResult, VectorStorePort


class TestLangchainPgvectorAdapter:
    """Tests for LangchainPgvectorAdapter."""

    @pytest.fixture
    def mock_pg_engine(self) -> MagicMock:
        """Mock PGEngine for external dependency."""
        engine = MagicMock()
        engine.close = AsyncMock()
        return engine

    @pytest.fixture
    def mock_pgvector_store(self) -> MagicMock:
        """Mock PGVectorStore for external dependency."""
        store = MagicMock()
        store.aadd_documents = AsyncMock(return_value=["id-1", "id-2"])
        store.asimilarity_search = AsyncMock(return_value=[])
        store.adelete = AsyncMock(return_value=None)
        return store

    @pytest.fixture
    def connection_string(self) -> str:
        return "postgresql+asyncpg://user:pass@localhost:5432/testdb"

    @pytest.fixture
    def adapter(self, connection_string: str):
        """Provide an adapter instance. Will fail until production code exists."""
        from infrastructure.vector_store.langchain_pgvector_adapter import (
            LangchainPgvectorAdapter,
        )

        return LangchainPgvectorAdapter(
            connection_string=connection_string,
            table_prefix="classical_rag_",
            embedding_dimension=1536,
        )

    # ------------------------------------------------------------------
    # Table naming convention
    # ------------------------------------------------------------------

    def test_table_name_format(self) -> None:
        """Table name should be {prefix}{sha256(working_dir)[:16]}."""
        from infrastructure.vector_store.langchain_pgvector_adapter import (
            LangchainPgvectorAdapter,
        )

        adapter = LangchainPgvectorAdapter(
            connection_string="postgresql+asyncpg://u:p@h:5432/db",
            table_prefix="classical_rag_",
            embedding_dimension=1536,
        )

        working_dir = "/tmp/rag/project_42"
        expected_hash = hashlib.sha256(working_dir.encode()).hexdigest()[:16]
        expected_table = f"classical_rag_{expected_hash}"

        table_name = adapter._get_table_name(working_dir)
        assert table_name == expected_table

    def test_table_name_deterministic(self) -> None:
        """Same working_dir should always produce the same table name."""
        from infrastructure.vector_store.langchain_pgvector_adapter import (
            LangchainPgvectorAdapter,
        )

        adapter = LangchainPgvectorAdapter(
            connection_string="postgresql+asyncpg://u:p@h:5432/db",
            table_prefix="classical_rag_",
            embedding_dimension=1536,
        )

        name1 = adapter._get_table_name("/tmp/rag/project_1")
        name2 = adapter._get_table_name("/tmp/rag/project_1")
        assert name1 == name2

    def test_different_working_dirs_produce_different_tables(self) -> None:
        """Different working_dirs should produce different table names."""
        from infrastructure.vector_store.langchain_pgvector_adapter import (
            LangchainPgvectorAdapter,
        )

        adapter = LangchainPgvectorAdapter(
            connection_string="postgresql+asyncpg://u:p@h:5432/db",
            table_prefix="classical_rag_",
            embedding_dimension=1536,
        )

        name1 = adapter._get_table_name("/tmp/rag/project_1")
        name2 = adapter._get_table_name("/tmp/rag/project_2")
        assert name1 != name2

    def test_custom_table_prefix(self) -> None:
        """Should use the configured table prefix."""
        from infrastructure.vector_store.langchain_pgvector_adapter import (
            LangchainPgvectorAdapter,
        )

        adapter = LangchainPgvectorAdapter(
            connection_string="postgresql+asyncpg://u:p@h:5432/db",
            table_prefix="custom_prefix_",
            embedding_dimension=1536,
        )

        table_name = adapter._get_table_name("/tmp/rag/project_1")
        assert table_name.startswith("custom_prefix_")

    # ------------------------------------------------------------------
    # ensure_table
    # ------------------------------------------------------------------

    @patch("infrastructure.vector_store.langchain_pgvector_adapter.PGVectorStore")
    @patch("infrastructure.vector_store.langchain_pgvector_adapter.PGEngine")
    async def test_ensure_table_creates_pgvector_store(
        self,
        mock_pg_engine_cls: MagicMock,
        mock_pgvector_store_cls: MagicMock,
        connection_string: str,
    ) -> None:
        """Should create PGEngine and PGVectorStore on ensure_table."""
        mock_engine = MagicMock()
        mock_pg_engine_cls.from_connection_string.return_value = mock_engine
        mock_store = MagicMock()
        mock_pgvector_store_cls.create.return_value = mock_store

        from infrastructure.vector_store.langchain_pgvector_adapter import (
            LangchainPgvectorAdapter,
        )

        adapter = LangchainPgvectorAdapter(
            connection_string=connection_string,
            table_prefix="classical_rag_",
            embedding_dimension=1536,
        )

        await adapter.ensure_table(working_dir="/tmp/rag/project_1")

        mock_pg_engine_cls.from_connection_string.assert_called_once_with(
            connection_string
        )
        mock_pgvector_store_cls.create.assert_called_once()

    # ------------------------------------------------------------------
    # add_documents
    # ------------------------------------------------------------------

    @patch("infrastructure.vector_store.langchain_pgvector_adapter.PGVectorStore")
    @patch("infrastructure.vector_store.langchain_pgvector_adapter.PGEngine")
    async def test_add_documents_calls_aadd_documents(
        self,
        mock_pg_engine_cls: MagicMock,
        mock_pgvector_store_cls: MagicMock,
        connection_string: str,
    ) -> None:
        """Should call aadd_documents on the PGVectorStore."""
        mock_engine = MagicMock()
        mock_pg_engine_cls.from_connection_string.return_value = mock_engine
        mock_store = MagicMock()
        mock_store.aadd_documents = AsyncMock(return_value=["id-1", "id-2"])
        mock_pgvector_store_cls.create.return_value = mock_store

        from infrastructure.vector_store.langchain_pgvector_adapter import (
            LangchainPgvectorAdapter,
        )

        adapter = LangchainPgvectorAdapter(
            connection_string=connection_string,
            table_prefix="classical_rag_",
            embedding_dimension=1536,
        )
        await adapter.ensure_table(working_dir="/tmp/rag/project_1")

        documents = [
            ("chunk text 1", "/docs/report.pdf", {"page": 1}),
            ("chunk text 2", "/docs/report.pdf", {"page": 2}),
        ]

        result = await adapter.add_documents(
            working_dir="/tmp/rag/project_1",
            documents=documents,
        )

        assert len(result) == 2
        for doc_id in result:
            assert len(doc_id) == 36
        mock_store.aadd_documents.assert_called_once()

    # ------------------------------------------------------------------
    # similarity_search
    # ------------------------------------------------------------------

    @patch("infrastructure.vector_store.langchain_pgvector_adapter.PGVectorStore")
    @patch("infrastructure.vector_store.langchain_pgvector_adapter.PGEngine")
    async def test_similarity_search_returns_search_results(
        self,
        mock_pg_engine_cls: MagicMock,
        mock_pgvector_store_cls: MagicMock,
        connection_string: str,
    ) -> None:
        """Should return list of SearchResult from asimilarity_search_with_score."""
        from langchain_core.documents import Document

        mock_engine = MagicMock()
        mock_pg_engine_cls.from_connection_string.return_value = mock_engine
        mock_store = MagicMock()
        mock_doc = Document(
            page_content="Relevant chunk text",
            metadata={
                "file_path": "/docs/report.pdf",
                "chunk_id": "chunk-1",
                "page": 1,
            },
        )
        mock_store.asimilarity_search_with_score = AsyncMock(
            return_value=[(mock_doc, 0.92)]
        )
        mock_pgvector_store_cls.create.return_value = mock_store

        from infrastructure.vector_store.langchain_pgvector_adapter import (
            LangchainPgvectorAdapter,
        )

        adapter = LangchainPgvectorAdapter(
            connection_string=connection_string,
            table_prefix="classical_rag_",
            embedding_dimension=1536,
        )
        await adapter.ensure_table(working_dir="/tmp/rag/project_1")

        results = await adapter.similarity_search(
            working_dir="/tmp/rag/project_1",
            query="What is machine learning?",
            top_k=10,
        )

        assert isinstance(results, list)
        assert len(results) >= 1
        assert isinstance(results[0], SearchResult)
        assert results[0].content == "Relevant chunk text"
        assert results[0].score == 0.92
        mock_store.asimilarity_search_with_score.assert_called_once()

    # ------------------------------------------------------------------
    # delete_documents
    # ------------------------------------------------------------------

    @patch("infrastructure.vector_store.langchain_pgvector_adapter.PGVectorStore")
    @patch("infrastructure.vector_store.langchain_pgvector_adapter.PGEngine")
    async def test_delete_documents_calls_adelete(
        self,
        mock_pg_engine_cls: MagicMock,
        mock_pgvector_store_cls: MagicMock,
        connection_string: str,
    ) -> None:
        """Should call adelete on the PGVectorStore for matching file_path."""
        mock_engine = MagicMock()
        mock_pg_engine_cls.from_connection_string.return_value = mock_engine
        mock_store = MagicMock()
        mock_store.aadd_documents = AsyncMock(return_value=None)
        mock_store.adelete = AsyncMock(return_value=None)
        mock_pgvector_store_cls.create.return_value = mock_store

        from infrastructure.vector_store.langchain_pgvector_adapter import (
            LangchainPgvectorAdapter,
        )

        adapter = LangchainPgvectorAdapter(
            connection_string=connection_string,
            table_prefix="classical_rag_",
            embedding_dimension=1536,
        )
        await adapter.ensure_table(working_dir="/tmp/rag/project_1")

        documents = [
            ("chunk text 1", "/docs/report.pdf", {"page": 1}),
        ]
        await adapter.add_documents(
            working_dir="/tmp/rag/project_1", documents=documents
        )

        result = await adapter.delete_documents(
            working_dir="/tmp/rag/project_1",
            file_path="/docs/report.pdf",
        )

        assert isinstance(result, int)
        assert result == 1
        mock_store.adelete.assert_called_once()

    # ------------------------------------------------------------------
    # close
    # ------------------------------------------------------------------

    @patch("infrastructure.vector_store.langchain_pgvector_adapter.PGVectorStore")
    @patch("infrastructure.vector_store.langchain_pgvector_adapter.PGEngine")
    async def test_close_closes_engine(
        self,
        mock_pg_engine_cls: MagicMock,
        mock_pgvector_store_cls: MagicMock,
        connection_string: str,
    ) -> None:
        """Should close the PGEngine connection pool."""
        mock_engine = MagicMock()
        mock_engine.close = AsyncMock()
        mock_pg_engine_cls.from_connection_string.return_value = mock_engine

        from infrastructure.vector_store.langchain_pgvector_adapter import (
            LangchainPgvectorAdapter,
        )

        adapter = LangchainPgvectorAdapter(
            connection_string=connection_string,
            table_prefix="classical_rag_",
            embedding_dimension=1536,
        )
        await adapter.ensure_table(working_dir="/tmp/rag/project_1")

        await adapter.close()

        mock_engine.close.assert_called_once()

    # ------------------------------------------------------------------
    # Interface compliance
    # ------------------------------------------------------------------

    def test_implements_vector_store_port(self) -> None:
        """LangchainPgvectorAdapter should implement VectorStorePort."""
        from infrastructure.vector_store.langchain_pgvector_adapter import (
            LangchainPgvectorAdapter,
        )

        assert issubclass(LangchainPgvectorAdapter, VectorStorePort)
