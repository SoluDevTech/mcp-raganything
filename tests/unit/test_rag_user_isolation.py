"""Tests for per-user RAG data isolation (TDD Red phase).

Isolation is enforced at the application level via ``langchain_metadata``:

- **Indexing**: the indexing use cases pass ``user_id`` (read from the
  ``current_user_id`` RLS contextvar) into the chunk metadata so each chunk
  is tagged with its owner.
- **Query**: the query use case adds a metadata filter
  ``{"user_id": current_user_id}`` to ``similarity_search`` so a user only
  retrieves their own chunks.

When ``current_user_id`` is None (legacy/tests, no auth wired), no
``user_id`` is added to the metadata and no filter is applied — the existing
behaviour is preserved so all 523 prior tests keep passing.

The vector store is mocked (``mock_vector_store``) so we assert on the
metadata passed to ``add_documents`` and the filter passed to
``similarity_search``.
"""

from unittest.mock import AsyncMock, MagicMock, patch

from infrastructure.database.rls_context import current_user_id


class TestIndexingUserIsolation:
    """Indexing use cases must tag chunks with ``user_id`` metadata."""

    @patch("application.use_cases.classical_index_file_use_case.extract_file")
    async def test_index_file_tags_chunks_with_user_id_when_contextvar_set(
        self,
        mock_extract: AsyncMock,
        mock_vector_store: AsyncMock,
        mock_storage: AsyncMock,
    ) -> None:
        # Arrange
        from application.use_cases.classical_index_file_use_case import (
            ClassicalIndexFileUseCase,
        )

        chunk = MagicMock()
        chunk.content = "chunk text"
        mock_result = MagicMock()
        mock_result.chunks = [chunk]
        mock_result.content = "full text"
        mock_extract.return_value = mock_result

        use_case = ClassicalIndexFileUseCase(
            vector_store=mock_vector_store,
            storage=mock_storage,
            bucket="test-bucket",
            output_dir="/tmp/output",
        )
        token = current_user_id.set("user-A")
        try:
            # Act
            await use_case.execute(
                file_name="docs/report.pdf",
                working_dir="/tmp/rag/project_1",
            )

            # Assert — every chunk metadata contains user_id=user-A
            call_kwargs = mock_vector_store.add_documents.call_args[1]
            documents = call_kwargs["documents"]
            assert len(documents) > 0
            for _content, _file_path, metadata in documents:
                assert metadata.get("user_id") == "user-A", (
                    f"user_id missing from chunk metadata: {metadata}"
                )
        finally:
            current_user_id.reset(token)

    @patch("application.use_cases.classical_index_file_use_case.extract_file")
    async def test_index_file_no_user_id_when_contextvar_none(
        self,
        mock_extract: AsyncMock,
        mock_vector_store: AsyncMock,
        mock_storage: AsyncMock,
    ) -> None:
        # Arrange — contextvar default None (legacy path)
        from application.use_cases.classical_index_file_use_case import (
            ClassicalIndexFileUseCase,
        )

        chunk = MagicMock()
        chunk.content = "chunk text"
        mock_result = MagicMock()
        mock_result.chunks = [chunk]
        mock_result.content = "full text"
        mock_extract.return_value = mock_result

        use_case = ClassicalIndexFileUseCase(
            vector_store=mock_vector_store,
            storage=mock_storage,
            bucket="test-bucket",
            output_dir="/tmp/output",
        )

        # Act
        await use_case.execute(
            file_name="docs/report.pdf",
            working_dir="/tmp/rag/project_1",
        )

        # Assert — no user_id key in metadata (legacy)
        call_kwargs = mock_vector_store.add_documents.call_args[1]
        documents = call_kwargs["documents"]
        assert len(documents) > 0
        for _content, _file_path, metadata in documents:
            assert "user_id" not in metadata, (
                f"unexpected user_id in legacy metadata: {metadata}"
            )


class TestQueryUserIsolation:
    """Query use case must filter by ``user_id`` when contextvar set."""

    async def test_query_filters_by_user_id_when_contextvar_set(
        self,
        mock_vector_store: AsyncMock,
        mock_llm: AsyncMock,
    ) -> None:
        # Arrange
        from application.use_cases.classical_query_use_case import (
            ClassicalQueryUseCase,
        )
        from config import ClassicalRAGConfig

        mock_vector_store.similarity_search.return_value = []
        use_case = ClassicalQueryUseCase(
            vector_store=mock_vector_store,
            llm=mock_llm,
            config=ClassicalRAGConfig(
                CLASSICAL_NUM_QUERY_VARIATIONS=1,
            ),
        )
        token = current_user_id.set("user-A")
        try:
            # Act
            await use_case.execute(
                working_dir="/tmp/rag/project_1",
                query="What is ML?",
                enable_llm_judge=False,
            )

            # Assert — similarity_search called with a metadata_filter
            # containing user_id=user-A
            assert mock_vector_store.similarity_search.await_count >= 1
            for call in mock_vector_store.similarity_search.await_args_list:
                kwargs = call.kwargs
                metadata_filter = kwargs.get("metadata_filter")
                assert metadata_filter is not None, (
                    f"metadata_filter missing from similarity_search call: {kwargs}"
                )
                assert metadata_filter.get("user_id") == "user-A", (
                    f"user_id not in filter: {metadata_filter}"
                )
        finally:
            current_user_id.reset(token)

    async def test_query_no_filter_when_contextvar_none(
        self,
        mock_vector_store: AsyncMock,
        mock_llm: AsyncMock,
    ) -> None:
        # Arrange — contextvar default None (legacy path)
        from application.use_cases.classical_query_use_case import (
            ClassicalQueryUseCase,
        )
        from config import ClassicalRAGConfig

        mock_vector_store.similarity_search.return_value = []
        use_case = ClassicalQueryUseCase(
            vector_store=mock_vector_store,
            llm=mock_llm,
            config=ClassicalRAGConfig(
                CLASSICAL_NUM_QUERY_VARIATIONS=1,
            ),
        )

        # Act
        await use_case.execute(
            working_dir="/tmp/rag/project_1",
            query="What is ML?",
            enable_llm_judge=False,
        )

        # Assert — no metadata_filter kwarg (or None) in legacy path
        for call in mock_vector_store.similarity_search.await_args_list:
            kwargs = call.kwargs
            metadata_filter = kwargs.get("metadata_filter")
            assert metadata_filter is None, (
                f"unexpected metadata_filter in legacy path: {metadata_filter}"
            )


class TestVectorStoreMetadataFilterForwarding:
    """The LangchainPgvectorAdapter must forward metadata_filter to PGVectorStore."""

    async def test_similarity_search_forwards_metadata_filter_to_store(
        self,
    ) -> None:
        # Arrange — mock the langchain-postgres internals
        from langchain_core.documents import Document

        with (
            patch(
                "infrastructure.vector_store.langchain_pgvector_adapter.PGVectorStore"
            ) as mock_store_cls,
            patch(
                "infrastructure.vector_store.langchain_pgvector_adapter.PGEngine"
            ) as mock_engine_cls,
        ):
            mock_engine = MagicMock()
            mock_engine.ainit_vectorstore_table = AsyncMock()
            mock_engine_cls.from_connection_string.return_value = mock_engine
            mock_store = MagicMock()
            mock_doc = Document(
                page_content="chunk",
                metadata={
                    "file_path": "docs/report.pdf",
                    "chunk_id": "c1",
                    "user_id": "user-A",
                },
            )
            mock_store.asimilarity_search_with_score = AsyncMock(
                return_value=[(mock_doc, 0.1)]
            )
            mock_store_cls.create = AsyncMock(return_value=mock_store)

            from infrastructure.vector_store.langchain_pgvector_adapter import (
                LangchainPgvectorAdapter,
            )

            adapter = LangchainPgvectorAdapter(
                connection_string="postgresql+asyncpg://u:p@h:5432/db",
                table_prefix="classical_rag_",
                embedding_dimension=1536,
            )
            await adapter.ensure_table(working_dir="/tmp/rag/p1")

            # Act
            await adapter.similarity_search(
                working_dir="/tmp/rag/p1",
                query="question",
                top_k=5,
                metadata_filter={"user_id": "user-A"},
            )

            # Assert — the underlying store search receives the filter
            mock_store.asimilarity_search_with_score.assert_awaited_once()
            call_kwargs = mock_store.asimilarity_search_with_score.await_args.kwargs
            # PGVectorStore supports a `filter` kwarg for metadata filtering.
            assert call_kwargs.get("filter") == {"user_id": "user-A"}, (
                f"filter not forwarded to PGVectorStore: {call_kwargs}"
            )
