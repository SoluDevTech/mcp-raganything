from unittest.mock import AsyncMock

from application.use_cases.query_use_case import QueryUseCase
from domain.ports.bm25_engine import BM25SearchResult


class TestQueryUseCase:
    """Tests for QueryUseCase — rag_engine is external, mocked."""

    async def test_execute_calls_init_project(
        self,
        mock_rag_engine: AsyncMock,
    ) -> None:
        """Should call rag_engine.init_project with the working_dir."""
        use_case = QueryUseCase(rag_engine=mock_rag_engine)

        await use_case.execute(
            working_dir="/tmp/rag/project_42",
            query="What are the findings?",
        )

        mock_rag_engine.init_project.assert_called_once_with("/tmp/rag/project_42")

    async def test_execute_calls_query_with_correct_params(
        self,
        mock_rag_engine: AsyncMock,
    ) -> None:
        """Should call rag_engine.query with query, mode, top_k, and working_dir."""
        mock_rag_engine.query.return_value = {"status": "success", "data": {}}
        use_case = QueryUseCase(rag_engine=mock_rag_engine)

        await use_case.execute(
            working_dir="/tmp/rag/test",
            query="Tell me about X",
            mode="hybrid",
            top_k=20,
        )

        mock_rag_engine.query.assert_called_once_with(
            query="Tell me about X",
            mode="hybrid",
            top_k=20,
            working_dir="/tmp/rag/test",
        )

    async def test_execute_returns_result_from_rag_engine(
        self,
        mock_rag_engine: AsyncMock,
    ) -> None:
        """Should return the dict result from rag_engine.query."""
        expected = {"status": "success", "data": {"answer": "42"}}
        mock_rag_engine.query.return_value = expected
        use_case = QueryUseCase(rag_engine=mock_rag_engine)

        result = await use_case.execute(
            working_dir="/tmp/rag/test",
            query="What is the answer?",
        )

        assert result == expected

    async def test_execute_uses_default_mode_and_top_k(
        self,
        mock_rag_engine: AsyncMock,
    ) -> None:
        """Should use mode='naive' and top_k=10 by default."""
        mock_rag_engine.query.return_value = {"status": "success", "data": {}}
        use_case = QueryUseCase(rag_engine=mock_rag_engine)

        await use_case.execute(
            working_dir="/tmp/rag/test",
            query="test query",
        )

        mock_rag_engine.query.assert_called_once_with(
            query="test query",
            mode="naive",
            top_k=10,
            working_dir="/tmp/rag/test",
        )

    async def test_execute_with_mix_mode(
        self,
        mock_rag_engine: AsyncMock,
    ) -> None:
        """Should pass different mode values through correctly."""
        mock_rag_engine.query.return_value = {"status": "success", "data": {}}
        use_case = QueryUseCase(rag_engine=mock_rag_engine)

        await use_case.execute(
            working_dir="/tmp/rag/test",
            query="test query",
            mode="mix",
            top_k=5,
        )

        mock_rag_engine.query.assert_called_once_with(
            query="test query",
            mode="mix",
            top_k=5,
            working_dir="/tmp/rag/test",
        )

    async def test_execute_hybrid_plus_with_bm25(
        self, mock_rag_engine: AsyncMock
    ) -> None:
        """hybrid+ mode should execute parallel BM25 + vector search."""
        mock_bm25 = AsyncMock()
        mock_bm25.search.return_value = [
            BM25SearchResult(
                chunk_id="chunk-abc123",
                content="bm25 result",
                file_path="/a.pdf",
                score=5.0,
                metadata={},
            )
        ]
        mock_rag_engine.query.return_value = {
            "data": {
                "chunks": [
                    {
                        "chunk_id": "chunk-abc123",
                        "reference_id": "2",
                        "content": "vector result",
                        "file_path": "/b.pdf",
                    }
                ]
            }
        }
        use_case = QueryUseCase(rag_engine=mock_rag_engine, bm25_engine=mock_bm25)

        result = await use_case.execute(
            working_dir="/tmp/rag/test", query="search", mode="hybrid+", top_k=10
        )

        mock_bm25.search.assert_called_once()
        mock_rag_engine.query.assert_called()
        assert result["status"] == "success"
        assert result["metadata"]["query_mode"] == "hybrid+"

        chunk = result["data"]["chunks"][0]
        assert chunk["chunk_id"] == "chunk-abc123"
        assert chunk["reference_id"] == "2"

    async def test_execute_hybrid_plus_without_bm25_falls_back(
        self, mock_rag_engine: AsyncMock
    ) -> None:
        """hybrid+ mode without BM25 should fall back to naive vector search."""
        mock_rag_engine.query.return_value = {"status": "success", "data": {}}
        use_case = QueryUseCase(rag_engine=mock_rag_engine, bm25_engine=None)

        await use_case.execute(
            working_dir="/tmp/rag/test", query="search", mode="hybrid+", top_k=10
        )

        mock_rag_engine.query.assert_called_once_with(
            query="search", mode="naive", top_k=10, working_dir="/tmp/rag/test"
        )

    async def test_execute_bm25_only_mode(self, mock_rag_engine: AsyncMock) -> None:
        """bm25 mode should only use BM25 search without vector."""
        mock_bm25 = AsyncMock()
        mock_bm25.search.return_value = [
            BM25SearchResult(
                chunk_id="1", content="test", file_path="/a.pdf", score=5.0, metadata={}
            )
        ]
        use_case = QueryUseCase(rag_engine=mock_rag_engine, bm25_engine=mock_bm25)

        result = await use_case.execute(
            working_dir="/tmp/rag/test", query="search", mode="bm25", top_k=10
        )

        mock_bm25.search.assert_called_once_with("search", "/tmp/rag/test", 10)
        mock_rag_engine.query.assert_not_called()
        assert result["status"] == "success"
        assert result["metadata"]["query_mode"] == "bm25"

    async def test_execute_bm25_mode_without_bm25_returns_error(
        self, mock_rag_engine: AsyncMock
    ) -> None:
        """bm25 mode without BM25 engine should return error."""
        use_case = QueryUseCase(rag_engine=mock_rag_engine, bm25_engine=None)

        result = await use_case.execute(
            working_dir="/tmp/rag/test", query="search", mode="bm25", top_k=10
        )

        assert result["status"] == "error"
        assert "BM25 engine not available" in result["message"]
