from unittest.mock import AsyncMock

from application.use_cases.query_use_case import QueryUseCase


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
