from unittest.mock import AsyncMock

from application.requests.query_request import MultimodalContentItem
from application.use_cases.multimodal_query_use_case import MultimodalQueryUseCase


class TestMultimodalQueryUseCase:
    """Tests for MultimodalQueryUseCase — rag_engine is external, mocked."""

    async def test_execute_calls_init_project(
        self,
        mock_rag_engine: AsyncMock,
    ) -> None:
        """Should call rag_engine.init_project with the working_dir."""
        use_case = MultimodalQueryUseCase(rag_engine=mock_rag_engine)
        content = [MultimodalContentItem(type="image", img_path="/tmp/img.png")]

        await use_case.execute(
            working_dir="/tmp/rag/project_42",
            query="What does this image show?",
            multimodal_content=content,
        )

        mock_rag_engine.init_project.assert_called_once_with("/tmp/rag/project_42")

    async def test_execute_calls_query_multimodal_with_correct_params(
        self,
        mock_rag_engine: AsyncMock,
    ) -> None:
        """Should call rag_engine.query_multimodal with all params."""
        use_case = MultimodalQueryUseCase(rag_engine=mock_rag_engine)
        content = [
            MultimodalContentItem(
                type="table", table_data="A,B\n1,2", table_caption="Test table"
            ),
        ]

        await use_case.execute(
            working_dir="/tmp/rag/test",
            query="Analyze this table",
            multimodal_content=content,
            mode="global",
            top_k=20,
        )

        mock_rag_engine.query_multimodal.assert_called_once_with(
            query="Analyze this table",
            multimodal_content=content,
            mode="global",
            top_k=20,
            working_dir="/tmp/rag/test",
        )

    async def test_execute_returns_success_with_result(
        self,
        mock_rag_engine: AsyncMock,
    ) -> None:
        """Should return dict with status='success' and data from rag_engine."""
        mock_rag_engine.query_multimodal.return_value = "Analysis of the image content"
        use_case = MultimodalQueryUseCase(rag_engine=mock_rag_engine)
        content = [MultimodalContentItem(type="image", img_path="/tmp/diagram.png")]

        result = await use_case.execute(
            working_dir="/tmp/rag/test",
            query="Describe this diagram",
            multimodal_content=content,
        )

        assert result == {"data": "Analysis of the image content"}

    async def test_execute_uses_default_mode_and_top_k(
        self,
        mock_rag_engine: AsyncMock,
    ) -> None:
        """Should use mode='hybrid' and top_k=10 by default."""
        use_case = MultimodalQueryUseCase(rag_engine=mock_rag_engine)
        content = [MultimodalContentItem(type="equation", latex="E=mc^2")]

        await use_case.execute(
            working_dir="/tmp/rag/test",
            query="Explain this formula",
            multimodal_content=content,
        )

        mock_rag_engine.query_multimodal.assert_called_once_with(
            query="Explain this formula",
            multimodal_content=content,
            mode="hybrid",
            top_k=10,
            working_dir="/tmp/rag/test",
        )
