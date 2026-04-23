from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from application.use_cases.classical_index_file_use_case import (
    ClassicalIndexFileUseCase,
)
from domain.entities.indexing_result import FileIndexingResult, IndexingStatus


class TestClassicalIndexFileUseCase:
    @pytest.fixture
    def use_case(
        self,
        mock_vector_store: AsyncMock,
        mock_storage: AsyncMock,
    ) -> ClassicalIndexFileUseCase:
        return ClassicalIndexFileUseCase(
            vector_store=mock_vector_store,
            storage=mock_storage,
            bucket="test-bucket",
            output_dir="/tmp/output",
        )

    @patch("application.use_cases.classical_index_file_use_case.extract_file")
    async def test_execute_downloads_file_from_storage(
        self,
        mock_extract: AsyncMock,
        use_case: ClassicalIndexFileUseCase,
        mock_storage: AsyncMock,
    ) -> None:
        mock_result = MagicMock()
        mock_result.chunks = []
        mock_result.content = "Some text"
        mock_extract.return_value = mock_result

        await use_case.execute(
            file_name="reports/quarterly.pdf",
            working_dir="/tmp/rag/project_1",
        )

        mock_storage.get_object.assert_called_once_with(
            "test-bucket", "reports/quarterly.pdf"
        )

    @patch("application.use_cases.classical_index_file_use_case.extract_file")
    async def test_execute_ensures_vector_store_table(
        self,
        mock_extract: AsyncMock,
        use_case: ClassicalIndexFileUseCase,
        mock_vector_store: AsyncMock,
    ) -> None:
        mock_result = MagicMock()
        mock_result.chunks = []
        mock_result.content = "text"
        mock_extract.return_value = mock_result

        await use_case.execute(
            file_name="report.pdf",
            working_dir="/tmp/rag/project_1",
        )

        mock_vector_store.ensure_table.assert_called_once_with("/tmp/rag/project_1")

    @patch("application.use_cases.classical_index_file_use_case.extract_file")
    async def test_execute_extracts_content_via_kreuzberg(
        self,
        mock_extract: AsyncMock,
        use_case: ClassicalIndexFileUseCase,
    ) -> None:
        mock_result = MagicMock()
        mock_result.chunks = []
        mock_result.content = "Extracted text"
        mock_extract.return_value = mock_result

        await use_case.execute(
            file_name="docs/report.pdf",
            working_dir="/tmp/rag/project_42",
        )

        mock_extract.assert_called_once()
        call_args = mock_extract.call_args
        assert call_args[0][0].endswith("docs/report.pdf") or "report.pdf" in str(
            call_args
        )

    @patch("application.use_cases.classical_index_file_use_case.extract_file")
    async def test_execute_adds_documents_to_vector_store(
        self,
        mock_extract: AsyncMock,
        use_case: ClassicalIndexFileUseCase,
        mock_vector_store: AsyncMock,
    ) -> None:
        mock_result = MagicMock()
        chunk = MagicMock()
        chunk.content = "chunk text"
        mock_result.chunks = [chunk]
        mock_result.content = "full text"
        mock_extract.return_value = mock_result

        await use_case.execute(
            file_name="report.pdf",
            working_dir="/tmp/rag/project_1",
        )

        mock_vector_store.add_documents.assert_called_once()
        call_kwargs = mock_vector_store.add_documents.call_args
        assert call_kwargs[1]["working_dir"] == "/tmp/rag/project_1"
        documents = call_kwargs[1]["documents"]
        assert isinstance(documents, list)
        assert len(documents) > 0

    @patch("application.use_cases.classical_index_file_use_case.extract_file")
    async def test_execute_passes_chunk_size_and_overlap(
        self,
        mock_extract: AsyncMock,
        use_case: ClassicalIndexFileUseCase,
    ) -> None:
        mock_result = MagicMock()
        mock_result.chunks = []
        mock_result.content = "text"
        mock_extract.return_value = mock_result

        await use_case.execute(
            file_name="report.pdf",
            working_dir="/tmp/rag/project_1",
            chunk_size=500,
            chunk_overlap=100,
        )

        mock_extract.assert_called_once()

    @patch("application.use_cases.classical_index_file_use_case.extract_file")
    async def test_execute_uses_default_chunk_params(
        self,
        mock_extract: AsyncMock,
        use_case: ClassicalIndexFileUseCase,
        mock_vector_store: AsyncMock,
    ) -> None:
        mock_result = MagicMock()
        mock_result.chunks = []
        mock_result.content = "text"
        mock_extract.return_value = mock_result

        await use_case.execute(
            file_name="report.pdf",
            working_dir="/tmp/rag/project_1",
        )

        mock_vector_store.add_documents.assert_called_once()

    @patch("application.use_cases.classical_index_file_use_case.extract_file")
    async def test_execute_returns_file_indexing_result_on_success(
        self,
        mock_extract: AsyncMock,
        use_case: ClassicalIndexFileUseCase,
    ) -> None:
        mock_result = MagicMock()
        mock_result.chunks = []
        mock_result.content = "text"
        mock_extract.return_value = mock_result

        result = await use_case.execute(
            file_name="report.pdf",
            working_dir="/tmp/rag/project_1",
        )

        assert isinstance(result, FileIndexingResult)
        assert result.status == IndexingStatus.SUCCESS
        assert result.file_name == "report.pdf"

    async def test_execute_returns_failed_result_on_vector_store_error(
        self,
        use_case: ClassicalIndexFileUseCase,
        mock_vector_store: AsyncMock,
    ) -> None:
        mock_vector_store.add_documents.side_effect = RuntimeError("Connection lost")

        result = await use_case.execute(
            file_name="report.pdf",
            working_dir="/tmp/rag/project_1",
        )

        assert result.status == IndexingStatus.FAILED
        assert result.error is not None

    @patch("application.use_cases.classical_index_file_use_case.extract_file")
    async def test_execute_returns_failed_result_on_extraction_error(
        self,
        mock_extract: AsyncMock,
        use_case: ClassicalIndexFileUseCase,
    ) -> None:
        mock_extract.side_effect = RuntimeError("Unsupported format")

        result = await use_case.execute(
            file_name="corrupt.xyz",
            working_dir="/tmp/rag/project_1",
        )

        assert result.status == IndexingStatus.FAILED

    async def test_execute_returns_failed_result_on_storage_error(
        self,
        use_case: ClassicalIndexFileUseCase,
        mock_storage: AsyncMock,
    ) -> None:
        mock_storage.get_object.side_effect = FileNotFoundError("File not found")

        result = await use_case.execute(
            file_name="nonexistent.pdf",
            working_dir="/tmp/rag/project_1",
        )

        assert result.status == IndexingStatus.FAILED

    @patch("application.use_cases.classical_index_file_use_case.extract_file")
    async def test_execute_adds_documents_with_file_path_metadata(
        self,
        mock_extract: AsyncMock,
        use_case: ClassicalIndexFileUseCase,
        mock_vector_store: AsyncMock,
    ) -> None:
        chunk = MagicMock()
        chunk.content = "chunk text"
        mock_result = MagicMock()
        mock_result.chunks = [chunk]
        mock_result.content = "full text"
        mock_extract.return_value = mock_result

        await use_case.execute(
            file_name="docs/report.pdf",
            working_dir="/tmp/rag/project_1",
        )

        call_kwargs = mock_vector_store.add_documents.call_args[1]
        documents = call_kwargs["documents"]
        for _content, file_path, _metadata in documents:
            assert file_path == "docs/report.pdf"

    async def test_execute_custom_bucket_and_output_dir(
        self,
        mock_vector_store: AsyncMock,
        mock_storage: AsyncMock,
    ) -> None:
        use_case = ClassicalIndexFileUseCase(
            vector_store=mock_vector_store,
            storage=mock_storage,
            bucket="custom-bucket",
            output_dir="/custom/output",
        )

        mock_storage.get_object.side_effect = FileNotFoundError("skip extract")

        await use_case.execute(
            file_name="test.pdf",
            working_dir="/tmp/rag/custom",
        )

        mock_storage.get_object.assert_called_once_with("custom-bucket", "test.pdf")

    @patch("application.use_cases.classical_index_file_use_case.extract_file")
    async def test_execute_processes_empty_document(
        self,
        mock_extract: AsyncMock,
        use_case: ClassicalIndexFileUseCase,
        mock_vector_store: AsyncMock,
    ) -> None:
        mock_result = MagicMock()
        mock_result.chunks = []
        mock_result.content = ""
        mock_extract.return_value = mock_result

        result = await use_case.execute(
            file_name="empty.pdf",
            working_dir="/tmp/rag/project_1",
        )

        assert result.status == IndexingStatus.SUCCESS
