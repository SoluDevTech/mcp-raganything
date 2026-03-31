import os
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from application.use_cases.index_file_use_case import IndexFileUseCase
from domain.entities.indexing_result import FileIndexingResult, IndexingStatus


class TestIndexFileUseCase:
    """Tests for IndexFileUseCase — storage and rag_engine are external, both mocked."""

    async def test_execute_downloads_file_from_storage(
        self,
        mock_rag_engine: AsyncMock,
        mock_storage: AsyncMock,
        tmp_path: Path,
    ) -> None:
        """Should call storage.get_object with the bucket and file_name."""
        use_case = IndexFileUseCase(
            rag_engine=mock_rag_engine,
            storage=mock_storage,
            bucket="my-bucket",
            output_dir=str(tmp_path),
        )

        await use_case.execute(
            file_name="reports/report.pdf", working_dir="/tmp/rag/p1"
        )

        mock_storage.get_object.assert_called_once_with(
            "my-bucket", "reports/report.pdf"
        )

    async def test_execute_writes_file_to_output_dir(
        self,
        mock_rag_engine: AsyncMock,
        mock_storage: AsyncMock,
        tmp_path: Path,
    ) -> None:
        """Should write the downloaded bytes to output_dir/<file_name>."""
        mock_storage.get_object.return_value = b"pdf binary data"
        use_case = IndexFileUseCase(
            rag_engine=mock_rag_engine,
            storage=mock_storage,
            bucket="my-bucket",
            output_dir=str(tmp_path),
        )

        await use_case.execute(file_name="docs/report.pdf", working_dir="/tmp/rag/p1")

        written_file = tmp_path / "docs" / "report.pdf"
        assert written_file.exists()
        assert written_file.read_bytes() == b"pdf binary data"

    async def test_execute_calls_init_project(
        self,
        mock_rag_engine: AsyncMock,
        mock_storage: AsyncMock,
        tmp_path: Path,
    ) -> None:
        """Should call rag_engine.init_project with the working_dir."""
        use_case = IndexFileUseCase(
            rag_engine=mock_rag_engine,
            storage=mock_storage,
            bucket="my-bucket",
            output_dir=str(tmp_path),
        )

        await use_case.execute(
            file_name="report.pdf", working_dir="/tmp/rag/project_42"
        )

        mock_rag_engine.init_project.assert_called_once_with("/tmp/rag/project_42")

    async def test_execute_calls_index_document(
        self,
        mock_rag_engine: AsyncMock,
        mock_storage: AsyncMock,
        tmp_path: Path,
    ) -> None:
        """Should call rag_engine.index_document with the local file path, name, and output_dir."""
        output_dir = str(tmp_path)
        use_case = IndexFileUseCase(
            rag_engine=mock_rag_engine,
            storage=mock_storage,
            bucket="my-bucket",
            output_dir=output_dir,
        )

        await use_case.execute(
            file_name="nested/dir/report.pdf", working_dir="/tmp/rag/p1"
        )

        expected_file_path = os.path.join(output_dir, "nested/dir/report.pdf")
        mock_rag_engine.index_document.assert_called_once_with(
            file_path=expected_file_path,
            file_name="nested/dir/report.pdf",
            output_dir=output_dir,
            working_dir="/tmp/rag/p1",
        )

    async def test_execute_returns_result(
        self,
        mock_rag_engine: AsyncMock,
        mock_storage: AsyncMock,
        tmp_path: Path,
    ) -> None:
        """Should return the FileIndexingResult from rag_engine."""
        expected_result = FileIndexingResult(
            status=IndexingStatus.SUCCESS,
            message="Indexed",
            file_path="/tmp/report.pdf",
            file_name="report.pdf",
            processing_time_ms=42.0,
        )
        mock_rag_engine.index_document.return_value = expected_result
        use_case = IndexFileUseCase(
            rag_engine=mock_rag_engine,
            storage=mock_storage,
            bucket="my-bucket",
            output_dir=str(tmp_path),
        )

        result = await use_case.execute(
            file_name="report.pdf", working_dir="/tmp/rag/p1"
        )

        assert result.status == IndexingStatus.SUCCESS
        assert result.file_name == "report.pdf"
        assert result.processing_time_ms == pytest.approx(42.0)

    async def test_execute_with_failure(
        self,
        mock_rag_engine: AsyncMock,
        mock_storage: AsyncMock,
        tmp_path: Path,
    ) -> None:
        """Should return a FAILED result when rag_engine reports failure."""
        mock_rag_engine.index_document.return_value = FileIndexingResult(
            status=IndexingStatus.FAILED,
            message="Parsing error",
            file_path="/tmp/bad.pdf",
            file_name="bad.pdf",
            error="Corrupt PDF",
        )
        use_case = IndexFileUseCase(
            rag_engine=mock_rag_engine,
            storage=mock_storage,
            bucket="my-bucket",
            output_dir=str(tmp_path),
        )

        result = await use_case.execute(file_name="bad.pdf", working_dir="/tmp/rag/p1")

        assert result.status == IndexingStatus.FAILED
        assert result.error == "Corrupt PDF"
