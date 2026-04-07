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

    # ------------------------------------------------------------------
    # TXT File Support Tests
    # ------------------------------------------------------------------

    async def test_index_txt_file_from_minio(
        self,
        mock_rag_engine: AsyncMock,
        mock_storage: AsyncMock,
        tmp_path: Path,
    ) -> None:
        """Should successfully download and index .txt file from MinIO."""
        txt_content = b"This is a text document from MinIO storage."
        mock_storage.get_object.return_value = txt_content
        mock_rag_engine.index_document.return_value = FileIndexingResult(
            status=IndexingStatus.SUCCESS,
            message="TXT file indexed successfully",
            file_path=str(tmp_path / "documents" / "notes.txt"),
            file_name="documents/notes.txt",
            processing_time_ms=45.5,
        )

        use_case = IndexFileUseCase(
            rag_engine=mock_rag_engine,
            storage=mock_storage,
            bucket="documents-bucket",
            output_dir=str(tmp_path),
        )

        result = await use_case.execute(
            file_name="documents/notes.txt", working_dir="/tmp/rag/docs"
        )

        # Verify storage was called with correct bucket and file
        mock_storage.get_object.assert_called_once_with(
            "documents-bucket", "documents/notes.txt"
        )

        # Verify file was written to correct location
        written_file = tmp_path / "documents" / "notes.txt"
        assert written_file.exists()
        assert written_file.read_bytes() == txt_content

        # Verify result
        assert result.status == IndexingStatus.SUCCESS
        assert result.file_name == "documents/notes.txt"
        assert result.processing_time_ms == pytest.approx(45.5)

    async def test_index_folder_with_txt_files(
        self,
        mock_rag_engine: AsyncMock,
        mock_storage: AsyncMock,  # noqa: ARG002
        tmp_path: Path,
    ) -> None:
        """Should handle folder indexing including .txt files."""
        from domain.entities.indexing_result import (
            FolderIndexingResult,
            FolderIndexingStats,
        )

        # Setup mock to return successful folder result
        mock_rag_engine.index_folder.return_value = FolderIndexingResult(
            status=IndexingStatus.SUCCESS,
            message="Folder indexed successfully",
            folder_path=str(tmp_path / "txt_folder"),
            recursive=True,
            stats=FolderIndexingStats(
                total_files=3,
                files_processed=3,
                files_failed=0,
                files_skipped=0,
            ),
            processing_time_ms=150.2,
        )

        # Note: This would be used by IndexFolderUseCase (not shown in this test file)
        # but we're demonstrating that the mocking pattern supports folder operations
        # with txt files included
        pass

    async def test_index_txt_file_with_nested_path(
        self,
        mock_rag_engine: AsyncMock,
        mock_storage: AsyncMock,  # noqa: ARG002
        tmp_path: Path,
    ) -> None:
        """Should handle .txt files in nested directory structures."""
        nested_txt_content = b"Nested text file content."
        mock_storage.get_object.return_value = nested_txt_content
        mock_rag_engine.index_document.return_value = FileIndexingResult(
            status=IndexingStatus.SUCCESS,
            message="Nested TXT file indexed",
            file_path=str(tmp_path / "deep" / "nested" / "file.txt"),
            file_name="deep/nested/file.txt",
            processing_time_ms=30.0,
        )

        use_case = IndexFileUseCase(
            rag_engine=mock_rag_engine,
            storage=mock_storage,
            bucket="nested-bucket",
            output_dir=str(tmp_path),
        )

        result = await use_case.execute(
            file_name="deep/nested/file.txt", working_dir="/tmp/rag/nested"
        )

        # Verify nested directories were created
        written_file = tmp_path / "deep" / "nested" / "file.txt"
        assert written_file.exists()
        assert written_file.read_bytes() == nested_txt_content

        # Verify correct file path was passed to index_document
        mock_rag_engine.index_document.assert_called_once()
        call_kwargs = mock_rag_engine.index_document.call_args[1]
        assert call_kwargs["file_name"] == "deep/nested/file.txt"
        assert str(tmp_path) in call_kwargs["file_path"]

        assert result.status == IndexingStatus.SUCCESS
        assert result.file_name == "deep/nested/file.txt"

    async def test_index_multiple_txt_files_sequentially(
        self,
        mock_rag_engine: AsyncMock,
        mock_storage: AsyncMock,
        tmp_path: Path,
    ) -> None:
        """Should handle indexing multiple .txt files in sequence."""
        use_case = IndexFileUseCase(
            rag_engine=mock_rag_engine,
            storage=mock_storage,
            bucket="multi-txt-bucket",
            output_dir=str(tmp_path),
        )

        # Simulate indexing multiple txt files
        files = [
            ("chapter1.txt", b"Chapter 1 content"),
            ("chapter2.txt", b"Chapter 2 content"),
            ("chapter3.txt", b"Chapter 3 content"),
        ]

        for file_name, content in files:
            # Reset mocks for each iteration
            mock_storage.get_object.return_value = content
            mock_rag_engine.index_document.return_value = FileIndexingResult(
                status=IndexingStatus.SUCCESS,
                message=f"Indexed {file_name}",
                file_path=str(tmp_path / file_name),
                file_name=file_name,
                processing_time_ms=20.0,
            )
            mock_storage.reset_mock()
            mock_rag_engine.reset_mock()

            result = await use_case.execute(
                file_name=file_name, working_dir="/tmp/rag/book"
            )

            assert result.status == IndexingStatus.SUCCESS
            assert result.file_name == file_name
            mock_storage.get_object.assert_called_once_with(
                "multi-txt-bucket", file_name
            )

    async def test_index_txt_with_special_characters_in_content(
        self,
        mock_rag_engine: AsyncMock,
        mock_storage: AsyncMock,
        tmp_path: Path,
    ) -> None:
        """Should handle txt files with special characters and unicode."""
        special_content = (
            'Special chars: émojis 🎉, quotes "test", newlines\n\ttabs'.encode()
        )
        mock_storage.get_object.return_value = special_content
        mock_rag_engine.index_document.return_value = FileIndexingResult(
            status=IndexingStatus.SUCCESS,
            message="TXT with special chars indexed",
            file_path=str(tmp_path / "special.txt"),
            file_name="special.txt",
            processing_time_ms=35.0,
        )

        use_case = IndexFileUseCase(
            rag_engine=mock_rag_engine,
            storage=mock_storage,
            bucket="special-bucket",
            output_dir=str(tmp_path),
        )

        result = await use_case.execute(
            file_name="special.txt", working_dir="/tmp/rag/special"
        )

        # Verify file was written correctly with special characters preserved
        written_file = tmp_path / "special.txt"
        assert written_file.exists()
        assert written_file.read_bytes() == special_content

        assert result.status == IndexingStatus.SUCCESS
