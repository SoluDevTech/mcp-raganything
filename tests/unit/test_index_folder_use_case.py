import os
from pathlib import Path
from unittest.mock import AsyncMock, call

from application.requests.indexing_request import IndexFolderRequest
from application.use_cases.index_folder_use_case import IndexFolderUseCase
from domain.entities.indexing_result import (
    FolderIndexingResult,
    FolderIndexingStats,
    IndexingStatus,
)


class TestIndexFolderUseCase:
    """Tests for IndexFolderUseCase — storage and rag_engine are external, both mocked."""

    async def test_execute_lists_objects_from_storage(
        self,
        mock_rag_engine: AsyncMock,
        mock_storage: AsyncMock,
        tmp_path: Path,
    ) -> None:
        """Should call storage.list_objects with bucket, working_dir prefix, and recursive flag."""
        use_case = IndexFolderUseCase(
            rag_engine=mock_rag_engine,
            storage=mock_storage,
            bucket="my-bucket",
            output_dir=str(tmp_path),
        )
        request = IndexFolderRequest(working_dir="project/docs", recursive=True)

        await use_case.execute(request)

        mock_storage.list_objects.assert_called_once_with(
            "my-bucket", prefix="project/docs", recursive=True
        )

    async def test_execute_downloads_all_listed_files(
        self,
        mock_rag_engine: AsyncMock,
        mock_storage: AsyncMock,
        tmp_path: Path,
    ) -> None:
        """Should call storage.get_object for each file returned by list_objects."""
        mock_storage.list_objects.return_value = [
            "project/docs/a.pdf",
            "project/docs/b.pdf",
            "project/docs/c.docx",
        ]
        mock_storage.get_object.return_value = b"content"
        use_case = IndexFolderUseCase(
            rag_engine=mock_rag_engine,
            storage=mock_storage,
            bucket="my-bucket",
            output_dir=str(tmp_path),
        )
        request = IndexFolderRequest(working_dir="project/docs")

        await use_case.execute(request)

        assert mock_storage.get_object.call_count == 3
        mock_storage.get_object.assert_has_calls(
            [
                call("my-bucket", "project/docs/a.pdf"),
                call("my-bucket", "project/docs/b.pdf"),
                call("my-bucket", "project/docs/c.docx"),
            ],
            any_order=False,
        )

    async def test_execute_filters_by_file_extensions(
        self,
        mock_rag_engine: AsyncMock,
        mock_storage: AsyncMock,
        tmp_path: Path,
    ) -> None:
        """Should only download files matching the requested extensions."""
        mock_storage.list_objects.return_value = [
            "project/docs/a.pdf",
            "project/docs/b.txt",
            "project/docs/c.docx",
        ]
        mock_storage.get_object.return_value = b"content"
        use_case = IndexFolderUseCase(
            rag_engine=mock_rag_engine,
            storage=mock_storage,
            bucket="my-bucket",
            output_dir=str(tmp_path),
        )
        request = IndexFolderRequest(
            working_dir="project/docs",
            file_extensions=[".pdf", ".docx"],
        )

        await use_case.execute(request)

        assert mock_storage.get_object.call_count == 2
        mock_storage.get_object.assert_has_calls(
            [
                call("my-bucket", "project/docs/a.pdf"),
                call("my-bucket", "project/docs/c.docx"),
            ],
            any_order=False,
        )

    async def test_execute_calls_init_project(
        self,
        mock_rag_engine: AsyncMock,
        mock_storage: AsyncMock,
        tmp_path: Path,
    ) -> None:
        """Should call rag_engine.init_project with the working_dir."""
        use_case = IndexFolderUseCase(
            rag_engine=mock_rag_engine,
            storage=mock_storage,
            bucket="my-bucket",
            output_dir=str(tmp_path),
        )
        request = IndexFolderRequest(working_dir="project/docs")

        await use_case.execute(request)

        mock_rag_engine.init_project.assert_called_once_with("project/docs")

    async def test_execute_calls_index_folder(
        self,
        mock_rag_engine: AsyncMock,
        mock_storage: AsyncMock,
        tmp_path: Path,
    ) -> None:
        """Should call rag_engine.index_folder with the local folder path and parameters."""
        output_dir = str(tmp_path)
        use_case = IndexFolderUseCase(
            rag_engine=mock_rag_engine,
            storage=mock_storage,
            bucket="my-bucket",
            output_dir=output_dir,
        )
        request = IndexFolderRequest(
            working_dir="project/docs",
            recursive=False,
            file_extensions=[".pdf"],
        )

        await use_case.execute(request)

        expected_local_folder = os.path.join(output_dir, "project/docs")
        mock_rag_engine.index_folder.assert_called_once_with(
            folder_path=expected_local_folder,
            output_dir=output_dir,
            recursive=False,
            file_extensions=[".pdf"],
            working_dir="project/docs",
        )

    async def test_execute_returns_result(
        self,
        mock_rag_engine: AsyncMock,
        mock_storage: AsyncMock,
        tmp_path: Path,
    ) -> None:
        """Should return the FolderIndexingResult from rag_engine."""
        expected = FolderIndexingResult(
            status=IndexingStatus.SUCCESS,
            message="All done",
            folder_path="/tmp/docs",
            recursive=True,
            stats=FolderIndexingStats(
                total_files=2, files_processed=2, files_failed=0, files_skipped=0
            ),
            processing_time_ms=200.0,
        )
        mock_rag_engine.index_folder.return_value = expected
        use_case = IndexFolderUseCase(
            rag_engine=mock_rag_engine,
            storage=mock_storage,
            bucket="my-bucket",
            output_dir=str(tmp_path),
        )
        request = IndexFolderRequest(working_dir="project/docs")

        result = await use_case.execute(request)

        assert result.status == IndexingStatus.SUCCESS
        assert result.stats.total_files == 2
        assert result.stats.files_processed == 2

    async def test_execute_with_empty_folder(
        self,
        mock_rag_engine: AsyncMock,
        mock_storage: AsyncMock,
        tmp_path: Path,
    ) -> None:
        """Should still call index_folder even when list_objects returns an empty list."""
        mock_storage.list_objects.return_value = []
        use_case = IndexFolderUseCase(
            rag_engine=mock_rag_engine,
            storage=mock_storage,
            bucket="my-bucket",
            output_dir=str(tmp_path),
        )
        request = IndexFolderRequest(working_dir="project/empty")

        await use_case.execute(request)

        mock_storage.get_object.assert_not_called()
        mock_rag_engine.index_folder.assert_called_once()

    # ------------------------------------------------------------------
    # TXT File Support Tests
    # ------------------------------------------------------------------

    async def test_index_folder_with_txt_files(
        self,
        mock_rag_engine: AsyncMock,
        mock_storage: AsyncMock,
        tmp_path: Path,
    ) -> None:
        """Should download and process .txt files when present in folder."""
        mock_storage.list_objects.return_value = [
            "project/docs/file1.txt",
            "project/docs/file2.txt",
            "project/docs/file3.pdf",
        ]
        mock_storage.get_object.return_value = b"text content"
        mock_rag_engine.index_folder.return_value = FolderIndexingResult(
            status=IndexingStatus.SUCCESS,
            message="Folder indexed",
            folder_path=str(tmp_path / "project" / "docs"),
            recursive=False,
            stats=FolderIndexingStats(
                total_files=3, files_processed=3, files_failed=0, files_skipped=0
            ),
            processing_time_ms=150.0,
        )

        use_case = IndexFolderUseCase(
            rag_engine=mock_rag_engine,
            storage=mock_storage,
            bucket="txt-folder-bucket",
            output_dir=str(tmp_path),
        )
        request = IndexFolderRequest(working_dir="project/docs", recursive=False)

        result = await use_case.execute(request)

        # Should download all 3 files (2 txt + 1 pdf)
        assert mock_storage.get_object.call_count == 3
        mock_storage.get_object.assert_any_call(
            "txt-folder-bucket", "project/docs/file1.txt"
        )
        mock_storage.get_object.assert_any_call(
            "txt-folder-bucket", "project/docs/file2.txt"
        )
        mock_storage.get_object.assert_any_call(
            "txt-folder-bucket", "project/docs/file3.pdf"
        )

        assert result.status == IndexingStatus.SUCCESS
        assert result.stats.total_files == 3

    async def test_index_folder_with_file_extensions_filter_txt(
        self,
        mock_rag_engine: AsyncMock,
        mock_storage: AsyncMock,
        tmp_path: Path,
    ) -> None:
        """Should filter to only .txt files when file_extensions=['.txt']."""
        mock_storage.list_objects.return_value = [
            "project/notes/note1.txt",
            "project/notes/note2.txt",
            "project/notes/report.pdf",
            "project/notes/data.xlsx",
        ]
        mock_storage.get_object.return_value = b"note content"
        mock_rag_engine.index_folder.return_value = FolderIndexingResult(
            status=IndexingStatus.SUCCESS,
            message="TXT files indexed",
            folder_path=str(tmp_path / "project" / "notes"),
            recursive=False,
            stats=FolderIndexingStats(
                total_files=2, files_processed=2, files_failed=0, files_skipped=0
            ),
            processing_time_ms=80.0,
        )

        use_case = IndexFolderUseCase(
            rag_engine=mock_rag_engine,
            storage=mock_storage,
            bucket="notes-bucket",
            output_dir=str(tmp_path),
        )
        request = IndexFolderRequest(
            working_dir="project/notes",
            file_extensions=[".txt"],
        )

        result = await use_case.execute(request)

        # Should only download .txt files (note1.txt and note2.txt)
        assert mock_storage.get_object.call_count == 2
        mock_storage.get_object.assert_any_call(
            "notes-bucket", "project/notes/note1.txt"
        )
        mock_storage.get_object.assert_any_call(
            "notes-bucket", "project/notes/note2.txt"
        )

        # Should not download .pdf or .xlsx
        assert not any(
            "report.pdf" in str(call) or "data.xlsx" in str(call)
            for call in mock_storage.get_object.call_args_list
        )

        assert result.status == IndexingStatus.SUCCESS
        assert result.stats.total_files == 2

    async def test_index_folder_with_txt_and_other_extensions(
        self,
        mock_rag_engine: AsyncMock,
        mock_storage: AsyncMock,
        tmp_path: Path,
    ) -> None:
        """Should handle mixed file extensions including .txt."""
        mock_storage.list_objects.return_value = [
            "project/mixed/doc.pdf",
            "project/mixed/notes.txt",
            "project/mixed/data.xlsx",
            "project/mixed/readme.text",  # Another text format
        ]
        mock_storage.get_object.return_value = b"mixed content"
        mock_rag_engine.index_folder.return_value = FolderIndexingResult(
            status=IndexingStatus.SUCCESS,
            message="Mixed files indexed",
            folder_path=str(tmp_path / "project" / "mixed"),
            recursive=False,
            stats=FolderIndexingStats(
                total_files=2, files_processed=2, files_failed=0, files_skipped=0
            ),
            processing_time_ms=120.0,
        )

        use_case = IndexFolderUseCase(
            rag_engine=mock_rag_engine,
            storage=mock_storage,
            bucket="mixed-bucket",
            output_dir=str(tmp_path),
        )
        request = IndexFolderRequest(
            working_dir="project/mixed",
            file_extensions=[".txt", ".text"],
        )

        result = await use_case.execute(request)

        # Should only download .txt and .text files
        assert mock_storage.get_object.call_count == 2
        mock_storage.get_object.assert_any_call(
            "mixed-bucket", "project/mixed/notes.txt"
        )
        mock_storage.get_object.assert_any_call(
            "mixed-bucket", "project/mixed/readme.text"
        )

        assert result.status == IndexingStatus.SUCCESS

    async def test_index_folder_recursive_with_txt_files(
        self,
        mock_rag_engine: AsyncMock,
        mock_storage: AsyncMock,
        tmp_path: Path,
    ) -> None:
        """Should handle .txt files in recursive folder indexing."""
        mock_storage.list_objects.return_value = [
            "project/recursive/chapter1.txt",
            "project/recursive/subdir/chapter2.txt",
            "project/recursive/subdir/nested/chapter3.txt",
        ]
        mock_storage.get_object.return_value = b"chapter content"
        mock_rag_engine.index_folder.return_value = FolderIndexingResult(
            status=IndexingStatus.SUCCESS,
            message="Recursive TXT indexing complete",
            folder_path=str(tmp_path / "project" / "recursive"),
            recursive=True,
            stats=FolderIndexingStats(
                total_files=3, files_processed=3, files_failed=0, files_skipped=0
            ),
            processing_time_ms=200.0,
        )

        use_case = IndexFolderUseCase(
            rag_engine=mock_rag_engine,
            storage=mock_storage,
            bucket="recursive-bucket",
            output_dir=str(tmp_path),
        )
        request = IndexFolderRequest(
            working_dir="project/recursive",
            recursive=True,
        )

        result = await use_case.execute(request)

        # Should download all nested .txt files
        assert mock_storage.get_object.call_count == 3

        # Verify rag_engine.index_folder was called with recursive=True
        mock_rag_engine.index_folder.assert_called_once()
        call_kwargs = mock_rag_engine.index_folder.call_args[1]
        assert call_kwargs["recursive"] is True
        assert call_kwargs["working_dir"] == "project/recursive"

        assert result.status == IndexingStatus.SUCCESS
        assert result.recursive is True
