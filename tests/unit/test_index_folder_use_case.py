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
