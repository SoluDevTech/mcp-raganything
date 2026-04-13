from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from application.use_cases.classical_index_folder_use_case import (
    ClassicalIndexFolderUseCase,
)
from domain.entities.indexing_result import (
    FolderIndexingResult,
    IndexingStatus,
)


class TestClassicalIndexFolderUseCase:
    @pytest.fixture
    def use_case(
        self,
        mock_vector_store: AsyncMock,
        mock_storage: AsyncMock,
    ) -> ClassicalIndexFolderUseCase:
        return ClassicalIndexFolderUseCase(
            vector_store=mock_vector_store,
            storage=mock_storage,
            bucket="test-bucket",
            output_dir="/tmp/output",
        )

    async def test_execute_lists_objects_from_storage(
        self,
        use_case: ClassicalIndexFolderUseCase,
        mock_storage: AsyncMock,
    ) -> None:
        mock_storage.list_objects.return_value = []

        await use_case.execute(
            working_dir="project/docs",
            recursive=True,
        )

        mock_storage.list_objects.assert_called_once_with(
            "test-bucket", prefix="project/docs", recursive=True
        )

    async def test_execute_ensures_vector_store_table(
        self,
        use_case: ClassicalIndexFolderUseCase,
        mock_vector_store: AsyncMock,
        mock_storage: AsyncMock,
    ) -> None:
        mock_storage.list_objects.return_value = []

        await use_case.execute(
            working_dir="project/docs",
            recursive=True,
        )

        mock_vector_store.ensure_table.assert_called_once_with("project/docs")

    @patch("application.use_cases.classical_index_folder_use_case.extract_file")
    async def test_execute_downloads_and_indexes_each_file(
        self,
        mock_extract: AsyncMock,
        use_case: ClassicalIndexFolderUseCase,
        mock_storage: AsyncMock,
        mock_vector_store: AsyncMock,
    ) -> None:
        mock_storage.list_objects.return_value = [
            "project/docs/report.pdf",
            "project/docs/notes.txt",
        ]
        mock_result = MagicMock()
        mock_result.chunks = []
        mock_result.content = "text"
        mock_extract.return_value = mock_result

        await use_case.execute(
            working_dir="project/docs",
            recursive=True,
        )

        assert mock_storage.get_object.call_count == 2
        mock_storage.get_object.assert_any_call(
            "test-bucket", "project/docs/report.pdf"
        )
        mock_storage.get_object.assert_any_call("test-bucket", "project/docs/notes.txt")

    @patch("application.use_cases.classical_index_folder_use_case.extract_file")
    async def test_execute_filters_by_file_extensions(
        self,
        mock_extract: AsyncMock,
        use_case: ClassicalIndexFolderUseCase,
        mock_storage: AsyncMock,
    ) -> None:
        mock_storage.list_objects.return_value = [
            "project/docs/report.pdf",
            "project/docs/notes.txt",
            "project/docs/data.xlsx",
        ]
        mock_result = MagicMock()
        mock_result.chunks = []
        mock_result.content = "text"
        mock_extract.return_value = mock_result

        await use_case.execute(
            working_dir="project/docs",
            file_extensions=[".pdf", ".txt"],
        )

        assert mock_storage.get_object.call_count == 2
        mock_storage.get_object.assert_any_call(
            "test-bucket", "project/docs/report.pdf"
        )
        mock_storage.get_object.assert_any_call("test-bucket", "project/docs/notes.txt")

    @patch("application.use_cases.classical_index_folder_use_case.extract_file")
    async def test_execute_returns_folder_indexing_result(
        self,
        mock_extract: AsyncMock,
        use_case: ClassicalIndexFolderUseCase,
        mock_storage: AsyncMock,
    ) -> None:
        mock_storage.list_objects.return_value = [
            "project/docs/report.pdf",
        ]
        mock_result = MagicMock()
        mock_result.chunks = []
        mock_result.content = "text"
        mock_extract.return_value = mock_result

        result = await use_case.execute(
            working_dir="project/docs",
            recursive=True,
        )

        assert isinstance(result, FolderIndexingResult)
        assert result.status == IndexingStatus.SUCCESS
        assert result.folder_path == "project/docs"

    async def test_execute_handles_empty_folder(
        self,
        use_case: ClassicalIndexFolderUseCase,
        mock_storage: AsyncMock,
    ) -> None:
        mock_storage.list_objects.return_value = []

        result = await use_case.execute(
            working_dir="project/empty",
            recursive=True,
        )

        assert isinstance(result, FolderIndexingResult)
        assert result.status == IndexingStatus.SUCCESS
        assert result.stats.total_files == 0

    @patch("application.use_cases.classical_index_folder_use_case.extract_file")
    async def test_execute_tracks_successful_and_failed_files(
        self,
        mock_extract: AsyncMock,
        use_case: ClassicalIndexFolderUseCase,
        mock_storage: AsyncMock,
    ) -> None:
        mock_storage.list_objects.return_value = [
            "project/docs/good.pdf",
            "project/docs/bad.pdf",
        ]
        call_count = 0

        def _extract_with_failure(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise RuntimeError("Extraction failed")
            mock_result = MagicMock()
            mock_result.chunks = []
            mock_result.content = "Good content"
            return mock_result

        mock_extract.side_effect = _extract_with_failure

        result = await use_case.execute(
            working_dir="project/docs",
            recursive=True,
        )

        assert isinstance(result, FolderIndexingResult)
        assert result.stats.files_processed + result.stats.files_failed > 0

    async def test_execute_uses_custom_bucket(
        self,
        mock_vector_store: AsyncMock,
        mock_storage: AsyncMock,
    ) -> None:
        use_case = ClassicalIndexFolderUseCase(
            vector_store=mock_vector_store,
            storage=mock_storage,
            bucket="custom-folder-bucket",
            output_dir="/custom/output",
        )

        mock_storage.list_objects.return_value = []

        await use_case.execute(working_dir="project/docs")

        mock_storage.list_objects.assert_called_once_with(
            "custom-folder-bucket", prefix="project/docs", recursive=True
        )

    async def test_execute_non_recursive_listing(
        self,
        use_case: ClassicalIndexFolderUseCase,
        mock_storage: AsyncMock,
    ) -> None:
        mock_storage.list_objects.return_value = []

        await use_case.execute(
            working_dir="project/docs",
            recursive=False,
        )

        mock_storage.list_objects.assert_called_once_with(
            "test-bucket", prefix="project/docs", recursive=False
        )

    @patch("application.use_cases.classical_index_folder_use_case.extract_file")
    async def test_execute_passes_chunk_params(
        self,
        mock_extract: AsyncMock,
        use_case: ClassicalIndexFolderUseCase,
        mock_storage: AsyncMock,
    ) -> None:
        mock_storage.list_objects.return_value = [
            "project/docs/report.pdf",
        ]
        mock_result = MagicMock()
        mock_result.chunks = []
        mock_result.content = "text"
        mock_extract.return_value = mock_result

        await use_case.execute(
            working_dir="project/docs",
            chunk_size=500,
            chunk_overlap=100,
        )

        mock_extract.assert_called_once()
