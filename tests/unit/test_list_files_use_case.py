from unittest.mock import AsyncMock

from application.use_cases.list_files_use_case import ListFilesUseCase
from domain.ports.storage_port import FileInfo


class TestListFilesUseCase:
    async def test_execute_calls_storage_list_files_metadata(
        self, mock_storage: AsyncMock
    ) -> None:
        mock_storage.list_files_metadata.return_value = [
            FileInfo(
                object_name="docs/report.pdf", size=1024, last_modified="2026-01-01"
            ),
        ]
        use_case = ListFilesUseCase(storage=mock_storage, bucket="test-bucket")

        await use_case.execute(prefix="docs/", recursive=True)

        mock_storage.list_files_metadata.assert_called_once_with(
            "test-bucket", "docs/", True
        )

    async def test_execute_returns_file_infos(self, mock_storage: AsyncMock) -> None:
        expected_files = [
            FileInfo(
                object_name="docs/report.pdf", size=1024, last_modified="2026-01-01"
            ),
            FileInfo(
                object_name="docs/notes.txt", size=512, last_modified="2026-01-02"
            ),
        ]
        mock_storage.list_files_metadata.return_value = expected_files
        use_case = ListFilesUseCase(storage=mock_storage, bucket="test-bucket")

        result = await use_case.execute(prefix="docs/")

        assert len(result) == 2
        assert result[0].object_name == "docs/report.pdf"
        assert result[0].size == 1024
        assert result[1].object_name == "docs/notes.txt"

    async def test_execute_with_default_prefix(self, mock_storage: AsyncMock) -> None:
        mock_storage.list_files_metadata.return_value = []
        use_case = ListFilesUseCase(storage=mock_storage, bucket="my-bucket")

        await use_case.execute()

        mock_storage.list_files_metadata.assert_called_once_with("my-bucket", "", True)

    async def test_execute_non_recursive(self, mock_storage: AsyncMock) -> None:
        mock_storage.list_files_metadata.return_value = []
        use_case = ListFilesUseCase(storage=mock_storage, bucket="my-bucket")

        await use_case.execute(prefix="docs/", recursive=False)

        mock_storage.list_files_metadata.assert_called_once_with(
            "my-bucket", "docs/", False
        )
