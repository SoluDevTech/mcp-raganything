from unittest.mock import AsyncMock

from application.use_cases.list_folders_use_case import ListFoldersUseCase


class TestListFoldersUseCase:
    async def test_execute_calls_storage_list_folders(
        self, mock_storage: AsyncMock
    ) -> None:
        mock_storage.list_folders.return_value = ["docs/", "photos/"]
        use_case = ListFoldersUseCase(storage=mock_storage, bucket="test-bucket")

        await use_case.execute(prefix="")

        mock_storage.list_folders.assert_called_once_with("test-bucket", "")

    async def test_execute_returns_folder_prefixes(
        self, mock_storage: AsyncMock
    ) -> None:
        expected_folders = ["docs/", "photos/", "reports/"]
        mock_storage.list_folders.return_value = expected_folders
        use_case = ListFoldersUseCase(storage=mock_storage, bucket="test-bucket")

        result = await use_case.execute(prefix="")

        assert result == expected_folders

    async def test_execute_with_default_prefix(self, mock_storage: AsyncMock) -> None:
        mock_storage.list_folders.return_value = []
        use_case = ListFoldersUseCase(storage=mock_storage, bucket="my-bucket")

        await use_case.execute()

        mock_storage.list_folders.assert_called_once_with("my-bucket", "")

    async def test_execute_empty_result(self, mock_storage: AsyncMock) -> None:
        mock_storage.list_folders.return_value = []
        use_case = ListFoldersUseCase(storage=mock_storage, bucket="test-bucket")

        result = await use_case.execute(prefix="nonexistent/")

        assert result == []
