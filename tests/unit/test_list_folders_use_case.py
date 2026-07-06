from unittest.mock import AsyncMock

from application.use_cases.list_folders_use_case import ListFoldersUseCase


class TestListFoldersUseCase:
    async def test_execute_calls_storage_list_folders(
        self, mock_storage: AsyncMock
    ) -> None:
        # Arrange
        mock_storage.list_folders.return_value = ["docs/", "photos/"]
        use_case = ListFoldersUseCase(storage=mock_storage, bucket="test-bucket")

        # Act
        await use_case.execute()

        # Assert
        mock_storage.list_folders.assert_called_once_with("test-bucket", "")

    async def test_execute_with_prefix_passes_prefix_to_storage(
        self, mock_storage: AsyncMock
    ) -> None:
        # Arrange
        mock_storage.list_folders.return_value = ["reports/", "exports/"]
        use_case = ListFoldersUseCase(storage=mock_storage, bucket="test-bucket")

        # Act
        result = await use_case.execute(prefix="docs/")

        # Assert
        mock_storage.list_folders.assert_called_once_with("test-bucket", "docs/")
        assert result == ["reports/", "exports/"]

    async def test_execute_returns_folder_prefixes(
        self, mock_storage: AsyncMock
    ) -> None:
        # Arrange
        expected_folders = ["docs/", "photos/", "reports/"]
        mock_storage.list_folders.return_value = expected_folders
        use_case = ListFoldersUseCase(storage=mock_storage, bucket="test-bucket")

        # Act
        result = await use_case.execute()

        # Assert
        assert result == expected_folders

    async def test_execute_empty_result(self, mock_storage: AsyncMock) -> None:
        # Arrange
        mock_storage.list_folders.return_value = []
        use_case = ListFoldersUseCase(storage=mock_storage, bucket="test-bucket")

        # Act
        result = await use_case.execute()

        # Assert
        assert result == []
