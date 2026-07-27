from unittest.mock import AsyncMock

import pytest

from application.use_cases.create_folder_use_case import CreateFolderUseCase


class TestCreateFolderUseCase:
    @pytest.fixture
    def use_case(self, mock_storage: AsyncMock) -> CreateFolderUseCase:
        return CreateFolderUseCase(storage=mock_storage, bucket="test-bucket")

    async def test_creates_folder_calls_put_object_with_trailing_slash(
        self, use_case: CreateFolderUseCase, mock_storage: AsyncMock
    ) -> None:
        await use_case.execute(prefix="docs")

        mock_storage.put_object.assert_called_once_with(
            "test-bucket",
            "docs/",
            b"",
            "application/octet-stream",
        )

    async def test_normalizes_prefix_without_trailing_slash(
        self, use_case: CreateFolderUseCase, mock_storage: AsyncMock
    ) -> None:
        await use_case.execute(prefix="docs")

        args = mock_storage.put_object.call_args.args
        assert args[1] == "docs/"

    async def test_preserves_existing_trailing_slash(
        self, use_case: CreateFolderUseCase, mock_storage: AsyncMock
    ) -> None:
        await use_case.execute(prefix="docs/")

        args = mock_storage.put_object.call_args.args
        assert args[1] == "docs/"

    async def test_returns_normalized_prefix(
        self, use_case: CreateFolderUseCase
    ) -> None:
        result = await use_case.execute(prefix="docs")

        assert result == "docs/"
