from unittest.mock import AsyncMock

import pytest

from application.use_cases.delete_folder_use_case import DeleteFolderUseCase
from domain.errors.file import FileValidationError


class TestDeleteFolderUseCase:
    @pytest.fixture
    def use_case(
        self,
        mock_storage: AsyncMock,
        mock_vector_store: AsyncMock,
    ) -> DeleteFolderUseCase:
        return DeleteFolderUseCase(
            storage=mock_storage,
            bucket="test-bucket",
            vector_store=mock_vector_store,
        )

    @pytest.fixture
    def use_case_without_vector_store(
        self,
        mock_storage: AsyncMock,
    ) -> DeleteFolderUseCase:
        return DeleteFolderUseCase(
            storage=mock_storage,
            bucket="test-bucket",
            vector_store=None,
        )

    async def test_deletes_folder_calls_remove_prefix_then_vector_store(
        self,
        use_case: DeleteFolderUseCase,
        mock_storage: AsyncMock,
        mock_vector_store: AsyncMock,
    ) -> None:
        """Should call storage.remove_prefix FIRST then vector_store.delete_by_prefix."""
        # Act
        await use_case.execute(prefix="docs")

        # Assert — MinIO called first with normalized prefix
        mock_storage.remove_prefix.assert_called_once_with(
            "test-bucket",
            "docs/",
        )
        # Assert — vector store called second with the same prefix
        mock_vector_store.delete_by_prefix.assert_called_once_with(
            "docs/",
            "docs/",
        )

    async def test_skip_vector_store_when_none(
        self,
        use_case_without_vector_store: DeleteFolderUseCase,
        mock_storage: AsyncMock,
        mock_vector_store: AsyncMock,
    ) -> None:
        """Should only call storage.remove_prefix when vector_store is None."""
        # Act
        result = await use_case_without_vector_store.execute(prefix="docs")

        # Assert
        assert result == "docs/"
        mock_storage.remove_prefix.assert_called_once_with("test-bucket", "docs/")
        mock_vector_store.delete_by_prefix.assert_not_called()

    async def test_normalizes_prefix_with_trailing_slash(
        self,
        use_case: DeleteFolderUseCase,
        mock_storage: AsyncMock,
    ) -> None:
        """Should append trailing slash to the prefix passed to storage."""
        # Act
        await use_case.execute(prefix="docs")

        # Assert
        args = mock_storage.remove_prefix.call_args.args
        assert args[1] == "docs/"

    async def test_returns_normalized_prefix(
        self,
        use_case: DeleteFolderUseCase,
    ) -> None:
        """Should return the normalized prefix ending with a slash."""
        # Act
        result = await use_case.execute(prefix="docs")

        # Assert
        assert result == "docs/"

    async def test_raises_on_empty_prefix(
        self,
        use_case: DeleteFolderUseCase,
    ) -> None:
        """Should raise FileValidationError for empty prefix."""
        # Act & Assert
        with pytest.raises(FileValidationError):
            await use_case.execute(prefix="")

    async def test_raises_on_whitespace_only_prefix(
        self,
        use_case: DeleteFolderUseCase,
    ) -> None:
        """Should raise FileValidationError for whitespace-only prefix."""
        # Act & Assert
        with pytest.raises(FileValidationError):
            await use_case.execute(prefix="   ")

    async def test_does_not_delete_vectors_if_minio_fails(
        self,
        use_case: DeleteFolderUseCase,
        mock_storage: AsyncMock,
        mock_vector_store: AsyncMock,
    ) -> None:
        """Should NOT call vector_store.delete_by_prefix if storage.remove_prefix raises."""
        # Arrange — make MinIO raise
        mock_storage.remove_prefix.side_effect = RuntimeError("MinIO unreachable")

        # Act & Assert — exception should propagate
        with pytest.raises(RuntimeError):
            await use_case.execute(prefix="docs")

        # Assert — vector store was never called
        mock_vector_store.delete_by_prefix.assert_not_called()
