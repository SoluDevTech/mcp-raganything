from unittest.mock import AsyncMock

import pytest

from application.use_cases.delete_file_use_case import DeleteFileUseCase


class TestDeleteFileUseCase:
    @pytest.fixture
    def use_case(
        self,
        mock_storage: AsyncMock,
        mock_vector_store: AsyncMock,
    ) -> DeleteFileUseCase:
        return DeleteFileUseCase(
            storage=mock_storage,
            bucket="test-bucket",
            vector_store=mock_vector_store,
        )

    @pytest.fixture
    def use_case_without_vector_store(
        self,
        mock_storage: AsyncMock,
    ) -> DeleteFileUseCase:
        return DeleteFileUseCase(
            storage=mock_storage,
            bucket="test-bucket",
            vector_store=None,
        )

    async def test_deletes_file_calls_remove_object_then_vector_store(
        self,
        use_case: DeleteFileUseCase,
        mock_storage: AsyncMock,
        mock_vector_store: AsyncMock,
    ) -> None:
        """Should call storage.remove_object FIRST then vector_store.delete_documents."""
        # Act
        await use_case.execute(
            object_path="docs/report.pdf",
            working_dir="docs/",
        )

        # Assert — MinIO called first
        mock_storage.remove_object.assert_called_once_with(
            "test-bucket",
            "docs/report.pdf",
        )
        # Assert — vector store called second with working_dir and object_path
        mock_vector_store.delete_documents.assert_called_once_with(
            "docs/",
            "docs/report.pdf",
        )

    async def test_skip_vector_store_when_none(
        self,
        use_case_without_vector_store: DeleteFileUseCase,
        mock_storage: AsyncMock,
        mock_vector_store: AsyncMock,
    ) -> None:
        """Should only call storage.remove_object when vector_store is None."""
        # Act
        result = await use_case_without_vector_store.execute(
            object_path="docs/report.pdf",
            working_dir="docs/",
        )

        # Assert
        assert result == "docs/report.pdf"
        mock_storage.remove_object.assert_called_once_with(
            "test-bucket",
            "docs/report.pdf",
        )
        mock_vector_store.delete_documents.assert_not_called()

    async def test_returns_object_path(
        self,
        use_case: DeleteFileUseCase,
    ) -> None:
        """Should return the deleted object path."""
        # Act
        result = await use_case.execute(
            object_path="docs/report.pdf",
            working_dir="docs/",
        )

        # Assert
        assert result == "docs/report.pdf"

    async def test_does_not_delete_vectors_if_minio_fails(
        self,
        use_case: DeleteFileUseCase,
        mock_storage: AsyncMock,
        mock_vector_store: AsyncMock,
    ) -> None:
        """Should NOT call vector_store.delete_documents if storage.remove_object raises."""
        # Arrange — make MinIO raise
        mock_storage.remove_object.side_effect = RuntimeError("MinIO unreachable")

        # Act & Assert — exception should propagate
        with pytest.raises(RuntimeError):
            await use_case.execute(
                object_path="docs/report.pdf",
                working_dir="docs/",
            )

        # Assert — vector store was never called
        mock_vector_store.delete_documents.assert_not_called()
