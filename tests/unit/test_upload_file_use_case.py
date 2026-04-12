"""TDD tests for UploadFileUseCase — the implementation does not exist yet."""

from unittest.mock import AsyncMock

import pytest

from application.use_cases.upload_file_use_case import UploadFileUseCase
from domain.ports.storage_port import FileInfo


class TestUploadFileUseCase:
    """Tests for the upload file use case.

    The use case will:
    - Depend on StoragePort (abstract port)
    - Construct the full object path as prefix + file_name
    - Call storage.put_object(bucket, object_path, file_data, content_type)
    - Return a FileInfo with object_name, size (len of data), last_modified=None
    """

    @pytest.fixture
    def use_case(self, mock_storage: AsyncMock) -> UploadFileUseCase:
        return UploadFileUseCase(storage=mock_storage, bucket="test-bucket")

    async def test_successful_upload_with_prefix_calls_put_object(
        self, use_case: UploadFileUseCase, mock_storage: AsyncMock
    ) -> None:
        """Should call storage.put_object with prefix + file_name."""
        file_data = b"hello world"
        file_name = "report.pdf"
        prefix = "documents/"
        content_type = "application/pdf"

        await use_case.execute(
            file_data=file_data,
            file_name=file_name,
            prefix=prefix,
            content_type=content_type,
        )

        mock_storage.put_object.assert_called_once_with(
            "test-bucket",
            "documents/report.pdf",
            b"hello world",
            "application/pdf",
        )

    async def test_successful_upload_returns_file_info(
        self, use_case: UploadFileUseCase
    ) -> None:
        """Should return a FileInfo with object_name, size, and last_modified=None."""
        file_data = b"hello world"
        file_name = "report.pdf"
        prefix = "documents/"
        content_type = "application/pdf"

        result = await use_case.execute(
            file_data=file_data,
            file_name=file_name,
            prefix=prefix,
            content_type=content_type,
        )

        assert isinstance(result, FileInfo)
        assert result.object_name == "documents/report.pdf"
        assert result.size == len(b"hello world")
        assert result.last_modified is None

    async def test_upload_with_empty_prefix_uses_just_file_name(
        self, use_case: UploadFileUseCase, mock_storage: AsyncMock
    ) -> None:
        """When prefix is empty, object_path should be just the file_name."""
        file_data = b"data"
        file_name = "image.png"
        prefix = ""
        content_type = "image/png"

        result = await use_case.execute(
            file_data=file_data,
            file_name=file_name,
            prefix=prefix,
            content_type=content_type,
        )

        assert result.object_name == "image.png"
        mock_storage.put_object.assert_called_once_with(
            "test-bucket",
            "image.png",
            b"data",
            "image/png",
        )

    async def test_upload_with_prefix_not_ending_in_slash_still_works(
        self, use_case: UploadFileUseCase, mock_storage: AsyncMock
    ) -> None:
        """Prefix without trailing slash should have slash appended."""
        file_data = b"content"
        file_name = "file.txt"
        prefix = "folder"
        content_type = "text/plain"

        result = await use_case.execute(
            file_data=file_data,
            file_name=file_name,
            prefix=prefix,
            content_type=content_type,
        )

        assert result.object_name == "folder/file.txt"
        mock_storage.put_object.assert_called_once_with(
            "test-bucket",
            "folder/file.txt",
            b"content",
            "text/plain",
        )

    async def test_file_not_found_error_from_storage_propagates(
        self, use_case: UploadFileUseCase, mock_storage: AsyncMock
    ) -> None:
        """FileNotFoundError raised by storage should propagate to the caller."""
        mock_storage.put_object.side_effect = FileNotFoundError("bucket not found")

        with pytest.raises(FileNotFoundError, match="bucket not found"):
            await use_case.execute(
                file_data=b"data",
                file_name="test.txt",
                prefix="",
                content_type="text/plain",
            )
