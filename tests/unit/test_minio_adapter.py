"""Tests for MinioAdapter — the MinIO implementation of StoragePort.

MinIO client is an external dependency (third-party S3-compatible service),
so we mock the Minio client itself while testing our adapter logic.
"""

from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import pytest
from minio.error import S3Error

from domain.ports.storage_port import FileInfo
from infrastructure.storage.minio_adapter import MinioAdapter


@pytest.fixture
def mock_minio_client() -> MagicMock:
    """Provide a mocked Minio client."""
    return MagicMock()


@pytest.fixture
def adapter(mock_minio_client: MagicMock) -> MinioAdapter:
    """Provide a MinioAdapter with mocked client."""
    with patch(
        "infrastructure.storage.minio_adapter.Minio",
        return_value=mock_minio_client,
    ):
        adapter = MinioAdapter(
            host="localhost:9000",
            access="minioadmin",
            secret="minioadmin",
            secure=False,
        )
    # Replace client directly to ensure mock is used
    adapter.client = mock_minio_client
    return adapter


class TestGetObject:
    """Tests for MinioAdapter.get_object."""

    async def test_returns_object_bytes_on_success(
        self, adapter: MinioAdapter, mock_minio_client: MagicMock
    ) -> None:
        """Should return the raw bytes of the requested object."""
        mock_response = MagicMock()
        mock_response.read.return_value = b"file content here"
        mock_response.close = MagicMock()
        mock_response.release_conn = MagicMock()
        mock_minio_client.get_object.return_value = mock_response

        result = await adapter.get_object("my-bucket", "docs/report.pdf")

        assert result == b"file content here"
        mock_response.close.assert_called_once()
        mock_response.release_conn.assert_called_once()

    async def test_raises_file_not_found_for_no_such_key(
        self, adapter: MinioAdapter, mock_minio_client: MagicMock
    ) -> None:
        """Should convert S3Error NoSuchKey to FileNotFoundError."""
        mock_minio_client.get_object.side_effect = S3Error(
            response=None,
            code="NoSuchKey",
            message="The specified key does not exist.",
            resource="resource",
            request_id="request_id",
            host_id="host_id",
        )

        with pytest.raises(FileNotFoundError, match="Object not found"):
            await adapter.get_object("my-bucket", "missing.pdf")

    async def test_raises_file_not_found_for_no_such_bucket(
        self, adapter: MinioAdapter, mock_minio_client: MagicMock
    ) -> None:
        """Should convert S3Error NoSuchBucket to FileNotFoundError."""
        mock_minio_client.get_object.side_effect = S3Error(
            response=None,
            code="NoSuchBucket",
            message="The specified bucket does not exist.",
            resource="resource",
            request_id="request_id",
            host_id="host_id",
        )

        with pytest.raises(FileNotFoundError, match="Object not found"):
            await adapter.get_object("bad-bucket", "any/path")

    async def test_re_raises_other_s3_errors(
        self, adapter: MinioAdapter, mock_minio_client: MagicMock
    ) -> None:
        """Should re-raise S3Error for non-404 error codes like AccessDenied."""
        mock_minio_client.get_object.side_effect = S3Error(
            response=None,
            code="AccessDenied",
            message="Access Denied.",
            resource="resource",
            request_id="request_id",
            host_id="host_id",
        )

        with pytest.raises(S3Error) as exc_info:
            await adapter.get_object("my-bucket", "private/doc.pdf")

        assert exc_info.value.code == "AccessDenied"


class TestListObjects:
    """Tests for MinioAdapter.list_objects."""

    async def test_returns_object_names_filtering_dirs(
        self, adapter: MinioAdapter, mock_minio_client: MagicMock
    ) -> None:
        """Should return only non-directory object names."""
        mock_obj1 = MagicMock()
        mock_obj1.object_name = "docs/report.pdf"
        mock_obj1.is_dir = False

        mock_obj2 = MagicMock()
        mock_obj2.object_name = "docs/"
        mock_obj2.is_dir = True

        mock_obj3 = MagicMock()
        mock_obj3.object_name = "docs/notes.txt"
        mock_obj3.is_dir = False

        mock_minio_client.list_objects.return_value = [mock_obj1, mock_obj2, mock_obj3]

        result = await adapter.list_objects("my-bucket", "docs/", recursive=True)

        assert result == ["docs/report.pdf", "docs/notes.txt"]
        mock_minio_client.list_objects.assert_called_once_with(
            "my-bucket", prefix="docs/", recursive=True
        )

    async def test_returns_empty_list_when_no_objects(
        self, adapter: MinioAdapter, mock_minio_client: MagicMock
    ) -> None:
        """Should return empty list when bucket is empty."""
        mock_minio_client.list_objects.return_value = []

        result = await adapter.list_objects("my-bucket", "", recursive=True)

        assert result == []


class TestListFilesMetadata:
    """Tests for MinioAdapter.list_files_metadata."""

    async def test_returns_file_info_list(
        self, adapter: MinioAdapter, mock_minio_client: MagicMock
    ) -> None:
        """Should return FileInfo list with size and last_modified."""
        dt = datetime(2026, 3, 15, 10, 30, 0, tzinfo=UTC)

        mock_obj = MagicMock()
        mock_obj.object_name = "data/file.csv"
        mock_obj.is_dir = False
        mock_obj.size = 4096
        mock_obj.last_modified = dt

        mock_dir = MagicMock()
        mock_dir.object_name = "data/"
        mock_dir.is_dir = True

        mock_minio_client.list_objects.return_value = [mock_obj, mock_dir]

        result = await adapter.list_files_metadata("my-bucket", "data/", recursive=True)

        assert len(result) == 1
        assert isinstance(result[0], FileInfo)
        assert result[0].object_name == "data/file.csv"
        assert result[0].size == 4096
        assert result[0].last_modified == str(dt)

    async def test_handles_none_size_as_zero(
        self, adapter: MinioAdapter, mock_minio_client: MagicMock
    ) -> None:
        """Should treat None size as 0 in FileInfo."""
        mock_obj = MagicMock()
        mock_obj.object_name = "unknown-size-file"
        mock_obj.is_dir = False
        mock_obj.size = None
        mock_obj.last_modified = datetime(2026, 1, 1, tzinfo=UTC)

        mock_minio_client.list_objects.return_value = [mock_obj]

        result = await adapter.list_files_metadata("my-bucket", "")

        assert result[0].size == 0

    async def test_handles_none_last_modified_as_none(
        self, adapter: MinioAdapter, mock_minio_client: MagicMock
    ) -> None:
        """Should set last_modified to None when MinIO returns None."""
        mock_obj = MagicMock()
        mock_obj.object_name = "no-date-file"
        mock_obj.is_dir = False
        mock_obj.size = 100
        mock_obj.last_modified = None

        mock_minio_client.list_objects.return_value = [mock_obj]

        result = await adapter.list_files_metadata("my-bucket", "")

        assert result[0].last_modified is None


class TestListMinioObjects:
    """Tests for MinioAdapter._list_minio_objects (internal helper)."""

    async def test_raises_file_not_found_for_missing_bucket(
        self, adapter: MinioAdapter, mock_minio_client: MagicMock
    ) -> None:
        """Should convert NoSuchBucket S3Error to FileNotFoundError."""
        mock_minio_client.list_objects.side_effect = S3Error(
            response=None,
            code="NoSuchBucket",
            message="The specified bucket does not exist.",
            resource="resource",
            request_id="request_id",
            host_id="host_id",
        )

        with pytest.raises(FileNotFoundError, match="Bucket not found"):
            await adapter._list_minio_objects("bad-bucket", "", recursive=True)

    async def test_re_raises_non_bucket_s3_errors(
        self, adapter: MinioAdapter, mock_minio_client: MagicMock
    ) -> None:
        """Should re-raise S3Errors that are not NoSuchBucket."""
        mock_minio_client.list_objects.side_effect = S3Error(
            response=None,
            code="InternalError",
            message="We encountered an internal error.",
            resource="resource",
            request_id="request_id",
            host_id="host_id",
        )

        with pytest.raises(S3Error) as exc_info:
            await adapter._list_minio_objects("my-bucket", "", recursive=True)

        assert exc_info.value.code == "InternalError"
