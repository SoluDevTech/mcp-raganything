"""Unit tests for MinioAdapter remove_object and remove_prefix."""

from unittest.mock import MagicMock, patch

import pytest

from infrastructure.storage.minio_adapter import MinioAdapter


class TestMinioAdapterRemoveObject:
    """Tests for MinioAdapter.remove_object."""

    @pytest.fixture
    def adapter(self) -> MinioAdapter:
        return MinioAdapter(
            host="localhost:9000",
            access="minioadmin",
            secret="minioadmin",
            secure=False,
        )

    @patch("infrastructure.storage.minio_adapter.S3Error")
    def test_remove_object_calls_client_with_correct_args(
        self, _mock_s3_error: MagicMock, adapter: MinioAdapter
    ) -> None:
        adapter.client = MagicMock()
        adapter.client.remove_object = MagicMock()

        import asyncio

        asyncio.run(adapter.remove_object("test-bucket", "docs/report.pdf"))

        adapter.client.remove_object.assert_called_once_with(
            "test-bucket", "docs/report.pdf"
        )

    def test_remove_object_raises_storage_not_found_on_no_such_bucket(
        self, adapter: MinioAdapter
    ) -> None:
        from minio.error import S3Error

        adapter.client = MagicMock()
        resp = MagicMock()
        resp.status = 404
        mock_error = S3Error(resp, "NoSuchBucket", "Bucket not found", None, None, None)
        adapter.client.remove_object = MagicMock(side_effect=mock_error)

        from domain.errors.storage import StorageNotFoundError

        with pytest.raises(StorageNotFoundError):
            import asyncio

            asyncio.run(adapter.remove_object("missing-bucket", "file.txt"))


class TestMinioAdapterRemovePrefix:
    """Tests for MinioAdapter.remove_prefix."""

    @pytest.fixture
    def adapter(self) -> MinioAdapter:
        return MinioAdapter(
            host="localhost:9000",
            access="minioadmin",
            secret="minioadmin",
            secure=False,
        )

    def test_remove_prefix_lists_and_deletes_recursively(
        self, adapter: MinioAdapter
    ) -> None:
        adapter.client = MagicMock()

        mock_obj1 = MagicMock()
        mock_obj1.object_name = "docs/file1.pdf"
        mock_obj1.is_dir = False
        mock_obj2 = MagicMock()
        mock_obj2.object_name = "docs/sub/file2.pdf"
        mock_obj2.is_dir = False

        adapter.client.list_objects = MagicMock(
            return_value=iter([mock_obj1, mock_obj2])
        )
        adapter.client.remove_objects = MagicMock(return_value=iter([]))
        adapter.client.remove_object = MagicMock()

        import asyncio

        asyncio.run(adapter.remove_prefix("test-bucket", "docs/"))

        adapter.client.list_objects.assert_called_once_with(
            "test-bucket", prefix="docs/", recursive=True
        )
        adapter.client.remove_objects.assert_called_once()
