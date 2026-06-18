import asyncio
import io
import logging

from minio import Minio
from minio.error import S3Error

from domain.errors.messages import ErrorMessage
from domain.errors.storage import StorageNotFoundError
from domain.logging.messages import LogMessage
from domain.ports.storage_port import FileInfo, StoragePort

logger = logging.getLogger(__name__)


class MinioAdapter(StoragePort):
    """MinIO implementation of the StoragePort."""

    def __init__(
        self, host: str, access: str, secret: str, secure: bool = False
    ) -> None:
        self.client = Minio(
            endpoint=host,
            access_key=access,
            secret_key=secret,
            secure=secure,
        )

    async def get_object(self, bucket: str, object_path: str) -> bytes:
        try:
            loop = asyncio.get_running_loop()
            response = await loop.run_in_executor(
                None, lambda: self.client.get_object(bucket, object_path)
            )
            try:
                return response.read()
            finally:
                response.close()
                response.release_conn()
        except S3Error as e:
            if e.code in ("NoSuchKey", "NoSuchBucket"):
                logger.info(
                    LogMessage.MINIO_OBJECT_NOT_FOUND, bucket, object_path
                )
                raise StorageNotFoundError(
                    ErrorMessage.OBJECT_NOT_FOUND.format(
                        bucket=bucket, path=object_path
                    )
                ) from e
            logger.error(LogMessage.MINIO_RETRIEVE_ERROR, e, exc_info=True)
            raise

    async def put_object(
        self, bucket: str, object_path: str, data: bytes, content_type: str
    ) -> None:
        try:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                None,
                lambda: self.client.put_object(
                    bucket,
                    object_path,
                    io.BytesIO(data),
                    len(data),
                    content_type=content_type,
                ),
            )
        except S3Error as e:
            if e.code == "NoSuchBucket":
                logger.info(LogMessage.MINIO_BUCKET_NOT_FOUND, bucket)
                raise StorageNotFoundError(
                    ErrorMessage.BUCKET_NOT_FOUND.format(bucket=bucket)
                ) from e
            logger.error(LogMessage.MINIO_UPLOAD_ERROR, e, exc_info=True)
            raise

    async def _list_minio_objects(
        self, bucket: str, prefix: str, recursive: bool = True
    ) -> list:
        """Fetch raw MinIO object list, raising FileNotFoundError for missing buckets."""
        try:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                None,
                lambda: list(
                    self.client.list_objects(bucket, prefix=prefix, recursive=recursive)
                ),
            )
        except S3Error as e:
            if e.code == "NoSuchBucket":
                logger.info(LogMessage.MINIO_BUCKET_NOT_FOUND, bucket)
                raise StorageNotFoundError(
                    ErrorMessage.BUCKET_NOT_FOUND.format(bucket=bucket)
                ) from e
            logger.error(LogMessage.MINIO_LIST_ERROR, e, exc_info=True)
            raise

    async def list_objects(
        self, bucket: str, prefix: str, recursive: bool = True
    ) -> list[str]:
        objects = await self._list_minio_objects(bucket, prefix, recursive)
        return [obj.object_name for obj in objects if not obj.is_dir]

    async def list_files_metadata(
        self, bucket: str, prefix: str, recursive: bool = True
    ) -> list[FileInfo]:
        objects = await self._list_minio_objects(bucket, prefix, recursive)
        return [
            FileInfo(
                object_name=obj.object_name,
                size=obj.size or 0,
                last_modified=str(obj.last_modified) if obj.last_modified else None,
            )
            for obj in objects
            if not obj.is_dir
        ]

    async def list_folders(self, bucket: str, prefix: str = "") -> list[str]:
        objects = await self._list_minio_objects(bucket, prefix=prefix, recursive=False)
        return [obj.object_name for obj in objects if obj.is_dir]

    async def ping(self, bucket: str) -> bool:
        try:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                None, lambda: self.client.bucket_exists(bucket)
            )
        except Exception:
            logger.warning(LogMessage.MINIO_HEALTH_CHECK_FAILED, exc_info=True)
            return False
