import asyncio
import logging

from minio import Minio
from minio.error import S3Error

from domain.ports.storage_port import StoragePort

logger = logging.getLogger(__name__)


class MinioAdapter(StoragePort):
    """MinIO implementation of the StoragePort."""

    def __init__(
        self, host: str, access: str, secret: str, secure: bool = False
    ) -> None:
        """
        Initialize the MinIO adapter with connection parameters.

        Args:
            host: The MinIO server endpoint (host:port).
            access: The access key for authentication.
            secret: The secret key for authentication.
            secure: Whether to use HTTPS. Defaults to False.
        """
        self.client = Minio(
            endpoint=host,
            access_key=access,
            secret_key=secret,
            secure=secure,
        )

    async def get_object(self, bucket: str, object_path: str) -> bytes:
        """
        Retrieve an object from MinIO storage.

        Args:
            bucket: The bucket name where the object is stored.
            object_path: The path/key of the object within the bucket.

        Returns:
            The object content as bytes.

        Raises:
            FileNotFoundError: If the object or bucket does not exist.
        """
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
                logger.warning(f"Object not found: bucket={bucket}, path={object_path}")
                raise FileNotFoundError(
                    f"Object not found: bucket={bucket}, path={object_path}"
                ) from None
            logger.error(f"MinIO error retrieving object: {e}", exc_info=True)
            raise

    async def list_objects(
        self, bucket: str, prefix: str, recursive: bool = True
    ) -> list[str]:
        """
        List object keys under a given prefix in MinIO.

        Args:
            bucket: The bucket name to list objects from.
            prefix: The prefix to filter objects by.
            recursive: Whether to list objects recursively.

        Returns:
            A list of object keys (excluding directories).
        """
        loop = asyncio.get_running_loop()
        objects = await loop.run_in_executor(
            None,
            lambda: list(
                self.client.list_objects(bucket, prefix=prefix, recursive=recursive)
            ),
        )
        return [obj.object_name for obj in objects if not obj.is_dir]
