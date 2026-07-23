import logging

from domain.errors.file import FileValidationError
from domain.errors.messages import ErrorMessage
from domain.logging.messages import LogMessage
from domain.ports.storage_port import StoragePort
from domain.ports.vector_store_port import VectorStorePort

logger = logging.getLogger(__name__)


class DeleteFolderUseCase:
    """Recursively delete a folder and all its contents from the storage bucket."""

    def __init__(
        self,
        storage: StoragePort,
        bucket: str,
        vector_store: VectorStorePort | None = None,
    ) -> None:
        self.storage = storage
        self.bucket = bucket
        self.vector_store = vector_store

    async def execute(self, prefix: str) -> str:
        """Delete all objects under the given prefix.

        Args:
            prefix: Folder prefix (auto-appended with ``/`` if missing).

        Returns:
            The normalized prefix ending with a trailing slash.
        """
        if not prefix or not prefix.strip():
            raise FileValidationError(ErrorMessage.PREFIX_MUST_BE_RELATIVE_SHORT)
        if prefix and not prefix.endswith("/"):
            prefix += "/"

        logger.info(LogMessage.DELETE_FOLDER_STARTED, self.bucket, prefix)

        await self.storage.remove_prefix(self.bucket, prefix)
        logger.info(LogMessage.DELETE_FOLDER_MINIO_DONE, self.bucket, prefix)

        if self.vector_store is not None:
            deleted = await self.vector_store.delete_by_prefix(prefix, prefix)
            logger.info(LogMessage.DELETE_FOLDER_VECTORS_DONE, prefix, prefix, deleted)
        else:
            logger.info(LogMessage.DELETE_FOLDER_VECTORS_SKIPPED, prefix)

        return prefix
