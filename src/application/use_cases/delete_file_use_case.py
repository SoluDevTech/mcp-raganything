import logging

from domain.constants import DEFAULT_WORKING_DIR
from domain.logging.messages import LogMessage
from domain.ports.storage_port import StoragePort
from domain.ports.vector_store_port import VectorStorePort

logger = logging.getLogger(__name__)


class DeleteFileUseCase:
    """Delete a single file from the storage bucket."""

    def __init__(
        self,
        storage: StoragePort,
        bucket: str,
        vector_store: VectorStorePort | None = None,
    ) -> None:
        self.storage = storage
        self.bucket = bucket
        self.vector_store = vector_store

    async def execute(
        self, object_path: str, working_dir: str = DEFAULT_WORKING_DIR
    ) -> str:
        """Delete the object at ``object_path`` from the bucket.

        Args:
            object_path: The path/key of the object to delete.
            working_dir: Workspace identifier for the vector store. If empty,
                only MinIO is cleaned (no pgvector cascade).

        Returns:
            The deleted object path.
        """
        logger.info(
            LogMessage.DELETE_FILE_STARTED, self.bucket, object_path, working_dir
        )

        await self.storage.remove_object(self.bucket, object_path)
        logger.info(LogMessage.DELETE_FILE_MINIO_DONE, self.bucket, object_path)

        if self.vector_store is not None and working_dir:
            deleted = await self.vector_store.delete_documents(working_dir, object_path)
            logger.info(
                LogMessage.DELETE_FILE_VECTORS_DONE,
                working_dir,
                object_path,
                deleted,
            )
        else:
            logger.info(LogMessage.DELETE_FILE_VECTORS_SKIPPED, object_path)

        return object_path
