from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class FileInfo:
    object_name: str
    size: int
    last_modified: str | None = None


class StoragePort(ABC):
    """Abstract port defining the interface for object storage operations."""

    @abstractmethod
    async def get_object(self, bucket: str, object_path: str) -> bytes:
        """
        Retrieve an object from storage.

        Args:
            bucket: The bucket name where the object is stored.
            object_path: The path/key of the object within the bucket.

        Returns:
            The object content as bytes.

        Raises:
            FileNotFoundError: If the object does not exist.
        """
        pass

    @abstractmethod
    async def list_objects(
        self, bucket: str, prefix: str, recursive: bool = True
    ) -> list[str]:
        """
        List object keys under a given prefix.

        Args:
            bucket: The bucket name to list objects from.
            prefix: The prefix to filter objects by.
            recursive: Whether to list objects recursively.

        Returns:
            A list of object keys matching the prefix.
        """
        pass

    @abstractmethod
    async def put_object(
        self, bucket: str, object_path: str, data: bytes, content_type: str
    ) -> None:
        """
        Store an object in the given bucket.

        Args:
            bucket: The bucket name to store the object in.
            object_path: The path/key for the object within the bucket.
            data: The raw bytes to store.
            content_type: The MIME type of the content.

        Raises:
            FileNotFoundError: If the bucket does not exist.
        """
        pass

    @abstractmethod
    async def list_files_metadata(
        self, bucket: str, prefix: str, recursive: bool = True
    ) -> list[FileInfo]:
        """
        List files with metadata under a given prefix.

        Args:
            bucket: The bucket name to list files from.
            prefix: The prefix to filter files by.
            recursive: Whether to list files recursively.

        Returns:
            A list of FileInfo objects with object_name, size, and last_modified.
        """
        pass

    @abstractmethod
    async def list_folders(self, bucket: str, prefix: str = "") -> list[str]:
        """
        List folder prefixes in the bucket under a given prefix.

        Args:
            bucket: The bucket name to list folders from.
            prefix: The prefix to list folders under (default: root).

        Returns:
            A list of folder prefix strings (e.g., ['docs/', 'photos/']).
        """
        pass

    @abstractmethod
    async def ping(self, bucket: str) -> bool:
        """
        Check connectivity to the storage backend.

        Args:
            bucket: The bucket name to check.

        Returns:
            True if the backend is reachable, False otherwise.
        """
        pass
