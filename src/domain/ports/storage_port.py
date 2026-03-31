from abc import ABC, abstractmethod


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
