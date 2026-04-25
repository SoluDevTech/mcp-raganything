"""Abstract port for vector store operations."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class SearchResult:
    """A single search result from the vector store."""

    chunk_id: str
    content: str
    file_path: str
    score: float
    metadata: dict[str, str | int | float | None] = field(default_factory=dict)


class VectorStorePort(ABC):
    """Port interface for vector store operations (classical RAG)."""

    @abstractmethod
    async def ensure_table(self, working_dir: str) -> None:
        """Create the vector store table for the given workspace if it doesn't exist."""
        pass

    @abstractmethod
    async def add_documents(
        self,
        working_dir: str,
        documents: list[tuple[str, str, dict[str, str | int | None]]],
    ) -> list[str]:
        """Add documents to the vector store.

        Args:
            working_dir: Workspace identifier (used to select the table).
            documents: List of (content, file_path, metadata) tuples.

        Returns:
            List of document IDs.
        """
        pass

    @abstractmethod
    async def similarity_search(
        self,
        working_dir: str,
        query: str,
        top_k: int = 10,
        score_threshold: float | None = None,
    ) -> list[SearchResult]:
        """Search for similar documents using vector similarity.

        Args:
            working_dir: Workspace identifier.
            query: The search query text.
            top_k: Maximum number of results to return.
            score_threshold: Maximum cosine distance threshold. Results with
                distance > threshold are excluded. None disables filtering.

        Returns:
            List of SearchResult objects sorted by relevance.
        """
        pass

    @abstractmethod
    async def delete_documents(self, working_dir: str, file_path: str) -> int:
        """Delete all documents for a given file_path from the vector store.

        Args:
            working_dir: Workspace identifier.
            file_path: The file path to delete documents for.

        Returns:
            Number of documents deleted.
        """
        pass

    @abstractmethod
    async def close(self) -> None:
        """Close the vector store connection pool and cleanup resources."""
        pass
