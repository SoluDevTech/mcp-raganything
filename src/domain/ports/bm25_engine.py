"""BM25 search engine port interface."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class BM25SearchResult:
    """Result from BM25 search."""

    chunk_id: str
    content: str
    file_path: str
    score: float
    metadata: dict[str, Any]


class BM25EnginePort(ABC):
    """Port interface for BM25 full-text search operations."""

    @abstractmethod
    async def search(
        self,
        query: str,
        working_dir: str,
        top_k: int = 10,
    ) -> list[BM25SearchResult]:
        """Search documents using BM25 ranking.

        Args:
            query: Search query string
            working_dir: Project/workspace directory
            top_k: Number of results to return

        Returns:
            List of BM25SearchResult ordered by relevance
        """
        pass

    @abstractmethod
    async def index_document(
        self,
        chunk_id: str,
        content: str,
        file_path: str,
        working_dir: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Index a document chunk for BM25 search.

        Args:
            chunk_id: Unique chunk identifier
            content: Text content to index
            file_path: Path to source file
            working_dir: Project/workspace directory
            metadata: Optional metadata dictionary
        """
        pass

    @abstractmethod
    async def create_index(self, working_dir: str) -> None:
        """Create BM25 index for workspace.

        Args:
            working_dir: Project/workspace directory
        """
        pass

    @abstractmethod
    async def drop_index(self, working_dir: str) -> None:
        """Drop BM25 index for workspace.

        Args:
            working_dir: Project/workspace directory
        """
        pass

    @abstractmethod
    async def close(self) -> None:
        """Close connection pool and cleanup resources.

        Called during application shutdown.
        """
        pass
