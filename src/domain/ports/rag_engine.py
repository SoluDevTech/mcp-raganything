from abc import ABC, abstractmethod

from domain.entities.indexing_result import FileIndexingResult, FolderIndexingResult


class RAGEnginePort(ABC):
    """Port interface for RAG engine operations."""

    @abstractmethod
    def init_project(self, working_dir: str) -> None:
        """Initialize the RAG engine for a specific project/workspace."""
        pass

    @abstractmethod
    async def index_document(
        self, file_path: str, file_name: str, output_dir: str, working_dir: str = ""
    ) -> FileIndexingResult:
        pass

    @abstractmethod
    async def index_folder(
        self,
        folder_path: str,
        output_dir: str,
        recursive: bool = True,
        file_extensions: list[str] | None = None,
        working_dir: str = "",
    ) -> FolderIndexingResult:
        pass

    @abstractmethod
    async def query(self, query: str, mode: str = "naive", top_k: int = 10, working_dir: str = "") -> dict:
        pass
