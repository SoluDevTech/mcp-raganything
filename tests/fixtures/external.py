from unittest.mock import AsyncMock

import pytest

from domain.entities.indexing_result import (
    FileIndexingResult,
    FolderIndexingResult,
    FolderIndexingStats,
    IndexingStatus,
)
from domain.ports.document_reader_port import (
    DocumentContent,
    DocumentMetadata,
    DocumentReaderPort,
)
from domain.ports.llm_port import LLMPort
from domain.ports.rag_engine import RAGEnginePort
from domain.ports.storage_port import FileInfo, StoragePort
from domain.ports.vector_store_port import SearchResult, VectorStorePort


@pytest.fixture
def mock_rag_engine() -> AsyncMock:
    """Provide an AsyncMock of RAGEnginePort for external adapter mocking."""
    mock = AsyncMock(spec=RAGEnginePort)

    mock.index_document.return_value = FileIndexingResult(
        status=IndexingStatus.SUCCESS,
        message="Document indexed successfully",
        file_path="/tmp/documents/report.pdf",
        file_name="report.pdf",
        processing_time_ms=100.0,
    )

    mock.index_folder.return_value = FolderIndexingResult(
        status=IndexingStatus.SUCCESS,
        message="Folder indexed successfully",
        folder_path="/tmp/documents",
        recursive=True,
        stats=FolderIndexingStats(
            total_files=5,
            files_processed=5,
            files_failed=0,
            files_skipped=0,
        ),
        processing_time_ms=500.0,
    )

    mock.query_multimodal.return_value = "Multimodal analysis result"

    return mock


@pytest.fixture
def mock_storage() -> AsyncMock:
    """Provide an AsyncMock of StoragePort for external adapter mocking."""
    mock = AsyncMock(spec=StoragePort)
    mock.get_object.return_value = b"fake file content"
    mock.list_objects.return_value = ["project/doc1.pdf", "project/doc2.pdf"]
    mock.list_files_metadata.return_value = [
        FileInfo(
            object_name="project/doc1.pdf",
            size=1024,
            last_modified="2026-01-01 00:00:00+00:00",
        ),
        FileInfo(
            object_name="project/doc2.pdf",
            size=2048,
            last_modified="2026-01-02 00:00:00+00:00",
        ),
    ]
    mock.list_folders.return_value = ["project/", "documents/"]
    return mock


@pytest.fixture
def mock_document_reader() -> AsyncMock:
    mock = AsyncMock(spec=DocumentReaderPort)
    mock.extract_content.return_value = DocumentContent(
        content="Extracted text content from document.",
        metadata=DocumentMetadata(format_type="pdf", mime_type="application/pdf"),
        tables=[],
    )
    return mock


@pytest.fixture
def mock_vector_store() -> AsyncMock:
    """Provide an AsyncMock of VectorStorePort for external adapter mocking."""
    mock = AsyncMock(spec=VectorStorePort)
    mock.ensure_table.return_value = None
    mock.add_documents.return_value = ["chunk-1", "chunk-2", "chunk-3"]
    mock.similarity_search.return_value = [
        SearchResult(
            chunk_id="chunk-abc123",
            content="Relevant text about the query topic.",
            file_path="/docs/report.pdf",
            score=0.92,
            metadata={"page": 1},
        ),
        SearchResult(
            chunk_id="chunk-def456",
            content="Another relevant chunk of text.",
            file_path="/docs/notes.txt",
            score=0.85,
            metadata={"page": 3},
        ),
    ]
    mock.delete_documents.return_value = 5
    mock.close.return_value = None
    return mock


@pytest.fixture
def mock_llm() -> AsyncMock:
    """Provide an AsyncMock of LLMPort for external adapter mocking."""
    mock = AsyncMock(spec=LLMPort)
    mock.generate.return_value = (
        '["alternative query 1", "alternative query 2", "alternative query 3"]'
    )
    mock.generate_chat.return_value = "LLM generated response text"
    return mock
