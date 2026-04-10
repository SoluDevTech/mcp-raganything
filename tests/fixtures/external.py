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
from domain.ports.rag_engine import RAGEnginePort
from domain.ports.storage_port import FileInfo, StoragePort


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
