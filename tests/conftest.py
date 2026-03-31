import importlib.util
from pathlib import Path

import pytest

from domain.entities.indexing_result import (
    FileIndexingResult,
    FolderIndexingResult,
    FolderIndexingStats,
    IndexingStatus,
)

# Load external fixtures from tests/fixtures/external.py without __init__.py
_fixtures_path = Path(__file__).parent / "fixtures" / "external.py"
_spec = importlib.util.spec_from_file_location("external_fixtures", _fixtures_path)
_external = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_external)

# Re-export fixtures so pytest discovers them
mock_rag_engine = _external.mock_rag_engine
mock_storage = _external.mock_storage


@pytest.fixture
def sample_file_indexing_result() -> FileIndexingResult:
    """Provide a sample successful file indexing result."""
    return FileIndexingResult(
        status=IndexingStatus.SUCCESS,
        message="Document indexed successfully",
        file_path="/tmp/test/document.pdf",
        file_name="document.pdf",
        processing_time_ms=150.0,
    )


@pytest.fixture
def sample_folder_indexing_result() -> FolderIndexingResult:
    """Provide a sample successful folder indexing result."""
    return FolderIndexingResult(
        status=IndexingStatus.SUCCESS,
        message="Folder indexed successfully",
        folder_path="/tmp/test/documents",
        recursive=True,
        stats=FolderIndexingStats(
            total_files=10,
            files_processed=8,
            files_failed=1,
            files_skipped=1,
        ),
        processing_time_ms=1200.0,
    )
