import os
from unittest.mock import MagicMock

import pytest

from domain.errors.classical import ClassicalConfigError
from domain.services.classical_helpers import (
    build_documents_from_extraction,
    validate_path,
)


class TestValidatePath:
    def test_simple_filename(self, tmp_path):
        # Arrange
        output_dir = str(tmp_path)

        # Act
        result = validate_path(output_dir, "report.pdf")

        # Assert
        assert result == os.path.join(output_dir, "report.pdf")

    def test_subdirectory_filename(self, tmp_path):
        # Arrange
        output_dir = str(tmp_path)

        # Act
        result = validate_path(output_dir, "subdir/report.pdf")

        # Assert
        assert result == os.path.join(output_dir, "subdir", "report.pdf")

    def test_path_traversal_raises(self, tmp_path):
        # Arrange
        output_dir = str(tmp_path)

        # Act & Assert
        with pytest.raises(ClassicalConfigError, match="escapes output directory"):
            validate_path(output_dir, "../../etc/passwd")

    def test_absolute_path_raises(self, tmp_path):
        # Arrange
        output_dir = str(tmp_path)

        # Act & Assert
        with pytest.raises(ClassicalConfigError, match="escapes output directory"):
            validate_path(output_dir, "/etc/passwd")

    def test_dotdot_in_middle_raises(self, tmp_path):
        # Arrange
        output_dir = str(tmp_path)

        # Act & Assert
        with pytest.raises(ClassicalConfigError, match="escapes output directory"):
            validate_path(output_dir, "docs/../../etc/shadow")


class TestBuildDocumentsFromExtraction:
    def test_with_chunks(self):
        # Arrange
        chunk1 = MagicMock()
        chunk1.content = "chunk one text"
        chunk2 = MagicMock()
        chunk2.content = "chunk two text"
        result = MagicMock()
        result.chunks = [chunk1, chunk2]
        result.content = ""

        # Act
        documents = build_documents_from_extraction(result, "doc.pdf")

        # Assert
        assert len(documents) == 2
        assert documents[0] == ("chunk one text", "doc.pdf", {"chunk_index": 0})
        assert documents[1] == ("chunk two text", "doc.pdf", {"chunk_index": 1})

    def test_with_no_chunks_but_content(self):
        # Arrange
        result = MagicMock()
        result.chunks = None
        result.content = "full document text"

        # Act
        documents = build_documents_from_extraction(result, "doc.pdf")

        # Assert
        assert len(documents) == 1
        assert documents[0] == ("full document text", "doc.pdf", {})

    def test_with_no_chunks_and_whitespace_content(self):
        # Arrange
        result = MagicMock()
        result.chunks = None
        result.content = "   \n\t  "

        # Act
        documents = build_documents_from_extraction(result, "doc.pdf")

        # Assert
        assert documents == []

    def test_with_no_chunks_and_empty_content(self):
        # Arrange
        result = MagicMock()
        result.chunks = None
        result.content = ""

        # Act
        documents = build_documents_from_extraction(result, "doc.pdf")

        # Assert
        assert documents == []

    def test_with_empty_chunks_list(self):
        # Arrange
        result = MagicMock()
        result.chunks = []
        result.content = "fallthrough content"

        # Act
        documents = build_documents_from_extraction(result, "doc.pdf")

        # Assert
        assert len(documents) == 1
        assert documents[0] == ("fallthrough content", "doc.pdf", {})
