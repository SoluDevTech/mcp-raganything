"""Tests for BM25EnginePort interface."""

import pytest

from domain.ports.bm25_engine import BM25EnginePort, BM25SearchResult


def test_bm25_engine_port_is_abstract():
    """BM25EnginePort should be abstract and not instantiable."""
    # Act & Assert
    with pytest.raises(TypeError, match="Can't instantiate abstract class"):
        BM25EnginePort()


def test_bm25_engine_port_has_required_methods():
    """BM25EnginePort should define required abstract methods."""
    # Assert
    assert hasattr(BM25EnginePort, "search")
    assert hasattr(BM25EnginePort, "index_document")
    assert hasattr(BM25EnginePort, "create_index")
    assert hasattr(BM25EnginePort, "drop_index")
    assert hasattr(BM25EnginePort, "close")


def test_bm25_search_result_dataclass():
    """BM25SearchResult should be a dataclass with required fields."""
    # Arrange & Act
    result = BM25SearchResult(
        chunk_id="123",
        content="test content",
        file_path="/test/doc.pdf",
        score=0.95,
        metadata={"page": 1},
    )

    # Assert
    assert result.chunk_id == "123"
    assert result.content == "test content"
    assert result.file_path == "/test/doc.pdf"
    assert result.score == 0.95
    assert result.metadata == {"page": 1}
