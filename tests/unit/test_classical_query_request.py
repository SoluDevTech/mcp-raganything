"""Tests for ClassicalQueryRequest model."""

import pytest
from pydantic import ValidationError

from application.requests.classical_query_request import ClassicalQueryRequest


class TestClassicalQueryRequest:
    def test_mode_defaults_to_vector(self) -> None:
        # Arrange & Act
        req = ClassicalQueryRequest(
            working_dir="/tmp/rag/project",
            query="test query",
        )

        # Assert
        assert req.mode == "vector"

    def test_mode_accepts_hybrid(self) -> None:
        # Arrange & Act
        req = ClassicalQueryRequest(
            working_dir="/tmp/rag/project",
            query="test query",
            mode="hybrid",
        )

        # Assert
        assert req.mode == "hybrid"

    def test_mode_rejects_invalid_value(self) -> None:
        # Act & Assert
        with pytest.raises(ValidationError):
            ClassicalQueryRequest(
                working_dir="/tmp/rag/project",
                query="test query",
                mode="bm25",
            )

    def test_mode_accepts_explicit_vector(self) -> None:
        # Arrange & Act
        req = ClassicalQueryRequest(
            working_dir="/tmp/rag/project",
            query="test query",
            mode="vector",
        )

        # Assert
        assert req.mode == "vector"

    def test_defaults_preserved(self) -> None:
        # Arrange & Act
        req = ClassicalQueryRequest(
            working_dir="/tmp/rag/project",
            query="test query",
        )

        # Assert
        assert req.top_k == 10
        assert req.num_variations == 3
        assert req.relevance_threshold == 5.0
        assert req.vector_distance_threshold is None
        assert req.enable_llm_judge is True
        assert req.mode == "vector"
