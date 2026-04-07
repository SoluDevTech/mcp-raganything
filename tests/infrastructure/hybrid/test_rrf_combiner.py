"""Tests for Reciprocal Rank Fusion combiner."""

from domain.ports.bm25_engine import BM25SearchResult
from infrastructure.hybrid.rrf_combiner import RRFCombiner


def test_rrf_combiner_initialization():
    """RRFCombiner should initialize with default k=60."""
    combiner = RRFCombiner()
    assert combiner.k == 60


def test_rrf_combiner_custom_k():
    """RRFCombiner should accept custom k parameter."""
    combiner = RRFCombiner(k=100)
    assert combiner.k == 100


def test_combine_results_basic():
    """RRF should combine ranks correctly."""
    combiner = RRFCombiner(k=60)

    # Mock BM25 results (already sorted by score)
    bm25_results = [
        BM25SearchResult(
            chunk_id="1",
            content="BM25 result 1",
            file_path="/a.pdf",
            score=5.0,
            metadata={},
        ),
        BM25SearchResult(
            chunk_id="2",
            content="BM25 result 2",
            file_path="/b.pdf",
            score=4.0,
            metadata={},
        ),
    ]

    # Mock vector results
    vector_results = {
        "data": {
            "chunks": [
                {
                    "chunk_id": "2",
                    "content": "Vector result 1",
                    "file_path": "/b.pdf",
                },
                {
                    "chunk_id": "3",
                    "content": "Vector result 2",
                    "file_path": "/c.pdf",
                },
            ]
        }
    }

    combined = combiner.combine(bm25_results, vector_results, top_k=10)

    # Check that results are combined
    assert len(combined) == 3  # chunk_ids: 1, 2, 3

    # Check that all results have combined scores
    for result in combined:
        assert result.combined_score > 0
        assert result.vector_score >= 0
        assert result.bm25_score >= 0


def test_combine_results_respects_top_k():
    """RRF should respect top_k parameter."""
    combiner = RRFCombiner()

    bm25_results = [
        BM25SearchResult(
            chunk_id=str(i),
            content=f"BM25 {i}",
            file_path="/a.pdf",
            score=1.0,
            metadata={},
        )
        for i in range(20)
    ]

    vector_results = {
        "data": {
            "chunks": [
                {"chunk_id": str(i), "content": f"Vector {i}", "file_path": "/b.pdf"}
                for i in range(20)
            ]
        }
    }

    combined = combiner.combine(bm25_results, vector_results, top_k=5)

    assert len(combined) == 5


def test_combine_results_sorted_by_score():
    """RRF results should be sorted by combined_score descending."""
    combiner = RRFCombiner()

    bm25_results = [
        BM25SearchResult(
            chunk_id="1", content="BM25", file_path="/a.pdf", score=5.0, metadata={}
        ),
        BM25SearchResult(
            chunk_id="2", content="BM25", file_path="/b.pdf", score=4.0, metadata={}
        ),
    ]

    vector_results = {
        "data": {
            "chunks": [
                {"chunk_id": "2", "content": "Vector", "file_path": "/b.pdf"},
                {"chunk_id": "1", "content": "Vector", "file_path": "/a.pdf"},
            ]
        }
    }

    combined = combiner.combine(bm25_results, vector_results, top_k=10)

    # Check sorted order
    scores = [r.combined_score for r in combined]
    assert scores == sorted(scores, reverse=True)


def test_rrf_formula():
    """RRF formula should be: 1/(k + rank)."""
    combiner = RRFCombiner(k=60)

    # Item appears at rank 1 in BM25, rank 3 in vector
    # Expected: 1/(60+1) + 1/(60+3) = 0.01639 + 0.01587 = 0.03226
    bm25_results = [
        BM25SearchResult(
            chunk_id="1",
            content="BM25 rank 1",
            file_path="/a.pdf",
            score=5.0,
            metadata={},
        ),
    ]

    vector_results = {
        "data": {
            "chunks": [
                {
                    "chunk_id": "other",
                    "content": "Vector rank 1",
                    "file_path": "/x.pdf",
                },
                {
                    "chunk_id": "other2",
                    "content": "Vector rank 2",
                    "file_path": "/y.pdf",
                },
                {"chunk_id": "1", "content": "Vector rank 3", "file_path": "/a.pdf"},
            ]
        }
    }

    combined = combiner.combine(bm25_results, vector_results, top_k=10)

    # Find our item
    item = next(r for r in combined if r.chunk_id == "1")

    # Check RRF calculation
    expected_bm25_score = 1 / (60 + 1)  # rank 1 in BM25
    expected_vector_score = 1 / (60 + 3)  # rank 3 in vector

    assert abs(item.bm25_score - expected_bm25_score) < 0.0001
    assert abs(item.vector_score - expected_vector_score) < 0.0001
    assert (
        abs(item.combined_score - (expected_bm25_score + expected_vector_score))
        < 0.0001
    )


def test_combine_only_bm25_results():
    """RRF should handle case where only BM25 has results."""
    combiner = RRFCombiner()

    bm25_results = [
        BM25SearchResult(
            chunk_id="1",
            content="BM25 only",
            file_path="/a.pdf",
            score=5.0,
            metadata={},
        )
    ]

    vector_results = {"data": {"chunks": []}}

    combined = combiner.combine(bm25_results, vector_results, top_k=10)

    assert len(combined) == 1
    assert combined[0].chunk_id == "1"
    assert combined[0].bm25_score > 0
    assert combined[0].vector_score == 0


def test_combine_only_vector_results():
    """RRF should handle case where only vector has results."""
    combiner = RRFCombiner()

    bm25_results = []

    vector_results = {
        "data": {
            "chunks": [
                {"chunk_id": "1", "content": "Vector only", "file_path": "/a.pdf"}
            ]
        }
    }

    combined = combiner.combine(bm25_results, vector_results, top_k=10)

    assert len(combined) == 1
    assert combined[0].chunk_id == "1"
    assert combined[0].bm25_score == 0
    assert combined[0].vector_score > 0
