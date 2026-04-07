"""Reciprocal Rank Fusion (RRF) combiner for hybrid search."""

from dataclasses import dataclass
from typing import Any

from domain.ports.bm25_engine import BM25SearchResult


@dataclass
class HybridSearchResult:
    """Combined result from BM25 and vector search."""

    chunk_id: str
    content: str
    file_path: str
    vector_score: float
    bm25_score: float
    combined_score: float
    metadata: dict[str, Any]
    bm25_rank: int | None = None
    vector_rank: int | None = None


class RRFCombiner:
    """Reciprocal Rank Fusion algorithm for combining search results.

    RRF formula: score = Σ (1 / (k + rank_i))
    where k is a constant (default 60) and rank_i is the rank in list i.

    This is a simple and effective method for combining ranked lists
    that doesn't require score normalization.
    """

    def __init__(self, k: int = 60):
        """Initialize RRF combiner.

        Args:
            k: RRF constant (default 60, industry standard)
        """
        self.k = k

    def combine(
        self,
        bm25_results: list[BM25SearchResult],
        vector_results: dict,
        top_k: int = 10,
    ) -> list[HybridSearchResult]:
        """Combine BM25 and vector search results using RRF.

        Args:
            bm25_results: Results from BM25 search (already ranked)
            vector_results: Results from vector search (already ranked)
            top_k: Number of results to return

        Returns:
            Combined results sorted by combined_score descending
        """
        scores: dict[str, dict[str, Any]] = {}

        # Process BM25 results
        for rank, result in enumerate(bm25_results, start=1):
            chunk_id = result.chunk_id
            if chunk_id not in scores:
                scores[chunk_id] = {
                    "content": result.content,
                    "file_path": result.file_path,
                    "metadata": result.metadata,
                    "bm25_score": 0.0,
                    "vector_score": 0.0,
                    "bm25_rank": rank,
                    "vector_rank": None,
                }
            else:
                scores[chunk_id]["bm25_rank"] = min(scores[chunk_id]["bm25_rank"], rank)

            scores[chunk_id]["bm25_score"] = 1.0 / (self.k + scores[chunk_id]["bm25_rank"])

        # Process vector results
        chunks = vector_results.get("data", {}).get("chunks", [])
        for rank, chunk in enumerate(chunks, start=1):
            chunk_id = chunk.get("reference_id") or chunk.get("chunk_id")
            if chunk_id is None:
                continue

            if chunk_id not in scores:
                scores[chunk_id] = {
                    "content": chunk.get("content", ""),
                    "file_path": chunk.get("file_path", ""),
                    "metadata": chunk.get("metadata", {}),
                    "bm25_score": 0.0,
                    "vector_score": 0.0,
                    "bm25_rank": None,
                    "vector_rank": rank,
                }
            else:
                existing = scores[chunk_id]["vector_rank"]
                scores[chunk_id]["vector_rank"] = min(existing, rank) if existing is not None else rank

            actual_rank = scores[chunk_id]["vector_rank"]
            if actual_rank is not None:
                scores[chunk_id]["vector_score"] = 1.0 / (self.k + actual_rank)

        # Calculate combined scores and create results
        results = [
            HybridSearchResult(
                chunk_id=chunk_id,
                content=data["content"],
                file_path=data["file_path"],
                vector_score=data["vector_score"],
                bm25_score=data["bm25_score"],
                combined_score=data["bm25_score"] + data["vector_score"],
                metadata=data["metadata"],
                bm25_rank=data["bm25_rank"],
                vector_rank=data["vector_rank"],
            )
            for chunk_id, data in scores.items()
        ]

        # Sort by combined score (descending)
        results.sort(key=lambda x: x.combined_score, reverse=True)

        return results[:top_k]
