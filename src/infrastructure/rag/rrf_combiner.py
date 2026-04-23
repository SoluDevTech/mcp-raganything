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
    reference_id: str | None = None
    bm25_rank: int | None = None
    vector_rank: int | None = None


class RRFCombiner:
    """Reciprocal Rank Fusion algorithm for combining search results.

    RRF formula: score = Σ (1 / (k + rank_i))
    where k is a constant (default 60) and rank_i is the rank in list i.
    """

    def __init__(self, k: int = 60):
        self.k = k

    def _add_bm25_result(
        self, scores: dict[str, dict[str, Any]], rank: int, result: BM25SearchResult
    ) -> None:
        chunk_id = result.chunk_id
        if chunk_id not in scores:
            scores[chunk_id] = {
                "content": result.content,
                "file_path": result.file_path,
                "metadata": result.metadata,
                "bm25_score": 0.0,
                "vector_score": 0.0,
                "reference_id": None,
                "bm25_rank": rank,
                "vector_rank": None,
            }
        else:
            scores[chunk_id]["bm25_rank"] = min(scores[chunk_id]["bm25_rank"], rank)
        scores[chunk_id]["bm25_score"] = 1.0 / (self.k + scores[chunk_id]["bm25_rank"])

    def _resolve_chunk_id(self, chunk: dict[str, Any]) -> tuple[str, str | None]:
        raw_chunk_id = chunk.get("chunk_id")
        raw_ref_id = chunk.get("reference_id")
        return raw_chunk_id or raw_ref_id, raw_ref_id

    def _add_vector_result(
        self, scores: dict[str, dict[str, Any]], rank: int, chunk: dict[str, Any]
    ) -> None:
        chunk_id, reference_id = self._resolve_chunk_id(chunk)
        if not chunk_id:
            return
        if chunk_id not in scores:
            scores[chunk_id] = {
                "content": chunk.get("content", ""),
                "file_path": chunk.get("file_path", ""),
                "metadata": chunk.get("metadata", {}),
                "bm25_score": 0.0,
                "vector_score": 0.0,
                "reference_id": reference_id,
                "bm25_rank": None,
                "vector_rank": rank,
            }
        else:
            existing = scores[chunk_id]["vector_rank"]
            scores[chunk_id]["vector_rank"] = (
                min(existing, rank) if existing is not None else rank
            )
            if reference_id:
                scores[chunk_id]["reference_id"] = reference_id

        actual_rank = scores[chunk_id]["vector_rank"]
        if actual_rank is not None:
            scores[chunk_id]["vector_score"] = 1.0 / (self.k + actual_rank)

    def combine(
        self,
        bm25_results: list[BM25SearchResult],
        vector_results: dict,
        top_k: int = 10,
    ) -> list[HybridSearchResult]:
        """Combine BM25 and vector search results using RRF."""
        scores: dict[str, dict[str, Any]] = {}

        for rank, result in enumerate(bm25_results, start=1):
            self._add_bm25_result(scores, rank, result)

        chunks = vector_results.get("data", {}).get("chunks", [])
        for rank, chunk in enumerate(chunks, start=1):
            self._add_vector_result(scores, rank, chunk)

        results = [
            HybridSearchResult(
                chunk_id=chunk_id,
                content=data["content"],
                file_path=data["file_path"],
                vector_score=data["vector_score"],
                bm25_score=data["bm25_score"],
                combined_score=data["bm25_score"] + data["vector_score"],
                metadata=data["metadata"],
                reference_id=data["reference_id"],
                bm25_rank=data["bm25_rank"],
                vector_rank=data["vector_rank"],
            )
            for chunk_id, data in scores.items()
        ]

        results.sort(key=lambda x: x.combined_score, reverse=True)
        return results[:top_k]
