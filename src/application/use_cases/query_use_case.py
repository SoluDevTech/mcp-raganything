"""Query use case with hybrid+ mode support."""

import asyncio
import logging
from typing import Literal

from domain.ports.bm25_engine import BM25EnginePort, BM25SearchResult
from domain.ports.rag_engine import RAGEnginePort
from infrastructure.hybrid.rrf_combiner import RRFCombiner

logger = logging.getLogger(__name__)


class QueryUseCase:
    """Use case for querying the RAG knowledge base."""

    def __init__(
        self,
        rag_engine: RAGEnginePort,
        bm25_engine: BM25EnginePort | None = None,
        rrf_k: int = 60,
    ):
        """Initialize use case.

        Args:
            rag_engine: RAG engine for vector search
            bm25_engine: BM25 engine for full-text search (optional)
            rrf_k: RRF constant for combining results
        """
        self.rag_engine = rag_engine
        self.bm25_engine = bm25_engine
        self.rrf_combiner = RRFCombiner(k=rrf_k)

    async def execute(
        self,
        working_dir: str,
        query: str,
        mode: Literal[
            "naive", "local", "global", "hybrid", "hybrid+", "mix", "bypass", "bm25"
        ] = "naive",
        top_k: int = 10,
    ) -> dict:
        """Execute search query.

        Args:
            working_dir: Project/workspace directory
            query: Search query string
            mode: Search mode
                - "naive": Vector search only
                - "local": Local knowledge graph search
                - "global": Global knowledge graph search
                - "hybrid": Local + global knowledge graph
                - "hybrid+": BM25 + vector search (parallel)
                - "mix": Knowledge graph + vector chunks
                - "bypass": Direct LLM query
                - "bm25": BM25 search only
            top_k: Number of results to return

        Returns:
            Search results
        """
        self.rag_engine.init_project(working_dir)

        if mode == "bm25":
            if self.bm25_engine is None:
                return {
                    "status": "error",
                    "message": "BM25 engine not available. Please configure pg_textsearch extension.",
                    "data": {},
                }

            results = await self.bm25_engine.search(query, working_dir, top_k)
            return self._format_bm25_results(results)

        if mode == "hybrid+":
            if self.bm25_engine is None:
                return await self.rag_engine.query(
                    query=query, mode="naive", top_k=top_k, working_dir=working_dir
                )

            bm25_results, vector_results = await asyncio.gather(
                self.bm25_engine.search(query, working_dir, top_k=top_k * 2),
                self.rag_engine.query(
                    query=query, mode="naive", top_k=top_k * 2, working_dir=working_dir
                ),
                return_exceptions=True,
            )

            bm25_hits: list[BM25SearchResult] = (
                bm25_results if isinstance(bm25_results, list) else []
            )
            if isinstance(bm25_results, Exception):
                logger.error("BM25 search failed in hybrid+ mode: %s", bm25_results)

            if isinstance(vector_results, Exception):
                logger.error("Vector search failed in hybrid+ mode: %s", vector_results)
                raise vector_results

            combined_results = self.rrf_combiner.combine(
                bm25_results=bm25_hits,
                vector_results=vector_results,  # type: ignore[arg-type]
                top_k=top_k,
            )

            return self._format_hybrid_results(combined_results)

        return await self.rag_engine.query(
            query=query, mode=mode, top_k=top_k, working_dir=working_dir
        )

    def _format_bm25_results(self, results: list) -> dict:
        """Format BM25 results to match API response format."""
        chunks = [
            {
                "reference_id": r.chunk_id,
                "content": r.content,
                "file_path": r.file_path,
                "chunk_id": r.chunk_id,
            }
            for r in results
        ]
        return {
            "status": "success",
            "message": "",
            "data": {
                "entities": [],
                "relationships": [],
                "chunks": chunks,
                "references": [],
            },
            "metadata": {
                "query_mode": "bm25",
                "total_results": len(results),
            },
        }

    def _format_hybrid_results(self, results: list) -> dict:
        """Format hybrid results to match API response format."""
        return {
            "status": "success",
            "message": "",
            "data": {
                "entities": [],
                "relationships": [],
                "chunks": [
                    {
                        "reference_id": r.reference_id,
                        "content": r.content,
                        "file_path": r.file_path,
                        "chunk_id": r.chunk_id,
                    }
                    for r in results
                ],
                "references": [],
            },
            "metadata": {
                "query_mode": "hybrid+",
                "total_results": len(results),
                "rrf_k": self.rrf_combiner.k,
            },
        }
