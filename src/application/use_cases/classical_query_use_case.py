import asyncio
import json
import logging
import re
from typing import Literal

from application.responses.classical_query_response import (
    ClassicalChunkResponse,
    ClassicalQueryResponse,
)
from config import ClassicalRAGConfig
from domain.ports.bm25_engine import BM25EnginePort
from domain.ports.llm_port import LLMPort
from domain.ports.vector_store_port import SearchResult, VectorStorePort
from infrastructure.rag.rrf_combiner import RRFCombiner

logger = logging.getLogger(__name__)


class ClassicalQueryUseCase:
    def __init__(
        self,
        vector_store: VectorStorePort,
        llm: LLMPort,
        config: ClassicalRAGConfig,
        bm25_engine: BM25EnginePort | None = None,
        rrf_k: int = 60,
    ) -> None:
        self.vector_store = vector_store
        self.llm = llm
        self.config = config
        self.bm25_engine = bm25_engine
        self.rrf_combiner = RRFCombiner(k=rrf_k)

    @staticmethod
    def _extract_json_array(text: str) -> list[str]:
        match = re.search(r"\[.*\]", text, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group(0))
                if isinstance(parsed, list):
                    return [v for v in parsed if isinstance(v, str)]
            except ValueError:
                pass
        return []

    async def _generate_variations(self, query: str, num_variations: int) -> list[str]:
        try:
            system_prompt = "You are a helpful assistant. Generate alternative phrasings of the given query for improved search retrieval."
            user_message = f"Generate {num_variations} alternative versions of this query: {query}. Return ONLY a JSON array of strings."
            response = await self.llm.generate(
                system_prompt=system_prompt, user_message=user_message
            )
            return self._extract_json_array(response)
        except Exception:
            return []

    async def _score_chunk(self, query: str, chunk: SearchResult) -> float:
        try:
            judge_system = "You are a relevance judge. Score how relevant the given chunk is to the query on a scale of 0 to 10. Return ONLY a number."
            judge_user = f"Query: {query}\nChunk: {chunk.content}\nScore 0-10:"
            score_response = await self.llm.generate(
                system_prompt=judge_system, user_message=judge_user
            )
            nums = re.findall(r"[\d.]+", score_response.strip())
            return float(nums[0]) if nums else 0.0
        except (ValueError, TypeError, IndexError):
            return 0.0

    async def execute(
        self,
        working_dir: str,
        query: str,
        top_k: int = 10,
        num_variations: int | None = None,
        relevance_threshold: float | None = None,
        vector_distance_threshold: float | None = None,
        enable_llm_judge: bool = True,
        mode: Literal["vector", "hybrid"] = "vector",
    ) -> ClassicalQueryResponse:
        if num_variations is None:
            num_variations = self.config.CLASSICAL_NUM_QUERY_VARIATIONS
        if relevance_threshold is None:
            relevance_threshold = self.config.CLASSICAL_RELEVANCE_THRESHOLD

        if mode == "hybrid" and self.bm25_engine is not None:
            return await self._execute_hybrid(
                working_dir=working_dir,
                query=query,
                top_k=top_k,
                num_variations=num_variations,
                relevance_threshold=relevance_threshold,
                vector_distance_threshold=vector_distance_threshold,
                enable_llm_judge=enable_llm_judge,
            )

        if mode == "hybrid" and self.bm25_engine is None:
            logger.warning("BM25 unavailable, falling back to vector mode")

        return await self._execute_vector(
            working_dir=working_dir,
            query=query,
            top_k=top_k,
            num_variations=num_variations,
            relevance_threshold=relevance_threshold,
            vector_distance_threshold=vector_distance_threshold,
            enable_llm_judge=enable_llm_judge,
        )

    async def _execute_vector(
        self,
        working_dir: str,
        query: str,
        top_k: int,
        num_variations: int,
        relevance_threshold: float,
        vector_distance_threshold: float | None,
        enable_llm_judge: bool,
    ) -> ClassicalQueryResponse:
        variations = (
            await self._generate_variations(query, num_variations)
            if num_variations > 1
            else []
        )

        queries = [query] + variations

        search_tasks = [
            self.vector_store.similarity_search(
                working_dir=working_dir,
                query=q,
                top_k=top_k,
                score_threshold=vector_distance_threshold,
            )
            for q in queries
        ]
        search_results = await asyncio.gather(*search_tasks)

        all_results: dict[str, SearchResult] = {}
        for results in search_results:
            for r in results:
                if r.chunk_id and r.chunk_id not in all_results:
                    all_results[r.chunk_id] = r

        scored_chunks = await self._score_and_filter(
            query, all_results, relevance_threshold, enable_llm_judge
        )

        return ClassicalQueryResponse(
            status="success",
            message="",
            queries=queries,
            chunks=scored_chunks,
            mode="vector",
        )

    async def _execute_hybrid(
        self,
        working_dir: str,
        query: str,
        top_k: int,
        num_variations: int,
        relevance_threshold: float,
        vector_distance_threshold: float | None,
        enable_llm_judge: bool,
    ) -> ClassicalQueryResponse:
        variations = (
            await self._generate_variations(query, num_variations)
            if num_variations > 1
            else []
        )
        queries = [query] + variations

        search_tasks = [
            self.vector_store.similarity_search(
                working_dir=working_dir,
                query=q,
                top_k=top_k,
                score_threshold=vector_distance_threshold,
            )
            for q in queries
        ]

        bm25_coro = self.bm25_engine.search(query, working_dir, top_k=top_k * 2)
        vector_coro = asyncio.gather(*search_tasks)

        gather_results = await asyncio.gather(
            bm25_coro, vector_coro, return_exceptions=True
        )

        bm25_result = gather_results[0]
        vector_result = gather_results[1]

        bm25_hits = []
        if isinstance(bm25_result, Exception):
            logger.error("BM25 search failed in hybrid mode: %s", bm25_result)
        elif isinstance(bm25_result, list):
            bm25_hits = bm25_result

        vector_hits: list[SearchResult] = []
        if isinstance(vector_result, Exception):
            logger.error("Vector search failed in hybrid mode: %s", vector_result)
        else:
            all_results: dict[str, SearchResult] = {}
            for results in vector_result:
                for r in results:
                    if r.chunk_id and r.chunk_id not in all_results:
                        all_results[r.chunk_id] = r
            vector_hits = list(all_results.values())

        combined = self.rrf_combiner.combine_classical(
            bm25_results=bm25_hits,
            vector_results=vector_hits,
            top_k=top_k,
        )

        combined_map: dict[str, SearchResult] = {}
        hybrid_meta: dict[str, dict] = {}
        for r in combined:
            sr = SearchResult(
                chunk_id=r.chunk_id,
                content=r.content,
                file_path=r.file_path,
                score=r.combined_score,
                metadata=r.metadata,
            )
            combined_map[r.chunk_id] = sr
            hybrid_meta[r.chunk_id] = {
                "bm25_score": r.bm25_score,
                "vector_score": r.vector_score,
                "combined_score": r.combined_score,
            }

        scored_chunks = await self._score_and_filter(
            query, combined_map, relevance_threshold, enable_llm_judge, hybrid_meta
        )

        return ClassicalQueryResponse(
            status="success",
            message="",
            queries=queries,
            chunks=scored_chunks,
            mode="hybrid",
        )

    async def _score_and_filter(
        self,
        query: str,
        all_results: dict[str, SearchResult],
        relevance_threshold: float,
        enable_llm_judge: bool,
        hybrid_meta: dict[str, dict] | None = None,
    ) -> list[ClassicalChunkResponse]:
        if enable_llm_judge:
            sem = asyncio.Semaphore(5)

            async def _bounded_score(c: SearchResult) -> tuple[SearchResult, float]:
                async with sem:
                    score = await self._score_chunk(query, c)
                    return (c, score)

            scored = await asyncio.gather(
                *(_bounded_score(c) for c in all_results.values())
            )

            scored_chunks = sorted(
                [
                    self._build_chunk_response(
                        chunk=chunk,
                        relevance_score=score,
                        hybrid_meta=hybrid_meta,
                    )
                    for chunk, score in scored
                    if score >= relevance_threshold
                ],
                key=lambda c: c.relevance_score,
                reverse=True,
            )
        else:
            if hybrid_meta:
                relevance_base = [
                    self._build_chunk_response(
                        chunk=chunk,
                        relevance_score=round(chunk.score, 4),
                        hybrid_meta=hybrid_meta,
                    )
                    for chunk in all_results.values()
                ]
            else:
                relevance_base = [
                    self._build_chunk_response(
                        chunk=chunk,
                        relevance_score=round(1.0 - chunk.score, 4),
                        hybrid_meta=None,
                    )
                    for chunk in all_results.values()
                ]
            scored_chunks = sorted(
                relevance_base,
                key=lambda c: c.relevance_score,
                reverse=True,
            )

        return scored_chunks

    @staticmethod
    def _build_chunk_response(
        chunk: SearchResult,
        relevance_score: float,
        hybrid_meta: dict[str, dict] | None = None,
    ) -> ClassicalChunkResponse:
        meta = hybrid_meta.get(chunk.chunk_id, {}) if hybrid_meta else {}
        return ClassicalChunkResponse(
            chunk_id=chunk.chunk_id,
            content=chunk.content,
            file_path=chunk.file_path,
            relevance_score=relevance_score,
            metadata=chunk.metadata,
            bm25_score=meta.get("bm25_score"),
            vector_score=meta.get("vector_score"),
            combined_score=meta.get("combined_score"),
        )
