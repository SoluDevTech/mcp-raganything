import asyncio
import json
import re

from application.responses.classical_query_response import (
    ClassicalChunkResponse,
    ClassicalQueryResponse,
)
from config import ClassicalRAGConfig
from domain.ports.llm_port import LLMPort
from domain.ports.vector_store_port import SearchResult, VectorStorePort


class ClassicalQueryUseCase:
    def __init__(
        self,
        vector_store: VectorStorePort,
        llm: LLMPort,
        config: ClassicalRAGConfig,
    ) -> None:
        self.vector_store = vector_store
        self.llm = llm
        self.config = config

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
            judge_user = f"Query: {query}\nChunk: {chunk.content[:500]}\nScore 0-10:"
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
    ) -> ClassicalQueryResponse:
        if num_variations is None:
            num_variations = self.config.CLASSICAL_NUM_QUERY_VARIATIONS
        if relevance_threshold is None:
            relevance_threshold = self.config.CLASSICAL_RELEVANCE_THRESHOLD

        variations = await self._generate_variations(query, num_variations)
        queries = [query] + variations

        search_tasks = [
            self.vector_store.similarity_search(
                working_dir=working_dir, query=q, top_k=top_k
            )
            for q in queries
        ]
        search_results = await asyncio.gather(*search_tasks)

        all_results: dict[str, SearchResult] = {}
        for results in search_results:
            for r in results:
                if r.chunk_id and r.chunk_id not in all_results:
                    all_results[r.chunk_id] = r

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
                ClassicalChunkResponse(
                    chunk_id=chunk.chunk_id,
                    content=chunk.content,
                    file_path=chunk.file_path,
                    relevance_score=score,
                    metadata=chunk.metadata,
                )
                for chunk, score in scored
                if score >= relevance_threshold
            ],
            key=lambda c: c.relevance_score,
            reverse=True,
        )

        return ClassicalQueryResponse(
            status="success",
            message="",
            queries=queries,
            chunks=scored_chunks,
        )
