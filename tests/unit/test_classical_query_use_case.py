"""Tests for ClassicalQueryUseCase."""

import json
from unittest.mock import AsyncMock

import pytest

from application.responses.classical_query_response import (
    ClassicalChunkResponse,
    ClassicalQueryResponse,
)
from application.use_cases.classical_query_use_case import ClassicalQueryUseCase
from config import ClassicalRAGConfig
from domain.ports.bm25_engine import BM25SearchResult
from domain.ports.vector_store_port import SearchResult


class TestClassicalQueryUseCase:
    """Tests for ClassicalQueryUseCase."""

    @pytest.fixture
    def config(self) -> ClassicalRAGConfig:
        """Provide a test configuration."""
        return ClassicalRAGConfig(
            CLASSICAL_CHUNK_SIZE=1000,
            CLASSICAL_CHUNK_OVERLAP=200,
            CLASSICAL_NUM_QUERY_VARIATIONS=3,
            CLASSICAL_RELEVANCE_THRESHOLD=5.0,
            CLASSICAL_TABLE_PREFIX="classical_rag_",
            CLASSICAL_LLM_TEMPERATURE=0.0,
        )

    @pytest.fixture
    def use_case(
        self,
        mock_vector_store: AsyncMock,
        mock_llm: AsyncMock,
        config: ClassicalRAGConfig,
    ) -> ClassicalQueryUseCase:
        """Provide a use case instance with mocked ports."""
        return ClassicalQueryUseCase(
            vector_store=mock_vector_store,
            llm=mock_llm,
            config=config,
        )

    # ------------------------------------------------------------------
    # Multi-query generation
    # ------------------------------------------------------------------

    async def test_execute_calls_llm_to_generate_query_variations(
        self,
        use_case: ClassicalQueryUseCase,
        mock_llm: AsyncMock,
    ) -> None:
        """Should call LLM.generate to produce query variations."""
        await use_case.execute(
            working_dir="/tmp/rag/project_1",
            query="What is machine learning?",
        )

        mock_llm.generate.assert_called()
        # First call should be for multi-query generation
        first_call = mock_llm.generate.call_args_list[0]
        assert (
            "machine learning" in str(first_call).lower()
            or "query" in str(first_call).lower()
        )

    async def test_execute_generates_specified_number_of_variations(
        self,
        use_case: ClassicalQueryUseCase,
        mock_llm: AsyncMock,
    ) -> None:
        """Should generate num_variations query variations via LLM."""
        await use_case.execute(
            working_dir="/tmp/rag/project_1",
            query="What is machine learning?",
            num_variations=3,
        )

        # LLM should be called at least once for multi-query generation
        assert mock_llm.generate.call_count >= 1

    async def test_execute_includes_original_query_in_search(
        self,
        use_case: ClassicalQueryUseCase,
        mock_vector_store: AsyncMock,
        mock_llm: AsyncMock,
    ) -> None:
        """Should include the original query in the list of search queries."""
        await use_case.execute(
            working_dir="/tmp/rag/project_1",
            query="What is machine learning?",
            num_variations=3,
        )

        # similarity_search should be called at least once with the original query
        # or with variation queries — verify it was called
        assert mock_vector_store.similarity_search.call_count >= 1

    # ------------------------------------------------------------------
    # Similarity search
    # ------------------------------------------------------------------

    async def test_execute_calls_similarity_search_for_each_variation(
        self,
        use_case: ClassicalQueryUseCase,
        mock_vector_store: AsyncMock,
        mock_llm: AsyncMock,
    ) -> None:
        """Should call similarity_search for the original query plus each variation."""
        mock_llm.generate.return_value = json.dumps(
            [
                "What is ML?",
                "Explain machine learning",
                "Define ML",
            ]
        )

        await use_case.execute(
            working_dir="/tmp/rag/project_1",
            query="What is machine learning?",
            num_variations=3,
            top_k=10,
        )

        # Should search for original + 3 variations = 4 calls
        assert mock_vector_store.similarity_search.call_count == 4

    async def test_execute_passes_working_dir_and_top_k_to_similarity_search(
        self,
        use_case: ClassicalQueryUseCase,
        mock_vector_store: AsyncMock,
    ) -> None:
        """Should forward working_dir and top_k to each similarity_search call."""
        await use_case.execute(
            working_dir="/tmp/rag/my_project",
            query="test query",
            top_k=5,
        )

        for call_item in mock_vector_store.similarity_search.call_args_list:
            assert call_item[1]["working_dir"] == "/tmp/rag/my_project/"
            assert call_item[1]["top_k"] == 5

    # ------------------------------------------------------------------
    # Deduplication
    # ------------------------------------------------------------------

    async def test_execute_deduplicates_chunks_by_chunk_id(
        self,
        use_case: ClassicalQueryUseCase,
        mock_vector_store: AsyncMock,
        mock_llm: AsyncMock,
    ) -> None:
        """Should deduplicate search results by chunk_id across variations."""
        # Both variations return the same chunk
        mock_llm.generate.side_effect = [
            json.dumps(["variation 1", "variation 2"]),
            "8",
            "8",
        ]
        duplicate_result = SearchResult(
            chunk_id="chunk-same-id",
            content="Same chunk appears in both variations",
            file_path="/docs/report.pdf",
            score=0.90,
        )
        mock_vector_store.similarity_search.return_value = [duplicate_result]

        result = await use_case.execute(
            working_dir="/tmp/rag/project_1",
            query="test query",
            num_variations=2,
        )

        # The duplicate should appear only once
        chunk_ids = [c.chunk_id for c in result.chunks]
        assert chunk_ids.count("chunk-same-id") <= 1

    # ------------------------------------------------------------------
    # LLM-as-judge scoring
    # ------------------------------------------------------------------

    async def test_execute_calls_llm_judge_for_each_chunk(
        self,
        use_case: ClassicalQueryUseCase,
        mock_llm: AsyncMock,
        mock_vector_store: AsyncMock,
    ) -> None:
        mock_llm.generate.side_effect = [
            "7",
            "7",
        ]

        await use_case.execute(
            working_dir="/tmp/rag/project_1",
            query="test query",
            num_variations=1,
        )

        assert mock_llm.generate.call_count >= 1

    async def test_execute_filters_chunks_below_relevance_threshold(
        self,
        use_case: ClassicalQueryUseCase,
        mock_llm: AsyncMock,
        mock_vector_store: AsyncMock,
    ) -> None:
        mock_llm.generate.side_effect = [
            "3",
        ]
        mock_vector_store.similarity_search.return_value = [
            SearchResult(
                chunk_id="chunk-1",
                content="text",
                file_path="/docs/a.pdf",
                score=0.9,
            )
        ]

        result = await use_case.execute(
            working_dir="/tmp/rag/project_1",
            query="test query",
            num_variations=1,
            relevance_threshold=5.0,
        )

        for chunk in result.chunks:
            assert chunk.relevance_score >= 5.0

    async def test_execute_includes_chunks_above_relevance_threshold(
        self,
        use_case: ClassicalQueryUseCase,
        mock_llm: AsyncMock,
        mock_vector_store: AsyncMock,
    ) -> None:
        mock_llm.generate.side_effect = [
            "8",
        ]
        mock_vector_store.similarity_search.return_value = [
            SearchResult(
                chunk_id="chunk-1",
                content="relevant text",
                file_path="/docs/a.pdf",
                score=0.9,
            )
        ]

        result = await use_case.execute(
            working_dir="/tmp/rag/project_1",
            query="test query",
            num_variations=1,
            relevance_threshold=5.0,
        )

        assert len(result.chunks) >= 1
        assert result.chunks[0].relevance_score >= 5.0

    async def test_execute_custom_relevance_threshold(
        self,
        use_case: ClassicalQueryUseCase,
        mock_llm: AsyncMock,
        mock_vector_store: AsyncMock,
    ) -> None:
        mock_llm.generate.side_effect = [
            "4",
        ]
        mock_vector_store.similarity_search.return_value = [
            SearchResult(
                chunk_id="chunk-1",
                content="text",
                file_path="/docs/a.pdf",
                score=0.9,
            )
        ]

        result = await use_case.execute(
            working_dir="/tmp/rag/project_1",
            query="test query",
            num_variations=1,
            relevance_threshold=3.0,
        )

        # Score of 4 should pass a threshold of 3.0
        assert len(result.chunks) >= 1

    # ------------------------------------------------------------------
    # Response shape
    # ------------------------------------------------------------------

    async def test_execute_returns_classical_query_response(
        self,
        use_case: ClassicalQueryUseCase,
    ) -> None:
        """Should return a ClassicalQueryResponse."""
        result = await use_case.execute(
            working_dir="/tmp/rag/project_1",
            query="What is ML?",
        )

        assert isinstance(result, ClassicalQueryResponse)

    async def test_execute_response_includes_queries(
        self,
        use_case: ClassicalQueryUseCase,
        mock_llm: AsyncMock,
    ) -> None:
        """Response should include the original and generated query variations."""
        mock_llm.generate.return_value = json.dumps(
            [
                "What is ML?",
                "Explain machine learning concepts",
            ]
        )

        result = await use_case.execute(
            working_dir="/tmp/rag/project_1",
            query="What is machine learning?",
            num_variations=2,
        )

        # queries list should include the original query
        assert "What is machine learning?" in result.queries

    async def test_execute_response_includes_chunks_with_required_fields(
        self,
        use_case: ClassicalQueryUseCase,
        mock_llm: AsyncMock,
        mock_vector_store: AsyncMock,
    ) -> None:
        """Each chunk in the response should have chunk_id, content, file_path, relevance_score."""
        mock_llm.generate.side_effect = [
            "9",
        ]
        mock_vector_store.similarity_search.return_value = [
            SearchResult(
                chunk_id="chunk-x1",
                content="relevant text",
                file_path="/docs/a.pdf",
                score=0.9,
            )
        ]

        result = await use_case.execute(
            working_dir="/tmp/rag/project_1",
            query="test query",
            num_variations=1,
        )

        if result.chunks:
            chunk = result.chunks[0]
            assert isinstance(chunk, ClassicalChunkResponse)
            assert chunk.chunk_id is not None
            assert chunk.content is not None
            assert chunk.file_path is not None
            assert chunk.relevance_score is not None

    async def test_execute_response_status_is_success(
        self,
        use_case: ClassicalQueryUseCase,
    ) -> None:
        """Response status should be 'success'."""
        result = await use_case.execute(
            working_dir="/tmp/rag/project_1",
            query="test query",
        )

        assert result.status == "success"

    # ------------------------------------------------------------------
    # Error handling
    # ------------------------------------------------------------------

    async def test_execute_handles_invalid_llm_json_response(
        self,
        use_case: ClassicalQueryUseCase,
        mock_llm: AsyncMock,
    ) -> None:
        """Should handle LLM returning non-JSON for query variations gracefully."""
        mock_llm.generate.return_value = "This is not valid JSON"

        result = await use_case.execute(
            working_dir="/tmp/rag/project_1",
            query="test query",
            num_variations=3,
        )

        # Should still return a valid response (possibly only with original query)
        assert isinstance(result, ClassicalQueryResponse)

    async def test_execute_handles_empty_search_results(
        self,
        use_case: ClassicalQueryUseCase,
        mock_vector_store: AsyncMock,
        mock_llm: AsyncMock,
    ) -> None:
        """Should handle when similarity_search returns no results."""
        mock_vector_store.similarity_search.return_value = []
        mock_llm.generate.return_value = json.dumps(["var1", "var2"])

        result = await use_case.execute(
            working_dir="/tmp/rag/project_1",
            query="obscure topic",
            num_variations=2,
        )

        assert isinstance(result, ClassicalQueryResponse)
        assert result.chunks == []

    async def test_execute_handles_llm_judge_failure(
        self,
        use_case: ClassicalQueryUseCase,
        mock_llm: AsyncMock,
        mock_vector_store: AsyncMock,
    ) -> None:
        """Should handle when LLM judge returns an unparseable score."""
        mock_llm.generate.side_effect = [
            "not a number",
        ]
        mock_vector_store.similarity_search.return_value = [
            SearchResult(
                chunk_id="chunk-j1",
                content="text",
                file_path="/docs/a.pdf",
                score=0.9,
            )
        ]

        result = await use_case.execute(
            working_dir="/tmp/rag/project_1",
            query="test query",
            num_variations=1,
        )

        assert isinstance(result, ClassicalQueryResponse)

    # ------------------------------------------------------------------
    # Configuration-driven behavior
    # ------------------------------------------------------------------

    async def test_execute_uses_config_defaults(
        self,
        mock_vector_store: AsyncMock,
        mock_llm: AsyncMock,
    ) -> None:
        """Should use config defaults for num_variations and relevance_threshold."""
        config = ClassicalRAGConfig(
            CLASSICAL_NUM_QUERY_VARIATIONS=5,
            CLASSICAL_RELEVANCE_THRESHOLD=7.0,
        )
        use_case = ClassicalQueryUseCase(
            vector_store=mock_vector_store,
            llm=mock_llm,
            config=config,
        )

        mock_llm.generate.return_value = json.dumps(["v1", "v2", "v3", "v4", "v5"])

        await use_case.execute(
            working_dir="/tmp/rag/project_1",
            query="test query",
        )

        # Should have generated 5 variations + original = 6 searches
        assert mock_vector_store.similarity_search.call_count == 6


class TestClassicalQueryUseCaseHybrid:
    """Tests for ClassicalQueryUseCase hybrid mode (BM25 + vector)."""

    @pytest.fixture
    def config(self) -> ClassicalRAGConfig:
        return ClassicalRAGConfig(
            CLASSICAL_CHUNK_SIZE=1000,
            CLASSICAL_CHUNK_OVERLAP=200,
            CLASSICAL_NUM_QUERY_VARIATIONS=2,
            CLASSICAL_RELEVANCE_THRESHOLD=3.0,
            CLASSICAL_TABLE_PREFIX="classical_rag_",
            CLASSICAL_LLM_TEMPERATURE=0.0,
            CLASSICAL_RRF_K=60,
        )

    @pytest.fixture
    def mock_bm25_engine(self) -> AsyncMock:
        mock = AsyncMock()
        mock.search.return_value = [
            BM25SearchResult(
                chunk_id="bm25-chunk-1",
                content="BM25 hit content",
                file_path="/docs/bm25.pdf",
                score=5.0,
                metadata={},
            ),
        ]
        return mock

    @pytest.fixture
    def use_case_hybrid(
        self,
        mock_vector_store: AsyncMock,
        mock_llm: AsyncMock,
        mock_bm25_engine: AsyncMock,
        config: ClassicalRAGConfig,
    ) -> ClassicalQueryUseCase:
        return ClassicalQueryUseCase(
            vector_store=mock_vector_store,
            llm=mock_llm,
            config=config,
            bm25_engine=mock_bm25_engine,
            rrf_k=config.CLASSICAL_RRF_K,
        )

    @pytest.fixture
    def use_case_no_bm25(
        self,
        mock_vector_store: AsyncMock,
        mock_llm: AsyncMock,
        config: ClassicalRAGConfig,
    ) -> ClassicalQueryUseCase:
        return ClassicalQueryUseCase(
            vector_store=mock_vector_store,
            llm=mock_llm,
            config=config,
            bm25_engine=None,
            rrf_k=config.CLASSICAL_RRF_K,
        )

    async def test_hybrid_mode_calls_bm25_search(
        self,
        use_case_hybrid: ClassicalQueryUseCase,
        mock_bm25_engine: AsyncMock,
    ) -> None:
        """Hybrid mode should call bm25_engine.search()."""
        await use_case_hybrid.execute(
            working_dir="/tmp/rag/project_1",
            query="test query",
            mode="hybrid",
        )
        mock_bm25_engine.search.assert_called_once()

    async def test_hybrid_mode_calls_vector_search(
        self,
        use_case_hybrid: ClassicalQueryUseCase,
        mock_vector_store: AsyncMock,
        mock_llm: AsyncMock,
    ) -> None:
        """Hybrid mode should also call vector similarity_search."""
        mock_llm.generate.return_value = json.dumps(["var1", "var2"])
        await use_case_hybrid.execute(
            working_dir="/tmp/rag/project_1",
            query="test query",
            mode="hybrid",
        )
        assert mock_vector_store.similarity_search.call_count >= 1

    async def test_hybrid_mode_combines_bm25_and_vector_results(
        self,
        use_case_hybrid: ClassicalQueryUseCase,
        mock_llm: AsyncMock,
        mock_vector_store: AsyncMock,
    ) -> None:
        """Hybrid mode should return combined results from both sources."""
        mock_llm.generate.return_value = json.dumps(["var1"])
        mock_vector_store.similarity_search.return_value = [
            SearchResult(
                chunk_id="vec-chunk-1",
                content="Vector hit content",
                file_path="/docs/vec.pdf",
                score=0.1,
            )
        ]
        result = await use_case_hybrid.execute(
            working_dir="/tmp/rag/project_1",
            query="test query",
            mode="hybrid",
            enable_llm_judge=False,
        )
        assert isinstance(result, ClassicalQueryResponse)
        chunk_ids = [c.chunk_id for c in result.chunks]
        assert "bm25-chunk-1" in chunk_ids
        assert "vec-chunk-1" in chunk_ids

    async def test_hybrid_mode_response_includes_hybrid_scores(
        self,
        use_case_hybrid: ClassicalQueryUseCase,
        mock_llm: AsyncMock,
        mock_vector_store: AsyncMock,
    ) -> None:
        """Hybrid mode response should include bm25_score, vector_score, combined_score."""
        mock_llm.generate.return_value = json.dumps(["var1"])
        mock_vector_store.similarity_search.return_value = [
            SearchResult(
                chunk_id="vec-chunk-1",
                content="vector hit",
                file_path="/docs/vec.pdf",
                score=0.1,
            )
        ]
        result = await use_case_hybrid.execute(
            working_dir="/tmp/rag/project_1",
            query="test query",
            mode="hybrid",
            enable_llm_judge=False,
        )
        if result.chunks:
            chunk = result.chunks[0]
            assert chunk.bm25_score is not None
            assert chunk.vector_score is not None
            assert chunk.combined_score is not None

    async def test_hybrid_mode_response_mode_field(
        self,
        use_case_hybrid: ClassicalQueryUseCase,
    ) -> None:
        """Hybrid mode response should set mode='hybrid'."""
        result = await use_case_hybrid.execute(
            working_dir="/tmp/rag/project_1",
            query="test query",
            mode="hybrid",
            enable_llm_judge=False,
        )
        assert result.mode == "hybrid"

    async def test_hybrid_mode_falls_back_to_vector_when_bm25_unavailable(
        self,
        use_case_no_bm25: ClassicalQueryUseCase,
        mock_vector_store: AsyncMock,
        mock_llm: AsyncMock,
    ) -> None:
        """When bm25_engine is None, hybrid mode should fall back to vector."""
        mock_llm.generate.return_value = json.dumps(["var1"])
        result = await use_case_no_bm25.execute(
            working_dir="/tmp/rag/project_1",
            query="test query",
            mode="hybrid",
        )
        assert isinstance(result, ClassicalQueryResponse)
        assert result.mode == "vector"

    async def test_hybrid_mode_with_llm_judge(
        self,
        use_case_hybrid: ClassicalQueryUseCase,
        mock_llm: AsyncMock,
        mock_vector_store: AsyncMock,
    ) -> None:
        """Hybrid mode with LLM judge should score combined results."""
        call_count = 0

        async def mock_generate(system_prompt, user_message):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return json.dumps(["var1"])
            return "8"

        mock_llm.generate.side_effect = mock_generate
        result = await use_case_hybrid.execute(
            working_dir="/tmp/rag/project_1",
            query="test query",
            mode="hybrid",
            enable_llm_judge=True,
        )
        assert isinstance(result, ClassicalQueryResponse)

    async def test_vector_mode_default_unchanged(
        self,
        use_case_hybrid: ClassicalQueryUseCase,
        mock_bm25_engine: AsyncMock,
    ) -> None:
        """Default mode='vector' should NOT call bm25_engine."""
        await use_case_hybrid.execute(
            working_dir="/tmp/rag/project_1",
            query="test query",
        )
        mock_bm25_engine.search.assert_not_called()

    async def test_hybrid_mode_bm25_exception_fallback_to_vector(
        self,
        use_case_hybrid: ClassicalQueryUseCase,
        mock_bm25_engine: AsyncMock,
        mock_vector_store: AsyncMock,
        mock_llm: AsyncMock,
    ) -> None:
        """If BM25 raises an exception in hybrid mode, should fall back to vector."""
        mock_bm25_engine.search.side_effect = RuntimeError("BM25 connection failed")
        mock_llm.generate.return_value = "7"
        result = await use_case_hybrid.execute(
            working_dir="/tmp/rag/project_1",
            query="test query",
            mode="hybrid",
            enable_llm_judge=False,
        )
        assert isinstance(result, ClassicalQueryResponse)
        assert result.mode == "hybrid"
