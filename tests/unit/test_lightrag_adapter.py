import os
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from config import LLMConfig, RAGConfig
from domain.entities.indexing_result import IndexingStatus
from infrastructure.rag.lightrag_adapter import LightRAGAdapter


@pytest.fixture
def llm_config() -> LLMConfig:
    return LLMConfig(
        OPEN_ROUTER_API_KEY="test-key",
        CHAT_MODEL="test-model",
        EMBEDDING_MODEL="test-embed",
        EMBEDDING_DIM=128,
        MAX_TOKEN_SIZE=512,
        VISION_MODEL="test-vision",
    )


@pytest.fixture
def rag_config_postgres() -> RAGConfig:
    return RAGConfig(RAG_STORAGE_TYPE="postgres")


@pytest.fixture
def rag_config_local() -> RAGConfig:
    return RAGConfig(RAG_STORAGE_TYPE="local")


class TestLightRAGAdapter:
    """Tests for LightRAGAdapter — the external boundary (RAGAnything) is mocked."""

    def test_init_stores_configs(
        self, llm_config: LLMConfig, rag_config_postgres: RAGConfig
    ) -> None:
        """Should store configs and leave rag as empty dict before init_project."""
        adapter = LightRAGAdapter(llm_config, rag_config_postgres)

        assert adapter._llm_config is llm_config
        assert adapter._rag_config is rag_config_postgres
        assert adapter.rag == {}

    @patch("infrastructure.rag.lightrag_adapter.EmbeddingFunc")
    @patch("infrastructure.rag.lightrag_adapter.RAGAnything")
    def test_init_project_creates_rag_instance(
        self,
        mock_rag_cls: MagicMock,
        _mock_embedding_func: MagicMock,
        llm_config: LLMConfig,
        rag_config_postgres: RAGConfig,
    ) -> None:
        """Should instantiate RAGAnything with safe working_dir on init_project."""
        adapter = LightRAGAdapter(llm_config, rag_config_postgres)
        adapter.init_project("/tmp/test_project")

        expected_dir = os.path.join(
            tempfile.gettempdir(), "raganything", "tmp/test_project"
        )
        mock_rag_cls.assert_called_once()
        call_kwargs = mock_rag_cls.call_args[1]
        assert call_kwargs["config"].working_dir == expected_dir
        assert adapter.rag["/tmp/test_project"] is mock_rag_cls.return_value

    @patch("infrastructure.rag.lightrag_adapter.EmbeddingFunc")
    @patch("infrastructure.rag.lightrag_adapter.RAGAnything")
    def test_init_project_is_idempotent(
        self,
        mock_rag_cls: MagicMock,
        _mock_embedding_func: MagicMock,
        llm_config: LLMConfig,
        rag_config_postgres: RAGConfig,
    ) -> None:
        """Should return existing instance on second call, not create a new one."""
        adapter = LightRAGAdapter(llm_config, rag_config_postgres)
        first = adapter.init_project("/tmp/test_project")
        second = adapter.init_project("/tmp/test_project")

        assert first is second
        mock_rag_cls.assert_called_once()

    @patch("infrastructure.rag.lightrag_adapter.EmbeddingFunc")
    @patch("infrastructure.rag.lightrag_adapter.RAGAnything")
    def test_init_project_passes_postgres_storage_when_configured(
        self,
        mock_rag_cls: MagicMock,
        _mock_embedding_func: MagicMock,
        llm_config: LLMConfig,
        rag_config_postgres: RAGConfig,
    ) -> None:
        """Should include PG storage keys in lightrag_kwargs when RAG_STORAGE_TYPE is postgres."""
        adapter = LightRAGAdapter(llm_config, rag_config_postgres)
        adapter.init_project("/tmp/pg_project")

        call_kwargs = mock_rag_cls.call_args[1]
        lightrag_kwargs = call_kwargs["lightrag_kwargs"]
        assert lightrag_kwargs["kv_storage"] == "PGKVStorage"
        assert lightrag_kwargs["vector_storage"] == "PGVectorStorage"
        assert lightrag_kwargs["graph_storage"] == "PGGraphStorage"
        assert lightrag_kwargs["doc_status_storage"] == "PGDocStatusStorage"

    @patch("infrastructure.rag.lightrag_adapter.EmbeddingFunc")
    @patch("infrastructure.rag.lightrag_adapter.RAGAnything")
    def test_init_project_passes_local_storage_when_configured(
        self,
        mock_rag_cls: MagicMock,
        _mock_embedding_func: MagicMock,
        llm_config: LLMConfig,
        rag_config_local: RAGConfig,
    ) -> None:
        """Should include local storage keys in lightrag_kwargs when RAG_STORAGE_TYPE is not postgres."""
        adapter = LightRAGAdapter(llm_config, rag_config_local)
        adapter.init_project("/tmp/local_project")

        call_kwargs = mock_rag_cls.call_args[1]
        lightrag_kwargs = call_kwargs["lightrag_kwargs"]
        assert lightrag_kwargs["kv_storage"] == "JsonKVStorage"
        assert lightrag_kwargs["vector_storage"] == "NanoVectorDBStorage"
        assert lightrag_kwargs["graph_storage"] == "NetworkXStorage"
        assert lightrag_kwargs["doc_status_storage"] == "JsonDocStatusStorage"

    @patch("infrastructure.rag.lightrag_adapter.EmbeddingFunc")
    @patch("infrastructure.rag.lightrag_adapter.RAGAnything")
    def test_init_project_passes_cosine_threshold(
        self,
        mock_rag_cls: MagicMock,
        _mock_embedding_func: MagicMock,
        llm_config: LLMConfig,
        rag_config_postgres: RAGConfig,
    ) -> None:
        """Should forward cosine_threshold from RAGConfig into lightrag_kwargs."""
        adapter = LightRAGAdapter(llm_config, rag_config_postgres)
        adapter.init_project("/tmp/cosine_project")

        call_kwargs = mock_rag_cls.call_args[1]
        lightrag_kwargs = call_kwargs["lightrag_kwargs"]
        assert (
            lightrag_kwargs["cosine_threshold"] == rag_config_postgres.COSINE_THRESHOLD
        )

    async def test_index_document_success(
        self,
        llm_config: LLMConfig,
        rag_config_postgres: RAGConfig,
    ) -> None:
        """Should return SUCCESS result when process_document_complete succeeds."""
        adapter = LightRAGAdapter(llm_config, rag_config_postgres)
        mock_rag = MagicMock()
        mock_rag.process_document_complete = AsyncMock()
        mock_rag._ensure_lightrag_initialized = AsyncMock()
        adapter.rag["test_dir"] = mock_rag

        result = await adapter.index_document(
            file_path="/tmp/doc.pdf",
            file_name="doc.pdf",
            output_dir="/tmp/output",
            working_dir="test_dir",
        )

        assert result.status == IndexingStatus.SUCCESS
        assert result.file_name == "doc.pdf"
        assert result.file_path == "/tmp/doc.pdf"
        assert result.processing_time_ms is not None
        assert result.error is None
        mock_rag.process_document_complete.assert_awaited_once_with(
            file_path="/tmp/doc.pdf",
            output_dir="/tmp/output",
            parse_method="txt",
        )

    async def test_index_document_failure(
        self,
        llm_config: LLMConfig,
        rag_config_postgres: RAGConfig,
    ) -> None:
        """Should return FAILED result with error when process_document_complete raises."""
        adapter = LightRAGAdapter(llm_config, rag_config_postgres)
        mock_rag = MagicMock()
        mock_rag._ensure_lightrag_initialized = AsyncMock()
        mock_rag.process_document_complete = AsyncMock(
            side_effect=RuntimeError("Parsing exploded")
        )
        adapter.rag["test_dir"] = mock_rag

        result = await adapter.index_document(
            file_path="/tmp/bad.pdf",
            file_name="bad.pdf",
            output_dir="/tmp/output",
            working_dir="test_dir",
        )

        assert result.status == IndexingStatus.FAILED
        assert result.file_name == "bad.pdf"
        assert result.error == "Parsing exploded"
        assert result.processing_time_ms is not None

    async def test_index_folder_success(
        self,
        llm_config: LLMConfig,
        rag_config_postgres: RAGConfig,
        tmp_path,
    ) -> None:
        """Should return SUCCESS result when all files are processed successfully."""
        adapter = LightRAGAdapter(llm_config, rag_config_postgres)
        mock_rag = MagicMock()
        mock_rag._ensure_lightrag_initialized = AsyncMock()
        mock_rag.process_document_complete = AsyncMock()
        adapter.rag["test_dir"] = mock_rag

        # Create test files
        (tmp_path / "a.pdf").write_text("pdf1")
        (tmp_path / "b.pdf").write_text("pdf2")
        (tmp_path / "c.txt").write_text("skip")

        result = await adapter.index_folder(
            folder_path=str(tmp_path),
            output_dir="/tmp/output",
            recursive=True,
            file_extensions=[".pdf"],
            working_dir="test_dir",
        )

        assert result.status == IndexingStatus.SUCCESS
        assert result.stats.total_files == 2
        assert result.stats.files_processed == 2
        assert result.stats.files_failed == 0
        assert result.folder_path == str(tmp_path)
        assert result.recursive is True
        assert result.processing_time_ms is not None

    async def test_index_folder_raises_when_not_initialized(
        self,
        llm_config: LLMConfig,
        rag_config_postgres: RAGConfig,
    ) -> None:
        """Should raise KeyError when calling index_folder without init_project."""
        adapter = LightRAGAdapter(llm_config, rag_config_postgres)

        with pytest.raises(RuntimeError):
            await adapter.index_folder(
                folder_path="/tmp/docs",
                output_dir="/tmp/output",
                working_dir="missing_dir",
            )

    async def test_index_document_raises_when_not_initialized(
        self,
        llm_config: LLMConfig,
        rag_config_postgres: RAGConfig,
    ) -> None:
        """Should raise KeyError when calling index_document without init_project."""
        adapter = LightRAGAdapter(llm_config, rag_config_postgres)

        with pytest.raises(RuntimeError):
            await adapter.index_document(
                file_path="/tmp/doc.pdf",
                file_name="doc.pdf",
                output_dir="/tmp/output",
                working_dir="missing_dir",
            )

    async def test_query_success(
        self,
        llm_config: LLMConfig,
        rag_config_postgres: RAGConfig,
    ) -> None:
        """Should return query result from lightrag.aquery_data."""
        adapter = LightRAGAdapter(llm_config, rag_config_postgres)
        mock_rag = MagicMock()
        mock_rag._ensure_lightrag_initialized = AsyncMock()
        mock_lightrag = MagicMock()
        mock_lightrag.aquery_data = AsyncMock(
            return_value={"status": "success", "data": {"answer": "42"}}
        )
        mock_rag.lightrag = mock_lightrag
        adapter.rag["test_dir"] = mock_rag

        result = await adapter.query(
            query="What is the answer?", mode="naive", top_k=5, working_dir="test_dir"
        )

        assert result["status"] == "success"
        assert result["data"]["answer"] == "42"
        mock_lightrag.aquery_data.assert_awaited_once()

    async def test_query_returns_failure_when_lightrag_none(
        self,
        llm_config: LLMConfig,
        rag_config_postgres: RAGConfig,
    ) -> None:
        """Should return failure dict when rag.lightrag is None."""
        adapter = LightRAGAdapter(llm_config, rag_config_postgres)
        mock_rag = MagicMock()
        mock_rag._ensure_lightrag_initialized = AsyncMock()
        mock_rag.lightrag = None
        adapter.rag["test_dir"] = mock_rag

        result = await adapter.query(query="anything", working_dir="test_dir")

        assert result["status"] == "failure"
        assert result["message"] == "RAG engine not initialized"

    async def test_query_raises_when_not_initialized(
        self,
        llm_config: LLMConfig,
        rag_config_postgres: RAGConfig,
    ) -> None:
        """Should raise KeyError when calling query without init_project."""
        adapter = LightRAGAdapter(llm_config, rag_config_postgres)

        with pytest.raises(RuntimeError):
            await adapter.query(query="anything", working_dir="missing_dir")

    async def test_index_folder_failure(
        self,
        llm_config: LLMConfig,
        rag_config_postgres: RAGConfig,
        tmp_path,
    ) -> None:
        """Should return FAILED result when all files fail to process."""
        adapter = LightRAGAdapter(llm_config, rag_config_postgres)
        mock_rag = MagicMock()
        mock_rag._ensure_lightrag_initialized = AsyncMock()
        mock_rag.process_document_complete = AsyncMock(
            side_effect=OSError("Processing failed")
        )
        adapter.rag["test_dir"] = mock_rag

        (tmp_path / "a.pdf").write_text("pdf")

        result = await adapter.index_folder(
            folder_path=str(tmp_path),
            output_dir="/tmp/output",
            working_dir="test_dir",
        )

        assert result.status == IndexingStatus.FAILED
        assert result.folder_path == str(tmp_path)
        assert result.processing_time_ms is not None

    async def test_index_folder_partial_result(
        self,
        llm_config: LLMConfig,
        rag_config_postgres: RAGConfig,
        tmp_path,
    ) -> None:
        """Should return PARTIAL status when some files succeed and some fail."""
        adapter = LightRAGAdapter(llm_config, rag_config_postgres)
        mock_rag = MagicMock()
        mock_rag._ensure_lightrag_initialized = AsyncMock()

        call_count = 0

        async def side_effect(**_kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                return None
            raise OSError("fail")

        mock_rag.process_document_complete = AsyncMock(side_effect=side_effect)
        adapter.rag["test_dir"] = mock_rag

        (tmp_path / "a.pdf").write_text("ok1")
        (tmp_path / "b.pdf").write_text("ok2")
        (tmp_path / "c.pdf").write_text("fail")

        result = await adapter.index_folder(
            folder_path=str(tmp_path),
            output_dir="/tmp/output",
            working_dir="test_dir",
        )

        assert result.status == IndexingStatus.PARTIAL
        assert result.stats.files_processed == 2
        assert result.stats.files_failed == 1

    # ------------------------------------------------------------------
    # TXT File Support Tests
    # ------------------------------------------------------------------

    async def test_index_txt_file_success(
        self,
        llm_config: LLMConfig,
        rag_config_postgres: RAGConfig,
        tmp_path,
    ) -> None:
        """Should successfully index .txt file with parse_method='txt'."""
        adapter = LightRAGAdapter(llm_config, rag_config_postgres)
        mock_rag = MagicMock()
        mock_rag.process_document_complete = AsyncMock()
        mock_rag._ensure_lightrag_initialized = AsyncMock()
        adapter.rag["test_dir"] = mock_rag

        # Create a temporary .txt file
        txt_file = tmp_path / "sample.txt"
        txt_file.write_text("This is sample text content for testing.")

        result = await adapter.index_document(
            file_path=str(txt_file),
            file_name="sample.txt",
            output_dir=str(tmp_path),
            working_dir="test_dir",
        )

        assert result.status == IndexingStatus.SUCCESS
        assert result.file_name == "sample.txt"
        assert result.file_path == str(txt_file)
        assert result.processing_time_ms is not None
        assert result.error is None
        mock_rag.process_document_complete.assert_awaited_once_with(
            file_path=str(txt_file),
            output_dir=str(tmp_path),
            parse_method="txt",
        )

    async def test_index_text_extension_success(
        self,
        llm_config: LLMConfig,
        rag_config_postgres: RAGConfig,
        tmp_path,
    ) -> None:
        """Should successfully index .text extension files."""
        adapter = LightRAGAdapter(llm_config, rag_config_postgres)
        mock_rag = MagicMock()
        mock_rag.process_document_complete = AsyncMock()
        mock_rag._ensure_lightrag_initialized = AsyncMock()
        adapter.rag["test_dir"] = mock_rag

        # Create a .text file (another TXT format)
        text_file = tmp_path / "notes.text"
        text_file.write_text("Notes in .text extension format.")

        result = await adapter.index_document(
            file_path=str(text_file),
            file_name="notes.text",
            output_dir=str(tmp_path),
            working_dir="test_dir",
        )

        assert result.status == IndexingStatus.SUCCESS
        assert result.file_name == "notes.text"
        assert result.processing_time_ms is not None
        mock_rag.process_document_complete.assert_awaited_once()

    async def test_index_empty_txt_file(
        self,
        llm_config: LLMConfig,
        rag_config_postgres: RAGConfig,
        tmp_path,
    ) -> None:
        """Should handle empty .txt files correctly."""
        adapter = LightRAGAdapter(llm_config, rag_config_postgres)
        mock_rag = MagicMock()
        mock_rag.process_document_complete = AsyncMock()
        mock_rag._ensure_lightrag_initialized = AsyncMock()
        adapter.rag["test_dir"] = mock_rag

        # Create an empty .txt file
        empty_file = tmp_path / "empty.txt"
        empty_file.write_text("")

        result = await adapter.index_document(
            file_path=str(empty_file),
            file_name="empty.txt",
            output_dir=str(tmp_path),
            working_dir="test_dir",
        )

        assert result.status == IndexingStatus.SUCCESS
        assert result.file_name == "empty.txt"
        assert result.processing_time_ms is not None
        # Verify it was called even with empty file
        mock_rag.process_document_complete.assert_awaited_once()

    async def test_index_large_txt_file(
        self,
        llm_config: LLMConfig,
        rag_config_postgres: RAGConfig,
        tmp_path,
    ) -> None:
        """Should handle large text files efficiently."""
        adapter = LightRAGAdapter(llm_config, rag_config_postgres)
        mock_rag = MagicMock()
        mock_rag.process_document_complete = AsyncMock()
        mock_rag._ensure_lightrag_initialized = AsyncMock()
        adapter.rag["test_dir"] = mock_rag

        # Create a large text file (1MB content)
        large_file = tmp_path / "large.txt"
        large_content = "Line of text.\n" * 50000  # ~500KB
        large_file.write_text(large_content)

        result = await adapter.index_document(
            file_path=str(large_file),
            file_name="large.txt",
            output_dir=str(tmp_path),
            working_dir="test_dir",
        )

        assert result.status == IndexingStatus.SUCCESS
        assert result.file_name == "large.txt"
        assert result.processing_time_ms is not None
        # Verify the file path was passed correctly
        call_args = mock_rag.process_document_complete.call_args
        assert call_args[1]["file_path"] == str(large_file)

    async def test_index_txt_with_various_encodings(
        self,
        llm_config: LLMConfig,
        rag_config_postgres: RAGConfig,
        tmp_path,
    ) -> None:
        """Should handle txt files with different encodings (UTF-8, UTF-16, ASCII)."""
        adapter = LightRAGAdapter(llm_config, rag_config_postgres)
        mock_rag = MagicMock()
        mock_rag.process_document_complete = AsyncMock()
        mock_rag._ensure_lightrag_initialized = AsyncMock()
        adapter.rag["test_dir"] = mock_rag

        # Test UTF-8
        utf8_file = tmp_path / "utf8.txt"
        utf8_file.write_text("ASCII and UTF-8: café ñ 北京", encoding="utf-8")

        result_utf8 = await adapter.index_document(
            file_path=str(utf8_file),
            file_name="utf8.txt",
            output_dir=str(tmp_path),
            working_dir="test_dir",
        )
        assert result_utf8.status == IndexingStatus.SUCCESS

        # Test UTF-16
        utf16_file = tmp_path / "utf16.txt"
        utf16_file.write_text("UTF-16 content: 你好", encoding="utf-16")

        result_utf16 = await adapter.index_document(
            file_path=str(utf16_file),
            file_name="utf16.txt",
            output_dir=str(tmp_path),
            working_dir="test_dir",
        )
        assert result_utf16.status == IndexingStatus.SUCCESS

        # Test ASCII
        ascii_file = tmp_path / "ascii.txt"
        ascii_file.write_text("Simple ASCII content only", encoding="ascii")

        result_ascii = await adapter.index_document(
            file_path=str(ascii_file),
            file_name="ascii.txt",
            output_dir=str(tmp_path),
            working_dir="test_dir",
        )
        assert result_ascii.status == IndexingStatus.SUCCESS

        # All three should be processed
        assert mock_rag.process_document_complete.call_count == 3
