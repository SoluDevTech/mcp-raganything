import os
import tempfile
from pathlib import Path
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
        """Should convert .txt to PDF then index it successfully."""
        adapter = LightRAGAdapter(llm_config, rag_config_postgres)
        mock_rag = MagicMock()
        mock_rag.process_document_complete = AsyncMock()
        mock_rag._ensure_lightrag_initialized = AsyncMock()
        adapter.rag["test_dir"] = mock_rag

        # Create a temporary .txt file
        txt_file = tmp_path / "sample.txt"
        txt_file.write_text("This is sample text content for testing.")

        # Mock the PDF conversion
        fake_pdf_path = tmp_path / "sample_abc12345.pdf"
        with patch(
            "infrastructure.rag.lightrag_adapter.Parser"
        ) as mock_parser_cls:
            mock_parser_cls.convert_text_to_pdf.return_value = fake_pdf_path

            result = await adapter.index_document(
                file_path=str(txt_file),
                file_name="sample.txt",
                output_dir=str(tmp_path),
                working_dir="test_dir",
            )

            # Verify conversion was called with the txt file and a unique output_dir
            convert_call = mock_parser_cls.convert_text_to_pdf.call_args
            assert convert_call[0][0] == str(txt_file)
            assert convert_call[1]["output_dir"].startswith(str(tmp_path))
            # Verify process_document_complete received the PDF path
            mock_rag.process_document_complete.assert_awaited_once_with(
                file_path=str(fake_pdf_path),
                output_dir=str(tmp_path),
                parse_method="txt",
            )

        assert result.status == IndexingStatus.SUCCESS
        assert result.file_name == "sample.txt"
        assert result.file_path == str(txt_file)
        assert result.processing_time_ms is not None
        assert result.error is None

    async def test_index_text_extension_not_converted(
        self,
        llm_config: LLMConfig,
        rag_config_postgres: RAGConfig,
        tmp_path,
    ) -> None:
        """Should NOT convert .text files — not in _TEXT_EXTENSIONS, passes through."""
        adapter = LightRAGAdapter(llm_config, rag_config_postgres)
        mock_rag = MagicMock()
        mock_rag.process_document_complete = AsyncMock()
        mock_rag._ensure_lightrag_initialized = AsyncMock()
        adapter.rag["test_dir"] = mock_rag

        # Create a .text file (unsupported for conversion, will be passed as-is)
        text_file = tmp_path / "notes.text"
        text_file.write_text("Notes in .text extension format.")

        with patch(
            "infrastructure.rag.lightrag_adapter.Parser"
        ) as mock_parser_cls:
            result = await adapter.index_document(
                file_path=str(text_file),
                file_name="notes.text",
                output_dir=str(tmp_path),
                working_dir="test_dir",
            )

            # convert_text_to_pdf should NOT be called for .text extension
            mock_parser_cls.convert_text_to_pdf.assert_not_called()

        # The file passes through as-is (mock succeeds, real raganything would fail)
        assert result.status == IndexingStatus.SUCCESS
        assert result.file_name == "notes.text"
        # process_document_complete receives the original path, no conversion
        mock_rag.process_document_complete.assert_awaited_once_with(
            file_path=str(text_file),
            output_dir=str(tmp_path),
            parse_method="txt",
        )

    async def test_index_empty_txt_file(
        self,
        llm_config: LLMConfig,
        rag_config_postgres: RAGConfig,
        tmp_path,
    ) -> None:
        """Should handle empty .txt files correctly — converted to PDF then indexed."""
        adapter = LightRAGAdapter(llm_config, rag_config_postgres)
        mock_rag = MagicMock()
        mock_rag.process_document_complete = AsyncMock()
        mock_rag._ensure_lightrag_initialized = AsyncMock()
        adapter.rag["test_dir"] = mock_rag

        # Create an empty .txt file
        empty_file = tmp_path / "empty.txt"
        empty_file.write_text("")

        fake_pdf_path = tmp_path / "empty_abc12345.pdf"
        with patch(
            "infrastructure.rag.lightrag_adapter.Parser"
        ) as mock_parser_cls:
            mock_parser_cls.convert_text_to_pdf.return_value = fake_pdf_path

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
        """Should handle large text files efficiently — converted to PDF then indexed."""
        adapter = LightRAGAdapter(llm_config, rag_config_postgres)
        mock_rag = MagicMock()
        mock_rag.process_document_complete = AsyncMock()
        mock_rag._ensure_lightrag_initialized = AsyncMock()
        adapter.rag["test_dir"] = mock_rag

        # Create a large text file (1MB content)
        large_file = tmp_path / "large.txt"
        large_content = "Line of text.\n" * 50000  # ~500KB
        large_file.write_text(large_content)

        fake_pdf_path = tmp_path / "large_abc12345.pdf"
        with patch(
            "infrastructure.rag.lightrag_adapter.Parser"
        ) as mock_parser_cls:
            mock_parser_cls.convert_text_to_pdf.return_value = fake_pdf_path

            result = await adapter.index_document(
                file_path=str(large_file),
                file_name="large.txt",
                output_dir=str(tmp_path),
                working_dir="test_dir",
            )

        assert result.status == IndexingStatus.SUCCESS
        assert result.file_name == "large.txt"
        assert result.processing_time_ms is not None
        # Verify the PDF path was passed to process_document_complete
        call_args = mock_rag.process_document_complete.call_args
        assert call_args[1]["file_path"] == str(fake_pdf_path)

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

        fake_pdf_utf8 = tmp_path / "utf8_abc12345.pdf"
        with patch(
            "infrastructure.rag.lightrag_adapter.Parser"
        ) as mock_parser_cls:
            mock_parser_cls.convert_text_to_pdf.return_value = fake_pdf_utf8

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

        fake_pdf_utf16 = tmp_path / "utf16_abc12345.pdf"
        with patch(
            "infrastructure.rag.lightrag_adapter.Parser"
        ) as mock_parser_cls:
            mock_parser_cls.convert_text_to_pdf.return_value = fake_pdf_utf16

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

        fake_pdf_ascii = tmp_path / "ascii_abc12345.pdf"
        with patch(
            "infrastructure.rag.lightrag_adapter.Parser"
        ) as mock_parser_cls:
            mock_parser_cls.convert_text_to_pdf.return_value = fake_pdf_ascii

            result_ascii = await adapter.index_document(
                file_path=str(ascii_file),
                file_name="ascii.txt",
                output_dir=str(tmp_path),
                working_dir="test_dir",
            )
        assert result_ascii.status == IndexingStatus.SUCCESS

        # All three should be processed
        assert mock_rag.process_document_complete.call_count == 3

    # ------------------------------------------------------------------
    # Pre-convert .txt/.md to PDF — TDD tests (should FAIL until implemented)
    # ------------------------------------------------------------------

    async def test_index_txt_file_converts_to_pdf_before_processing(
        self,
        llm_config: LLMConfig,
        rag_config_postgres: RAGConfig,
        tmp_path,
    ) -> None:
        """Should convert .txt to PDF via Parser.convert_text_to_pdf before indexing."""
        adapter = LightRAGAdapter(llm_config, rag_config_postgres)
        mock_rag = MagicMock()
        mock_rag.process_document_complete = AsyncMock()
        mock_rag._ensure_lightrag_initialized = AsyncMock()
        adapter.rag["test_dir"] = mock_rag

        # Create a real .txt file in tmp_path
        txt_file = tmp_path / "fiche.txt"
        txt_file.write_text("Contenu de la fiche technique.")

        # Mock the PDF conversion to return a fake PDF path
        fake_pdf_path = tmp_path / "fiche_abc12345.pdf"
        with patch(
            "infrastructure.rag.lightrag_adapter.Parser"
        ) as mock_parser_cls:
            mock_parser_cls.convert_text_to_pdf.return_value = fake_pdf_path

            result = await adapter.index_document(
                file_path=str(txt_file),
                file_name="fiche.txt",
                output_dir=str(tmp_path),
                working_dir="test_dir",
            )

        # Verify convert_text_to_pdf was called with the original .txt path
        convert_call = mock_parser_cls.convert_text_to_pdf.call_args
        assert convert_call[0][0] == str(txt_file)
        assert convert_call[1]["output_dir"].startswith(str(tmp_path))
        # Verify process_document_complete received the PDF path, NOT the .txt path
        mock_rag.process_document_complete.assert_awaited_once_with(
            file_path=str(fake_pdf_path),
            output_dir=str(tmp_path),
            parse_method="txt",
        )
        # Verify result preserves original file_name and reports SUCCESS
        assert result.status == IndexingStatus.SUCCESS
        assert result.file_name == "fiche.txt"
        assert result.file_path == str(txt_file)

    async def test_index_txt_file_cleans_up_pdf_on_success(
        self,
        llm_config: LLMConfig,
        rag_config_postgres: RAGConfig,
        tmp_path,
    ) -> None:
        """Should delete the temporary PDF after successful indexing."""
        adapter = LightRAGAdapter(llm_config, rag_config_postgres)
        mock_rag = MagicMock()
        mock_rag.process_document_complete = AsyncMock()
        mock_rag._ensure_lightrag_initialized = AsyncMock()
        adapter.rag["test_dir"] = mock_rag

        txt_file = tmp_path / "cleanup.txt"
        txt_file.write_text("Content to be cleaned up.")

        fake_pdf_path = tmp_path / "cleanup_abc12345.pdf"
        with (
            patch(
                "infrastructure.rag.lightrag_adapter.Parser"
            ) as mock_parser_cls,
            patch("os.unlink") as mock_unlink,
        ):
            mock_parser_cls.convert_text_to_pdf.return_value = fake_pdf_path

            await adapter.index_document(
                file_path=str(txt_file),
                file_name="cleanup.txt",
                output_dir=str(tmp_path),
                working_dir="test_dir",
            )

            # Verify os.unlink was called on the temp PDF path
            mock_unlink.assert_called_once_with(str(fake_pdf_path))

    async def test_index_txt_file_cleans_up_pdf_on_failure(
        self,
        llm_config: LLMConfig,
        rag_config_postgres: RAGConfig,
        tmp_path,
    ) -> None:
        """Should delete the temporary PDF even when process_document_complete raises."""
        adapter = LightRAGAdapter(llm_config, rag_config_postgres)
        mock_rag = MagicMock()
        mock_rag.process_document_complete = AsyncMock(
            side_effect=RuntimeError("Parsing exploded")
        )
        mock_rag._ensure_lightrag_initialized = AsyncMock()
        adapter.rag["test_dir"] = mock_rag

        txt_file = tmp_path / "resilient.txt"
        txt_file.write_text("Content that will fail to parse.")

        fake_pdf_path = tmp_path / "resilient_abc12345.pdf"
        with (
            patch(
                "infrastructure.rag.lightrag_adapter.Parser"
            ) as mock_parser_cls,
            patch("os.unlink") as mock_unlink,
        ):
            mock_parser_cls.convert_text_to_pdf.return_value = fake_pdf_path

            result = await adapter.index_document(
                file_path=str(txt_file),
                file_name="resilient.txt",
                output_dir=str(tmp_path),
                working_dir="test_dir",
            )

        # The result should be FAILED, but cleanup must still have happened
        assert result.status == IndexingStatus.FAILED
        assert result.file_name == "resilient.txt"
        mock_unlink.assert_called_once_with(str(fake_pdf_path))

    async def test_index_md_file_converts_to_pdf(
        self,
        llm_config: LLMConfig,
        rag_config_postgres: RAGConfig,
        tmp_path,
    ) -> None:
        """Should convert .md files to PDF before indexing, same as .txt."""
        adapter = LightRAGAdapter(llm_config, rag_config_postgres)
        mock_rag = MagicMock()
        mock_rag.process_document_complete = AsyncMock()
        mock_rag._ensure_lightrag_initialized = AsyncMock()
        adapter.rag["test_dir"] = mock_rag

        md_file = tmp_path / "readme.md"
        md_file.write_text("# Hello\n\nThis is **markdown**.")

        fake_pdf_path = tmp_path / "readme_abc12345.pdf"
        with patch(
            "infrastructure.rag.lightrag_adapter.Parser"
        ) as mock_parser_cls:
            mock_parser_cls.convert_text_to_pdf.return_value = fake_pdf_path

            result = await adapter.index_document(
                file_path=str(md_file),
                file_name="readme.md",
                output_dir=str(tmp_path),
                working_dir="test_dir",
            )

        # Verify conversion was called and the PDF path was forwarded
        convert_call = mock_parser_cls.convert_text_to_pdf.call_args
        assert convert_call[0][0] == str(md_file)
        assert convert_call[1]["output_dir"].startswith(str(tmp_path))
        mock_rag.process_document_complete.assert_awaited_once_with(
            file_path=str(fake_pdf_path),
            output_dir=str(tmp_path),
            parse_method="txt",
        )
        assert result.status == IndexingStatus.SUCCESS
        assert result.file_name == "readme.md"

    async def test_index_pdf_file_not_converted(
        self,
        llm_config: LLMConfig,
        rag_config_postgres: RAGConfig,
        tmp_path,
    ) -> None:
        """Should NOT attempt conversion for .pdf files — pass through unchanged."""
        adapter = LightRAGAdapter(llm_config, rag_config_postgres)
        mock_rag = MagicMock()
        mock_rag.process_document_complete = AsyncMock()
        mock_rag._ensure_lightrag_initialized = AsyncMock()
        adapter.rag["test_dir"] = mock_rag

        pdf_file = tmp_path / "report.pdf"
        pdf_file.write_text("fake pdf content")

        with patch(
            "infrastructure.rag.lightrag_adapter.Parser"
        ) as mock_parser_cls:
            result = await adapter.index_document(
                file_path=str(pdf_file),
                file_name="report.pdf",
                output_dir=str(tmp_path),
                working_dir="test_dir",
            )

            # convert_text_to_pdf must NOT be called for .pdf files
            mock_parser_cls.convert_text_to_pdf.assert_not_called()

        # process_document_complete receives original path unchanged
        mock_rag.process_document_complete.assert_awaited_once_with(
            file_path=str(pdf_file),
            output_dir=str(tmp_path),
            parse_method="txt",
        )
        assert result.status == IndexingStatus.SUCCESS
        assert result.file_name == "report.pdf"

    async def test_index_txt_conversion_failure_returns_failed_result(
        self,
        llm_config: LLMConfig,
        rag_config_postgres: RAGConfig,
        tmp_path,
    ) -> None:
        """Should return FAILED when convert_text_to_pdf raises."""
        adapter = LightRAGAdapter(llm_config, rag_config_postgres)
        mock_rag = MagicMock()
        mock_rag.process_document_complete = AsyncMock()
        mock_rag._ensure_lightrag_initialized = AsyncMock()
        adapter.rag["test_dir"] = mock_rag

        txt_file = tmp_path / "broken.txt"
        txt_file.write_text("Content that cannot be decoded.")

        with patch(
            "infrastructure.rag.lightrag_adapter.Parser"
        ) as mock_parser_cls:
            mock_parser_cls.convert_text_to_pdf.side_effect = RuntimeError(
                "Could not decode text file broken.txt with any supported encoding"
            )

            result = await adapter.index_document(
                file_path=str(txt_file),
                file_name="broken.txt",
                output_dir=str(tmp_path),
                working_dir="test_dir",
            )

        # Should return FAILED with the error message
        assert result.status == IndexingStatus.FAILED
        assert result.file_name == "broken.txt"
        assert result.error is not None
        assert "Could not decode" in result.error
        # process_document_complete should NOT have been called
        mock_rag.process_document_complete.assert_not_awaited()

    async def test_index_folder_txt_file_converts_to_pdf(
        self,
        llm_config: LLMConfig,
        rag_config_postgres: RAGConfig,
        tmp_path,
    ) -> None:
        """Should convert .txt files to PDF when indexing via index_folder."""
        adapter = LightRAGAdapter(llm_config, rag_config_postgres)
        mock_rag = MagicMock()
        mock_rag.process_document_complete = AsyncMock()
        mock_rag._ensure_lightrag_initialized = AsyncMock()
        adapter.rag["test_dir"] = mock_rag

        # Create a folder with a .txt file
        folder = tmp_path / "docs_folder"
        folder.mkdir()
        txt_file = folder / "notes.txt"
        txt_file.write_text("Some text content.")

        fake_pdf_path = tmp_path / "notes_abc12345.pdf"
        with patch(
            "infrastructure.rag.lightrag_adapter.Parser"
        ) as mock_parser_cls:
            mock_parser_cls.convert_text_to_pdf.return_value = fake_pdf_path

            result = await adapter.index_folder(
                folder_path=str(folder),
                output_dir=str(tmp_path),
                working_dir="test_dir",
            )

        # Conversion should have happened
        convert_call = mock_parser_cls.convert_text_to_pdf.call_args
        assert convert_call[0][0] == str(txt_file)
        assert convert_call[1]["output_dir"].startswith(str(tmp_path))
        # process_document_complete should have received the PDF path
        mock_rag.process_document_complete.assert_awaited_once_with(
            file_path=str(fake_pdf_path),
            output_dir=str(tmp_path),
            parse_method="txt",
        )
        assert result.status == IndexingStatus.SUCCESS
        assert result.stats.files_processed == 1

    async def test_index_txt_file_unique_pdf_name(
        self,
        llm_config: LLMConfig,
        rag_config_postgres: RAGConfig,
        tmp_path,
    ) -> None:
        """Should use a unique PDF filename (UUID suffix) to avoid concurrent collisions."""
        adapter = LightRAGAdapter(llm_config, rag_config_postgres)
        mock_rag = MagicMock()
        mock_rag.process_document_complete = AsyncMock()
        mock_rag._ensure_lightrag_initialized = AsyncMock()
        adapter.rag["test_dir"] = mock_rag

        txt_file = tmp_path / "concurrent.txt"
        txt_file.write_text("Same file indexed concurrently.")

        # Capture what output_dir is passed to convert_text_to_pdf.
        # The adapter should use a tempdir or unique output_dir so the PDF
        # gets a unique name (e.g., {stem}_{uuid8}.pdf).
        conversion_calls: list[tuple] = []

        def fake_convert(text_path, output_dir=None):
            conversion_calls.append((text_path, output_dir))
            # Simulate what raganything would produce with a unique name
            stem = Path(text_path).stem
            return Path(output_dir) / f"{stem}_a1b2c3d4.pdf" if output_dir else None

        with patch(
            "infrastructure.rag.lightrag_adapter.Parser"
        ) as mock_parser_cls:
            mock_parser_cls.convert_text_to_pdf.side_effect = fake_convert

            result = await adapter.index_document(
                file_path=str(txt_file),
                file_name="concurrent.txt",
                output_dir=str(tmp_path),
                working_dir="test_dir",
            )

        assert result.status == IndexingStatus.SUCCESS

        # The key assertion: convert_text_to_pdf must have been called,
        # and the result PDF path must differ from a naive "{stem}.pdf".
        # The adapter is responsible for ensuring the output uses a unique name
        # so that concurrent index_document calls for the same .txt file
        # do not collide on the same PDF path.
        assert len(conversion_calls) == 1
        _text_path_arg, output_dir_arg = conversion_calls[0]

        # The PDF path produced by the conversion must contain a unique suffix
        # (not just "concurrent.pdf" which would collide under concurrency).
        pdf_path_used = mock_rag.process_document_complete.call_args[1]["file_path"]
        pdf_name = Path(pdf_path_used).name
        assert pdf_name.startswith("concurrent")
        assert pdf_name.endswith(".pdf")
        # The PDF name should include a UUID-like suffix, not be just "concurrent.pdf"
        assert pdf_name != "concurrent.pdf", (
            "PDF filename must include a unique suffix to avoid concurrent collisions, "
            f"got: {pdf_name}"
        )
