from unittest.mock import MagicMock, patch

import pytest
from raganything.parser import _CUSTOM_PARSERS, register_parser

from config import RAGConfig
from infrastructure.rag.kreuzberg_raganything_parser import KreuzbergRAGAnythingParser
from infrastructure.rag.lightrag_adapter import _ensure_parser_registered


@pytest.fixture(autouse=True)
def _clean_custom_parsers():
    original = dict(_CUSTOM_PARSERS)
    yield
    _CUSTOM_PARSERS.clear()
    _CUSTOM_PARSERS.update(original)


class TestKreuzbergRAGAnythingParserRegistration:
    def test_register_kreuzberg_parser_succeeds(self):
        register_parser("kreuzberg", KreuzbergRAGAnythingParser)
        assert "kreuzberg" in _CUSTOM_PARSERS

    def test_cannot_override_builtin_parser(self):
        with pytest.raises(ValueError, match="Cannot override built-in parser"):
            register_parser("mineru", KreuzbergRAGAnythingParser)


class TestKreuzbergRAGAnythingParserFormat:
    def _make_parser(self):
        return KreuzbergRAGAnythingParser()

    def _mock_extraction_result(self, content="Hello world", tables=None, metadata=None):
        result = MagicMock()
        result.content = content
        result.metadata = metadata or {}
        result.mime_type = "application/pdf"
        result.tables = tables or []
        return result

    def _mock_table(self, markdown="| A | B |\n|---|---|", page_number=0):
        table = MagicMock()
        table.markdown = markdown
        table.page_number = page_number
        return table

    @patch("infrastructure.rag.kreuzberg_raganything_parser.extract_file_sync")
    @pytest.mark.parametrize(
        "method,path,expected_content",
        [
            ("parse_pdf", "/tmp/test.pdf", "Sample PDF text"),
            ("parse_image", "/tmp/photo.png", "Extracted image text"),
        ],
    )
    def test_parse_returns_content_list(self, mock_extract_sync, method, path, expected_content):
        mock_extract_sync.return_value = self._mock_extraction_result(content=expected_content)
        parser = self._make_parser()
        result = getattr(parser, method)(path, output_dir="/tmp/output")

        assert isinstance(result, list)
        assert len(result) >= 1
        assert result[0]["type"] == "text"
        assert result[0]["text"] == expected_content
        assert result[0]["page_idx"] == 0

    @patch("infrastructure.rag.kreuzberg_raganything_parser.extract_file_sync")
    def test_parse_pdf_with_tables(self, mock_extract_sync):
        table = self._mock_table(markdown="| A | B |\n|---|---|", page_number=1)
        mock_extract_sync.return_value = self._mock_extraction_result(
            content="Table below:", tables=[table]
        )
        parser = self._make_parser()
        result = parser.parse_pdf("/tmp/test.pdf", output_dir="/tmp/output")

        text_items = [r for r in result if r["type"] == "text"]
        table_items = [r for r in result if r["type"] == "table"]
        assert len(text_items) == 1
        assert len(table_items) == 1
        assert table_items[0]["table_body"] == "| A | B |\n|---|---|"
        assert table_items[0]["page_idx"] == 1

    @patch("infrastructure.rag.kreuzberg_raganything_parser.extract_file_sync")
    @pytest.mark.parametrize(
        "path,expected_content",
        [
            ("/tmp/doc.pdf", "PDF doc content"),
            ("/tmp/photo.png", "Image content"),
        ],
    )
    def test_parse_document_routes_files(self, mock_extract_sync, path, expected_content):
        mock_extract_sync.return_value = self._mock_extraction_result(content=expected_content)
        parser = self._make_parser()
        result = parser.parse_document(path, method="auto", output_dir="/tmp/output")

        assert isinstance(result, list)
        assert result[0]["type"] == "text"
        assert result[0]["text"] == expected_content

    @patch("infrastructure.rag.kreuzberg_raganything_parser.extract_file_sync")
    @pytest.mark.parametrize("content", ["", "   \n\n  "])
    def test_empty_or_whitespace_content_returns_empty_list(self, mock_extract_sync, content):
        mock_extract_sync.return_value = self._mock_extraction_result(content=content)
        parser = self._make_parser()
        result = parser.parse_pdf("/tmp/empty.pdf", output_dir="/tmp/output")

        assert result == []

    @patch("infrastructure.rag.kreuzberg_raganything_parser.extract_file_sync")
    def test_extraction_error_raises_value_error(self, mock_extract_sync):
        from kreuzberg import KreuzbergError

        mock_extract_sync.side_effect = KreuzbergError("Extraction crashed")
        parser = self._make_parser()

        with pytest.raises(ValueError, match="Kreuzberg extraction failed"):
            parser.parse_pdf("/tmp/bad.pdf", output_dir="/tmp/output")

    def test_check_installation_returns_true(self):
        parser = self._make_parser()
        assert parser.check_installation() is True

    @patch(
        "infrastructure.rag.kreuzberg_raganything_parser.importlib.util.find_spec",
        return_value=None,
    )
    def test_check_installation_returns_false_when_missing(self, mock_find_spec):
        parser = self._make_parser()
        assert parser.check_installation() is False

    @patch("infrastructure.rag.kreuzberg_raganything_parser.extract_file_sync")
    def test_table_with_no_page_number_defaults_to_zero(self, mock_extract_sync):
        table = MagicMock()
        table.markdown = "| X |"
        del table.page_number
        mock_extract_sync.return_value = self._mock_extraction_result(
            content="Text", tables=[table]
        )
        parser = self._make_parser()
        result = parser.parse_pdf("/tmp/test.pdf", output_dir="/tmp/output")

        table_items = [r for r in result if r["type"] == "table"]
        assert table_items[0]["page_idx"] == 0

    @patch("infrastructure.rag.kreuzberg_raganything_parser.extract_file_sync")
    def test_multiple_tables(self, mock_extract_sync):
        t1 = self._mock_table(markdown="| A |", page_number=0)
        t2 = self._mock_table(markdown="| B |", page_number=1)
        mock_extract_sync.return_value = self._mock_extraction_result(
            content="Text with two tables", tables=[t1, t2]
        )
        parser = self._make_parser()
        result = parser.parse_pdf("/tmp/test.pdf", output_dir="/tmp/output")

        table_items = [r for r in result if r["type"] == "table"]
        assert len(table_items) == 2
        assert table_items[0]["page_idx"] == 0
        assert table_items[1]["page_idx"] == 1


class TestEnsureParserRegistered:
    def test_unknown_parser_name_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown document parser"):
            _ensure_parser_registered("unknown_parser")

    @pytest.mark.parametrize("name", ["mineru", "docling", "paddleocr"])
    def test_builtin_parser_names_are_accepted(self, name):
        _ensure_parser_registered(name)


class TestRAGConfigDocumentParser:
    def test_default_document_parser_is_kreuzberg(self):
        config = RAGConfig()
        assert config.DOCUMENT_PARSER == "kreuzberg"

    def test_document_parser_can_be_overridden(self):
        config = RAGConfig(DOCUMENT_PARSER="docling")
        assert config.DOCUMENT_PARSER == "docling"
