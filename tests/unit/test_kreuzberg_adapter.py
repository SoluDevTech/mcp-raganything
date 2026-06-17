from unittest.mock import AsyncMock, patch

import pytest

from domain.ports.document_reader_port import DocumentContent
from infrastructure.document_reader.kreuzberg_adapter import KreuzbergAdapter


def _make_page(page_number: int, content: str, tables: list | None = None) -> dict:
    return {
        "page_number": page_number,
        "content": content,
        "tables": tables or [],
        "images": [],
    }


class TestKreuzbergAdapter:
    @patch("infrastructure.document_reader.kreuzberg_adapter.extract_file")
    async def test_extract_content_returns_document_content(self, mock_extract) -> None:
        mock_result = AsyncMock()
        mock_result.pages = [_make_page(1, "Hello world")]
        mock_result.mime_type = "application/pdf"
        mock_result.metadata = {"format_type": "pdf"}
        mock_result.tables = []
        mock_extract.return_value = mock_result

        adapter = KreuzbergAdapter(ocr_mode="vlm")
        result = await adapter.extract_content("/tmp/test.pdf")

        assert isinstance(result, DocumentContent)
        assert result.content == [{"page_number": 1, "content": "Hello world", "tables": []}]

    @patch("infrastructure.document_reader.kreuzberg_adapter.extract_file")
    async def test_extract_content_with_tables(self, mock_extract) -> None:
        class FakeTable:
            def __init__(self, markdown: str) -> None:
                self.markdown = markdown

        mock_result = AsyncMock()
        mock_result.pages = [_make_page(1, "text with table", tables=[FakeTable("| A | B |\n|---|---|")])]
        mock_result.mime_type = "application/pdf"
        mock_result.metadata = {}
        mock_result.tables = [FakeTable("| A | B |\n|---|---|")]
        mock_extract.return_value = mock_result

        adapter = KreuzbergAdapter(ocr_mode="vlm")
        result = await adapter.extract_content("/tmp/test.pdf")

        assert result.content == [{
            "page_number": 1,
            "content": "text with table",
            "tables": [{"markdown": "| A | B |\n|---|---|"}],
        }]

    @patch("infrastructure.document_reader.kreuzberg_adapter.extract_file")
    async def test_extract_content_uses_tesseract_mode(self, mock_extract) -> None:
        mock_result = AsyncMock()
        mock_result.pages = [_make_page(1, "ocr text")]
        mock_result.mime_type = "application/pdf"
        mock_result.metadata = {}
        mock_result.tables = []
        mock_extract.return_value = mock_result

        adapter = KreuzbergAdapter(ocr_mode="tesseract")
        result = await adapter.extract_content("/tmp/test.pdf")

        assert result.content == [{"page_number": 1, "content": "ocr text", "tables": []}]
        mock_extract.assert_awaited_once()

    @patch("infrastructure.document_reader.kreuzberg_adapter.extract_file")
    async def test_extract_content_raises_value_error_for_parsing_error(
        self, mock_extract
    ) -> None:
        from kreuzberg import ParsingError

        mock_extract.side_effect = ParsingError("unsupported format")
        adapter = KreuzbergAdapter(ocr_mode="vlm")

        with pytest.raises(ValueError, match="Unsupported file format"):
            await adapter.extract_content("/tmp/test.xyz")

    @patch("infrastructure.document_reader.kreuzberg_adapter.extract_file")
    async def test_extract_content_raises_value_error_for_validation_error(
        self, mock_extract
    ) -> None:
        from kreuzberg import ValidationError

        mock_extract.side_effect = ValidationError("invalid file")
        adapter = KreuzbergAdapter(ocr_mode="vlm")

        with pytest.raises(ValueError, match="Invalid file"):
            await adapter.extract_content("/tmp/test.bad")

    @patch("infrastructure.document_reader.kreuzberg_adapter.extract_file")
    async def test_extract_content_raises_on_other_kreuzberg_error(
        self, mock_extract
    ) -> None:
        from kreuzberg import KreuzbergError

        mock_extract.side_effect = KreuzbergError("some other error")
        adapter = KreuzbergAdapter(ocr_mode="vlm")

        with pytest.raises(KreuzbergError):
            await adapter.extract_content("/tmp/test.pdf")

    @patch("infrastructure.document_reader.kreuzberg_adapter.extract_file")
    async def test_extract_content_passes_mime_type(self, mock_extract) -> None:
        mock_result = AsyncMock()
        mock_result.pages = [_make_page(1, "text")]
        mock_result.mime_type = "application/pdf"
        mock_result.metadata = {}
        mock_result.tables = []
        mock_extract.return_value = mock_result

        adapter = KreuzbergAdapter(ocr_mode="tesseract")
        await adapter.extract_content("/tmp/doc.bin", mime_type="application/pdf")

        mock_extract.assert_awaited_once()
        call_kwargs = mock_extract.call_args.kwargs
        assert call_kwargs.get("mime_type") == "application/pdf"

    @patch("infrastructure.document_reader.kreuzberg_adapter.extract_file")
    async def test_extract_content_without_mime_type(self, mock_extract) -> None:
        mock_result = AsyncMock()
        mock_result.pages = [_make_page(1, "text")]
        mock_result.mime_type = "application/pdf"
        mock_result.metadata = {}
        mock_result.tables = []
        mock_extract.return_value = mock_result

        adapter = KreuzbergAdapter(ocr_mode="tesseract")
        await adapter.extract_content("/tmp/test.pdf")

        call_kwargs = mock_extract.call_args.kwargs
        assert "mime_type" not in call_kwargs

    @patch("infrastructure.document_reader.kreuzberg_adapter.extract_file")
    async def test_extract_content_handles_none_pages(self, mock_extract) -> None:
        mock_result = AsyncMock()
        mock_result.pages = None
        mock_result.mime_type = "application/pdf"
        mock_result.metadata = {}
        mock_result.tables = []
        mock_extract.return_value = mock_result

        adapter = KreuzbergAdapter(ocr_mode="tesseract")
        result = await adapter.extract_content("/tmp/test.pdf")

        assert result.content == []

    @patch("infrastructure.document_reader.kreuzberg_adapter.extract_file")
    async def test_extract_content_handles_kreuzberg_table_object_in_pages(self, mock_extract) -> None:
        """Regression test: Kreuzberg ExtractedTable objects must be serialized to dicts.

        Bug: previously, result.pages[i]['tables'] contained ExtractedTable objects
        that Pydantic could not serialize, causing
        'outputSchema defined but no structured output returned' in the MCP server.
        """
        class FakeExtractedTable:
            def __init__(self, markdown: str) -> None:
                self.markdown = markdown
                self.cells = [["A", "B"], ["1", "2"]]
                self.page_number = 1

        mock_table = FakeExtractedTable("| A | B |\n|---|---|")
        mock_result = AsyncMock()
        mock_result.pages = [_make_page(1, "page text", tables=[mock_table])]
        mock_result.mime_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        mock_result.metadata = {}
        mock_result.tables = []
        mock_extract.return_value = mock_result

        adapter = KreuzbergAdapter(ocr_mode="tesseract")
        result = await adapter.extract_content("/tmp/test.xlsx")

        import pydantic_core
        serialized = pydantic_core.to_jsonable_python(result)
        assert serialized["content"][0]["tables"][0] == {"markdown": "| A | B |\n|---|---|"}
