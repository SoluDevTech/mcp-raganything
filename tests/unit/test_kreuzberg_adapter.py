from unittest.mock import AsyncMock, patch

import pytest

from domain.ports.document_reader_port import DocumentContent
from infrastructure.document_reader.kreuzberg_adapter import KreuzbergAdapter


class TestKreuzbergAdapter:
    @patch("infrastructure.document_reader.kreuzberg_adapter.extract_file")
    async def test_extract_content_returns_document_content(self, mock_extract) -> None:
        mock_result = AsyncMock()
        mock_result.pages = ["Hello world"]
        mock_result.mime_type = "application/pdf"
        mock_result.metadata = {"format_type": "pdf"}
        mock_result.tables = []
        mock_extract.return_value = mock_result

        adapter = KreuzbergAdapter(ocr_mode="vlm")
        result = await adapter.extract_content("/tmp/test.pdf")

        assert isinstance(result, DocumentContent)
        assert result.content == ["Hello world"]

    @patch("infrastructure.document_reader.kreuzberg_adapter.extract_file")
    async def test_extract_content_with_tables(self, mock_extract) -> None:
        mock_result = AsyncMock()
        mock_result.pages = ["text with table"]
        mock_result.mime_type = "application/pdf"
        mock_result.metadata = {}
        mock_table = AsyncMock()
        mock_table.markdown = "| A | B |\n|---|---|"
        mock_result.tables = [mock_table]
        mock_extract.return_value = mock_result

        adapter = KreuzbergAdapter(ocr_mode="vlm")
        result = await adapter.extract_content("/tmp/test.pdf")

        assert result.content == ["text with table"]

    @patch("infrastructure.document_reader.kreuzberg_adapter.extract_file")
    async def test_extract_content_uses_tesseract_mode(self, mock_extract) -> None:
        mock_result = AsyncMock()
        mock_result.pages = ["ocr text"]
        mock_result.mime_type = "application/pdf"
        mock_result.metadata = {}
        mock_result.tables = []
        mock_extract.return_value = mock_result

        adapter = KreuzbergAdapter(ocr_mode="tesseract")
        result = await adapter.extract_content("/tmp/test.pdf")

        assert result.content == ["ocr text"]
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
        mock_result.pages = ["text"]
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
        mock_result.pages = ["text"]
        mock_result.mime_type = "application/pdf"
        mock_result.metadata = {}
        mock_result.tables = []
        mock_extract.return_value = mock_result

        adapter = KreuzbergAdapter(ocr_mode="tesseract")
        await adapter.extract_content("/tmp/test.pdf")

        call_kwargs = mock_extract.call_args.kwargs
        assert "mime_type" not in call_kwargs
