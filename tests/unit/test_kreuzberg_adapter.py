from unittest.mock import AsyncMock, patch

import pytest

from domain.ports.document_reader_port import DocumentContent
from infrastructure.document_reader.kreuzberg_adapter import KreuzbergAdapter


class TestKreuzbergAdapter:
    @patch("infrastructure.document_reader.kreuzberg_adapter.extract_file")
    async def test_extract_content_returns_document_content(self, mock_extract) -> None:
        mock_result = AsyncMock()
        mock_result.content = "Hello world"
        mock_result.mime_type = "application/pdf"
        mock_result.metadata = {"format_type": "pdf"}
        mock_result.tables = []
        mock_extract.return_value = mock_result

        adapter = KreuzbergAdapter()
        result = await adapter.extract_content("/tmp/test.pdf")

        assert isinstance(result, DocumentContent)
        assert result.content == "Hello world"
        assert result.metadata.mime_type == "application/pdf"

    @patch("infrastructure.document_reader.kreuzberg_adapter.extract_file")
    async def test_extract_content_with_tables(self, mock_extract) -> None:
        mock_result = AsyncMock()
        mock_result.content = "text with table"
        mock_result.mime_type = "application/pdf"
        mock_result.metadata = {}
        mock_table = AsyncMock()
        mock_table.markdown = "| A | B |\n|---|---|"
        mock_result.tables = [mock_table]
        mock_extract.return_value = mock_result

        adapter = KreuzbergAdapter()
        result = await adapter.extract_content("/tmp/test.pdf")

        assert len(result.tables) == 1
        assert result.tables[0].markdown == "| A | B |\n|---|---|"

    @patch("infrastructure.document_reader.kreuzberg_adapter.extract_file")
    async def test_extract_content_raises_value_error_for_parsing_error(
        self, mock_extract
    ) -> None:
        from kreuzberg import ParsingError

        mock_extract.side_effect = ParsingError("unsupported format")
        adapter = KreuzbergAdapter()

        with pytest.raises(ValueError, match="Unsupported file format"):
            await adapter.extract_content("/tmp/test.xyz")

    @patch("infrastructure.document_reader.kreuzberg_adapter.extract_file")
    async def test_extract_content_raises_value_error_for_validation_error(
        self, mock_extract
    ) -> None:
        from kreuzberg import ValidationError

        mock_extract.side_effect = ValidationError("invalid file")
        adapter = KreuzbergAdapter()

        with pytest.raises(ValueError, match="Invalid file"):
            await adapter.extract_content("/tmp/test.bad")

    @patch("infrastructure.document_reader.kreuzberg_adapter.extract_file")
    async def test_extract_content_raises_on_other_kreuzberg_error(
        self, mock_extract
    ) -> None:
        from kreuzberg import KreuzbergError

        mock_extract.side_effect = KreuzbergError("some other error")
        adapter = KreuzbergAdapter()

        with pytest.raises(KreuzbergError):
            await adapter.extract_content("/tmp/test.pdf")
