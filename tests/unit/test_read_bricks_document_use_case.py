"""Tests for ReadBricksDocumentUseCase — follows exact pattern of test_read_file_use_case.py.

Downloads from Bricks API → saves temp file → calls Kreuzberg extract → returns DocumentContent.
Ensures temp file cleanup on both success and error paths.
"""

import os
from unittest.mock import AsyncMock

import pytest

from application.use_cases.read_bricks_document_use_case import (
    ReadBricksDocumentUseCase,
)
from domain.ports.document_reader_port import DocumentContent, DocumentMetadata


class TestReadBricksDocumentUseCase:
    """Tests for ReadBricksDocumentUseCase."""

    async def test_execute_downloads_from_bricks_api(
        self,
        mock_bricks_api: AsyncMock,
        mock_document_reader: AsyncMock,
        tmp_path,
    ) -> None:
        """Should call bricks_api.download_document with the file URL."""
        mock_bricks_api.download_document.return_value = (b"file content", "report.pdf")
        mock_document_reader.extract_content.return_value = DocumentContent(
            content="extracted text",
            metadata=DocumentMetadata(format_type="pdf", mime_type="application/pdf"),
            tables=[],
        )
        use_case = ReadBricksDocumentUseCase(
            bricks_api=mock_bricks_api,
            document_reader=mock_document_reader,
            output_dir=str(tmp_path),
        )

        await use_case.execute(file_url="https://s3.example.com/presigned/report.pdf")

        mock_bricks_api.download_document.assert_called_once_with(
            download_url="https://s3.example.com/presigned/report.pdf"
        )

    async def test_execute_returns_document_content(
        self,
        mock_bricks_api: AsyncMock,
        mock_document_reader: AsyncMock,
        tmp_path,
    ) -> None:
        """Should return DocumentContent from Kreuzberg extraction."""
        mock_bricks_api.download_document.return_value = (b"pdf data", "doc.pdf")
        expected = DocumentContent(
            content="extracted text from bricks document",
            metadata=DocumentMetadata(format_type="pdf", mime_type="application/pdf"),
            tables=[],
        )
        mock_document_reader.extract_content.return_value = expected
        use_case = ReadBricksDocumentUseCase(
            bricks_api=mock_bricks_api,
            document_reader=mock_document_reader,
            output_dir=str(tmp_path),
        )

        result = await use_case.execute(
            file_url="https://s3.example.com/presigned/doc.pdf"
        )

        assert result.content == "extracted text from bricks document"
        assert result.metadata.format_type == "pdf"
        assert result.metadata.mime_type == "application/pdf"

    async def test_execute_calls_document_reader_with_temp_file(
        self,
        mock_bricks_api: AsyncMock,
        mock_document_reader: AsyncMock,
        tmp_path,
    ) -> None:
        """Should save temp file and pass its path to document_reader.extract_content."""
        mock_bricks_api.download_document.return_value = (b"pdf binary", "report.pdf")
        mock_document_reader.extract_content.return_value = DocumentContent(
            content="text",
            metadata=DocumentMetadata(format_type="pdf"),
            tables=[],
        )
        use_case = ReadBricksDocumentUseCase(
            bricks_api=mock_bricks_api,
            document_reader=mock_document_reader,
            output_dir=str(tmp_path),
        )

        await use_case.execute(file_url="https://s3.example.com/report.pdf")

        call_args = mock_document_reader.extract_content.call_args
        tmp_file_path = call_args[0][0]
        assert tmp_file_path.endswith(".pdf")
        assert os.path.dirname(tmp_file_path) == str(tmp_path)

    async def test_execute_temp_file_suffix_matches_filename_extension(
        self,
        mock_bricks_api: AsyncMock,
        mock_document_reader: AsyncMock,
        tmp_path,
    ) -> None:
        """Should use the filename extension from the download as temp file suffix."""
        mock_bricks_api.download_document.return_value = (b"data", "spreadsheet.xlsx")
        mock_document_reader.extract_content.return_value = DocumentContent(
            content="spreadsheet data",
            metadata=DocumentMetadata(format_type="xlsx"),
            tables=[],
        )
        use_case = ReadBricksDocumentUseCase(
            bricks_api=mock_bricks_api,
            document_reader=mock_document_reader,
            output_dir=str(tmp_path),
        )

        await use_case.execute(file_url="https://s3.example.com/spreadsheet.xlsx")

        call_args = mock_document_reader.extract_content.call_args
        tmp_file_path = call_args[0][0]
        assert tmp_file_path.endswith(".xlsx")

    async def test_execute_propagates_download_failure(
        self,
        mock_bricks_api: AsyncMock,
        mock_document_reader: AsyncMock,
        tmp_path,
    ) -> None:
        """Should propagate errors when download fails."""
        mock_bricks_api.download_document.side_effect = ConnectionError(
            "S3 download failed"
        )
        use_case = ReadBricksDocumentUseCase(
            bricks_api=mock_bricks_api,
            document_reader=mock_document_reader,
            output_dir=str(tmp_path),
        )

        with pytest.raises(ConnectionError, match="S3 download failed"):
            await use_case.execute(file_url="https://s3.example.com/missing.pdf")

    async def test_execute_propagates_kreuzberg_failure(
        self,
        mock_bricks_api: AsyncMock,
        mock_document_reader: AsyncMock,
        tmp_path,
    ) -> None:
        """Should propagate errors when Kreuzberg extraction fails."""
        mock_bricks_api.download_document.return_value = (b"corrupt data", "broken.pdf")
        mock_document_reader.extract_content.side_effect = ValueError(
            "Unsupported format"
        )
        use_case = ReadBricksDocumentUseCase(
            bricks_api=mock_bricks_api,
            document_reader=mock_document_reader,
            output_dir=str(tmp_path),
        )

        with pytest.raises(ValueError, match="Unsupported format"):
            await use_case.execute(file_url="https://s3.example.com/broken.pdf")

    async def test_execute_cleans_up_temp_file(
        self,
        mock_bricks_api: AsyncMock,
        mock_document_reader: AsyncMock,
        tmp_path,
    ) -> None:
        """Should delete the temp file after successful extraction."""
        mock_bricks_api.download_document.return_value = (b"data", "report.pdf")
        mock_document_reader.extract_content.return_value = DocumentContent(
            content="text",
            metadata=DocumentMetadata(format_type="txt"),
            tables=[],
        )
        use_case = ReadBricksDocumentUseCase(
            bricks_api=mock_bricks_api,
            document_reader=mock_document_reader,
            output_dir=str(tmp_path),
        )

        await use_case.execute(file_url="https://s3.example.com/report.pdf")

        call_args = mock_document_reader.extract_content.call_args
        tmp_file_path = call_args[0][0]
        assert not os.path.exists(tmp_file_path)

    async def test_execute_cleans_up_temp_file_on_error(
        self,
        mock_bricks_api: AsyncMock,
        mock_document_reader: AsyncMock,
        tmp_path,
    ) -> None:
        """Should delete the temp file even when Kreuzberg extraction fails."""
        mock_bricks_api.download_document.return_value = (b"data", "report.pdf")
        mock_document_reader.extract_content.side_effect = ValueError("bad format")
        use_case = ReadBricksDocumentUseCase(
            bricks_api=mock_bricks_api,
            document_reader=mock_document_reader,
            output_dir=str(tmp_path),
        )

        with pytest.raises(ValueError):
            await use_case.execute(file_url="https://s3.example.com/report.pdf")

        call_args = mock_document_reader.extract_content.call_args
        tmp_file_path = call_args[0][0]
        assert not os.path.exists(tmp_file_path)

    async def test_execute_with_tables(
        self,
        mock_bricks_api: AsyncMock,
        mock_document_reader: AsyncMock,
        tmp_path,
    ) -> None:
        """Should include tables in the returned DocumentContent."""
        from domain.ports.document_reader_port import TableData

        mock_bricks_api.download_document.return_value = (b"data", "financials.pdf")
        mock_document_reader.extract_content.return_value = DocumentContent(
            content="text with table",
            metadata=DocumentMetadata(format_type="pdf", mime_type="application/pdf"),
            tables=[TableData(markdown="| A | B |\n|---|---|\n| 1 | 2 |")],
        )
        use_case = ReadBricksDocumentUseCase(
            bricks_api=mock_bricks_api,
            document_reader=mock_document_reader,
            output_dir=str(tmp_path),
        )

        result = await use_case.execute(
            file_url="https://s3.example.com/financials.pdf"
        )

        assert len(result.tables) == 1
        assert result.tables[0].markdown == "| A | B |\n|---|---|\n| 1 | 2 |"
