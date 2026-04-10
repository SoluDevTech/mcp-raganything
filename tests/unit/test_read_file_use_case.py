import os
from unittest.mock import AsyncMock

import pytest

from application.use_cases.read_file_use_case import ReadFileUseCase
from domain.ports.document_reader_port import DocumentContent, DocumentMetadata


class TestReadFileUseCase:
    async def test_execute_downloads_file_from_storage(
        self,
        mock_storage: AsyncMock,
        mock_document_reader: AsyncMock,
        tmp_path,
    ) -> None:
        mock_storage.get_object.return_value = b"file content"
        mock_document_reader.extract_content.return_value = DocumentContent(
            content="extracted text",
            metadata=DocumentMetadata(format_type="pdf", mime_type="application/pdf"),
            tables=[],
        )
        use_case = ReadFileUseCase(
            storage=mock_storage,
            document_reader=mock_document_reader,
            bucket="my-bucket",
            output_dir=str(tmp_path),
        )

        await use_case.execute(file_path="docs/report.pdf")

        mock_storage.get_object.assert_called_once_with("my-bucket", "docs/report.pdf")

    async def test_execute_returns_document_content(
        self,
        mock_storage: AsyncMock,
        mock_document_reader: AsyncMock,
        tmp_path,
    ) -> None:
        mock_storage.get_object.return_value = b"file content"
        expected = DocumentContent(
            content="extracted text",
            metadata=DocumentMetadata(format_type="pdf", mime_type="application/pdf"),
            tables=[],
        )
        mock_document_reader.extract_content.return_value = expected
        use_case = ReadFileUseCase(
            storage=mock_storage,
            document_reader=mock_document_reader,
            bucket="my-bucket",
            output_dir=str(tmp_path),
        )

        result = await use_case.execute(file_path="docs/report.pdf")

        assert result.content == "extracted text"
        assert result.metadata.format_type == "pdf"

    async def test_execute_calls_document_reader_with_temp_file(
        self,
        mock_storage: AsyncMock,
        mock_document_reader: AsyncMock,
        tmp_path,
    ) -> None:
        mock_storage.get_object.return_value = b"pdf binary data"
        mock_document_reader.extract_content.return_value = DocumentContent(
            content="text",
            metadata=DocumentMetadata(format_type="pdf"),
            tables=[],
        )
        use_case = ReadFileUseCase(
            storage=mock_storage,
            document_reader=mock_document_reader,
            bucket="my-bucket",
            output_dir=str(tmp_path),
        )

        await use_case.execute(file_path="docs/report.pdf")

        call_args = mock_document_reader.extract_content.call_args
        tmp_file_path = call_args[0][0]
        assert tmp_file_path.endswith(".pdf")
        assert os.path.dirname(tmp_file_path) == str(tmp_path)

    async def test_execute_propagates_file_not_found(
        self,
        mock_storage: AsyncMock,
        mock_document_reader: AsyncMock,
        tmp_path,
    ) -> None:
        mock_storage.get_object.side_effect = FileNotFoundError("not found")
        use_case = ReadFileUseCase(
            storage=mock_storage,
            document_reader=mock_document_reader,
            bucket="my-bucket",
            output_dir=str(tmp_path),
        )

        with pytest.raises(FileNotFoundError):
            await use_case.execute(file_path="nonexistent.pdf")

    async def test_execute_cleans_up_temp_file(
        self,
        mock_storage: AsyncMock,
        mock_document_reader: AsyncMock,
        tmp_path,
    ) -> None:
        mock_storage.get_object.return_value = b"data"
        mock_document_reader.extract_content.return_value = DocumentContent(
            content="text",
            metadata=DocumentMetadata(format_type="txt"),
            tables=[],
        )
        use_case = ReadFileUseCase(
            storage=mock_storage,
            document_reader=mock_document_reader,
            bucket="my-bucket",
            output_dir=str(tmp_path),
        )

        await use_case.execute(file_path="report.pdf")

        call_args = mock_document_reader.extract_content.call_args
        tmp_file_path = call_args[0][0]
        assert not os.path.exists(tmp_file_path)

    async def test_execute_cleans_up_temp_file_on_error(
        self,
        mock_storage: AsyncMock,
        mock_document_reader: AsyncMock,
        tmp_path,
    ) -> None:
        mock_storage.get_object.return_value = b"data"
        mock_document_reader.extract_content.side_effect = ValueError("bad format")
        use_case = ReadFileUseCase(
            storage=mock_storage,
            document_reader=mock_document_reader,
            bucket="my-bucket",
            output_dir=str(tmp_path),
        )

        with pytest.raises(ValueError):
            await use_case.execute(file_path="report.pdf")

        call_args = mock_document_reader.extract_content.call_args
        tmp_file_path = call_args[0][0]
        assert not os.path.exists(tmp_file_path)

    async def test_execute_with_tables(
        self,
        mock_storage: AsyncMock,
        mock_document_reader: AsyncMock,
        tmp_path,
    ) -> None:
        from domain.ports.document_reader_port import TableData

        mock_storage.get_object.return_value = b"data"
        mock_document_reader.extract_content.return_value = DocumentContent(
            content="text with table",
            metadata=DocumentMetadata(format_type="pdf", mime_type="application/pdf"),
            tables=[TableData(markdown="| A | B |\n|---|---|\n| 1 | 2 |")],
        )
        use_case = ReadFileUseCase(
            storage=mock_storage,
            document_reader=mock_document_reader,
            bucket="my-bucket",
            output_dir=str(tmp_path),
        )

        result = await use_case.execute(file_path="report.pdf")

        assert len(result.tables) == 1
        assert result.tables[0].markdown == "| A | B |\n|---|---|\n| 1 | 2 |"
