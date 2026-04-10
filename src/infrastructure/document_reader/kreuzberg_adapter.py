from kreuzberg import ParsingError, ValidationError, extract_file

from domain.ports.document_reader_port import (
    DocumentContent,
    DocumentMetadata,
    DocumentReaderPort,
    TableData,
)


class KreuzbergAdapter(DocumentReaderPort):
    async def extract_content(self, file_path: str) -> DocumentContent:
        try:
            result = await extract_file(file_path)
        except ParsingError as e:
            raise ValueError(f"Unsupported file format: {e}") from e
        except ValidationError as e:
            raise ValueError(f"Invalid file: {e}") from e

        metadata = result.metadata if isinstance(result.metadata, dict) else {}
        return DocumentContent(
            content=result.content or "",
            metadata=DocumentMetadata(
                format_type=metadata.get("format_type", ""),
                mime_type=result.mime_type or "",
            ),
            tables=[
                TableData(markdown=getattr(t, "markdown", str(t)))
                for t in (result.tables or [])
            ],
        )
