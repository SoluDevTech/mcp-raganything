"""Kreuzberg adapter for document extraction."""

import logging

from config import LLMConfig
from kreuzberg import (
    ExtractionConfig,
    LlmConfig,
    OcrConfig,
    OutputFormat,
    ParsingError,
    PdfConfig,
    ValidationError,
    extract_file,
)

from domain.ports.document_reader_port import (
    DocumentContent,
    DocumentMetadata,
    DocumentReaderPort,
    TableData,
)

logger = logging.getLogger(__name__)

_llm_config = LLMConfig()

_KREUZBERG_CONFIG = ExtractionConfig(
    use_cache=True,
    output_format=OutputFormat.MARKDOWN,
    enable_quality_processing=True,
    pdf_options=PdfConfig(extract_images=True, extract_metadata=True),
    ocr=OcrConfig(
        backend="vlm",
        vlm_config=LlmConfig(
            model=_llm_config.VISION_MODEL,
            api_key=_llm_config.api_key,
            base_url=_llm_config.api_base_url,
        ),
    ),
)


class KreuzbergAdapter(DocumentReaderPort):
    async def extract_content(self, file_path: str) -> DocumentContent:
        try:
            result = await extract_file(file_path, config=_KREUZBERG_CONFIG)
            logger.debug("Full extraction result for %s: %s", file_path, result)
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
