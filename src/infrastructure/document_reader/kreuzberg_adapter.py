"""Kreuzberg adapter for document extraction."""

import logging

from kreuzberg import (
    ChunkingConfig,
    PageConfig,
    ExtractionConfig,
    OcrConfig,
    OutputFormat,
    ParsingError,
    PdfConfig,
    ValidationError,
    extract_file,
)
from kreuzberg import (
    LlmConfig as KreuzbergLlmConfig,
)

from config import LLMConfig
from domain.ports.document_reader_port import (
    ContentPages,
    DocumentContent,
    DocumentMetadata,
    DocumentReaderPort,
    TableData,
)

logger = logging.getLogger(__name__)


def make_extraction_config(
    ocr_mode: str | None = None,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> ExtractionConfig:
    from config import RAGConfig

    if ocr_mode is None:
        ocr_mode = RAGConfig().KREUZBERG_OCR_MODE
    llm_config = LLMConfig()
    if ocr_mode == "vlm":
        ocr = OcrConfig(
            backend="vlm",
            vlm_config=KreuzbergLlmConfig(
                model=llm_config.VISION_MODEL,
                api_key=llm_config.api_key,
                base_url=llm_config.api_base_url,
            ),
        )
    else:
        ocr = OcrConfig(backend="tesseract")
    return ExtractionConfig(
        pages=PageConfig(
            extract_pages=True,
            insert_page_markers=True,
            marker_format="\n\n<!-- PAGE {page_num} -->\n\n"
        ),
        use_cache=True,
        output_format=OutputFormat.MARKDOWN,
        enable_quality_processing=True,
        pdf_options=PdfConfig(extract_images=True, extract_metadata=True),
        ocr=ocr,
        chunking=ChunkingConfig(max_chars=chunk_size, max_overlap=chunk_overlap),
    )


class KreuzbergAdapter(DocumentReaderPort):
    def __init__(self, ocr_mode: str | None = None) -> None:
        self._config = make_extraction_config(ocr_mode=ocr_mode)

    async def extract_content(self, file_path: str, mime_type: str = "") -> DocumentContent:
        try:
            kwargs = {"config": self._config}
            if mime_type:
                kwargs["mime_type"] = mime_type
            result = await extract_file(file_path, **kwargs)
            logger.debug("Full extraction result for %s: %s", file_path, result)
        except ParsingError as e:
            raise ValueError(f"Unsupported file format: {e}") from e
        except ValidationError as e:
            raise ValueError(f"Invalid file: {e}") from e

        metadata = result.metadata if isinstance(result.metadata, dict) else {}

        return DocumentContent(
            content=result.pages or [],
        )
