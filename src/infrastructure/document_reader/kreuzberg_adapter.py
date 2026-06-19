"""Kreuzberg adapter for document extraction."""

import logging

from kreuzberg import (
    ChunkingConfig,
    ExtractionConfig,
    ExtractionTimeoutError,
    OcrConfig,
    OutputFormat,
    PageConfig,
    ParsingError,
    PdfConfig,
    ValidationError,
    extract_file,
)
from kreuzberg import (
    LlmConfig as KreuzbergLlmConfig,
)

from config import LLMConfig
from domain.errors.document import DocumentReadError, UnsupportedFormatError
from domain.errors.messages import ErrorMessage
from domain.ports.document_reader_port import (
    DocumentContent,
    DocumentReaderPort,
)

logger = logging.getLogger(__name__)


def make_extraction_config(
    ocr_mode: str | None = None,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> ExtractionConfig:
    from config import RAGConfig

    rag = RAGConfig()
    if ocr_mode is None:
        ocr_mode = rag.KREUZBERG_OCR_MODE
    llm_config = LLMConfig()
    if ocr_mode == "vlm":
        ocr = OcrConfig(
            backend="vlm",
            vlm_config=KreuzbergLlmConfig(
                model=llm_config.VISION_MODEL,
                api_key=llm_config.api_key,
                base_url=llm_config.api_base_url,
                timeout_secs=rag.KREUZBERG_VLM_TIMEOUT,
            ),
        )
    else:
        ocr = OcrConfig(backend="tesseract")
    return ExtractionConfig(
        pages=PageConfig(
            extract_pages=True,
            insert_page_markers=True,
            marker_format="\n\n<!-- PAGE {page_num} -->\n\n",
        ),
        use_cache=True,
        output_format=OutputFormat.MARKDOWN,
        enable_quality_processing=rag.KREUZBERG_ENABLE_QUALITY_PROCESSING,
        extraction_timeout_secs=rag.KREUZBERG_EXTRACTION_TIMEOUT,
        pdf_options=PdfConfig(
            extract_images=rag.KREUZBERG_EXTRACT_IMAGES,
            extract_metadata=True,
        ),
        ocr=ocr,
        chunking=ChunkingConfig(max_chars=chunk_size, max_overlap=chunk_overlap),
    )


class KreuzbergAdapter(DocumentReaderPort):
    def __init__(self, ocr_mode: str | None = None) -> None:
        self._config = make_extraction_config(ocr_mode=ocr_mode)
        self._extraction_timeout = self._config.extraction_timeout_secs

    async def extract_content(
        self, file_path: str, mime_type: str = ""
    ) -> DocumentContent:
        try:
            kwargs = {"config": self._config}
            if mime_type:
                kwargs["mime_type"] = mime_type
            result = await extract_file(file_path, **kwargs)
        except ExtractionTimeoutError as e:
            logger.info(ErrorMessage.KREUZBERG_EXTRACTION_TIMED_OUT.format(
                timeout=self._extraction_timeout
            ))
            raise DocumentReadError(
                ErrorMessage.KREUZBERG_EXTRACTION_TIMED_OUT.format(
                    timeout=self._extraction_timeout
                )
            ) from e
        except ParsingError as e:
            logger.info(ErrorMessage.UNSUPPORTED_FILE_FORMAT.format(error=e))
            raise UnsupportedFormatError(
                ErrorMessage.UNSUPPORTED_FILE_FORMAT.format(error=e)
            ) from e
        except ValidationError as e:
            logger.info(ErrorMessage.INVALID_FILE.format(error=e))
            raise DocumentReadError(ErrorMessage.INVALID_FILE.format(error=e)) from e

        pages_raw = result.pages or []
        content: list[dict] = []

        for page in pages_raw:
            page_dict = (
                page
                if isinstance(page, dict)
                else dict(page)
                if hasattr(page, "__dict__")
                else {}
            )
            tables_raw = page_dict.get("tables") or []
            tables_serializable: list[dict] = []
            for t in tables_raw:
                markdown = (
                    getattr(t, "markdown", None)
                    or (t.get("markdown") if isinstance(t, dict) else "")
                    or ""
                )
                tables_serializable.append({"markdown": str(markdown)})
            content.append(
                {
                    "page_number": int(page_dict.get("page_number", 1) or 1),
                    "content": str(page_dict.get("content", "") or ""),
                    "tables": tables_serializable,
                }
            )

        return DocumentContent(
            content=content,
        )
