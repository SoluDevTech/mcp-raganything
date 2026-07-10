"""Kreuzberg adapter for document extraction."""

import logging
import time

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
from domain.logging.messages import LogMessage
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
    from config import KreuzbergConfig

    cfg = KreuzbergConfig()
    if ocr_mode is None:
        ocr_mode = cfg.KREUZBERG_OCR_MODE
    llm_config = LLMConfig()
    if ocr_mode == "vlm":
        ocr = OcrConfig(
            backend="vlm",
            vlm_config=KreuzbergLlmConfig(
                model=llm_config.VISION_MODEL,
                api_key=llm_config.api_key,
                base_url=llm_config.api_base_url,
                timeout_secs=cfg.KREUZBERG_VLM_TIMEOUT,
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
        enable_quality_processing=cfg.KREUZBERG_ENABLE_QUALITY_PROCESSING,
        extraction_timeout_secs=cfg.KREUZBERG_EXTRACTION_TIMEOUT,
        pdf_options=PdfConfig(
            extract_images=cfg.KREUZBERG_EXTRACT_IMAGES,
            extract_metadata=True,
        ),
        ocr=ocr,
        chunking=ChunkingConfig(max_chars=chunk_size, max_overlap=chunk_overlap),
    )


class KreuzbergAdapter(DocumentReaderPort):
    """Kreuzberg-based document extraction adapter.

    Supports 91+ file formats via Kreuzberg, with configurable OCR backend
    (VLM or Tesseract), page-level extraction, table serialization, and
    chunking. Translates Kreuzberg exceptions into domain errors.
    """

    def __init__(self, ocr_mode: str | None = None) -> None:
        self._config = make_extraction_config(ocr_mode=ocr_mode)
        self._extraction_timeout = self._config.extraction_timeout_secs

    async def extract_content(
        self, file_path: str, mime_type: str = ""
    ) -> DocumentContent:
        """Extract textual content and tables from a document file.

        Args:
            file_path: Path to the local file to extract from.
            mime_type: Optional MIME type hint for Kreuzberg.

        Returns:
            DocumentContent with per-page content and serialized tables.

        Raises:
            DocumentReadError: On timeout or validation error.
            UnsupportedFormatError: If the file format cannot be parsed.
        """
        try:
            kwargs = self._build_extract_kwargs(mime_type)
            start_extract = time.time()
            result = await extract_file(file_path, **kwargs)
            extract_ms = (time.time() - start_extract) * 1000
            logger.info(LogMessage.KREUZBERG_EXTRACTION_DURATION, file_path, extract_ms)
        except ExtractionTimeoutError as e:
            self._raise_document_error(
                ErrorMessage.KREUZBERG_EXTRACTION_TIMED_OUT.format(
                    timeout=self._extraction_timeout
                ),
                e,
            )
        except ParsingError as e:
            self._raise_unsupported_format(e)
        except ValidationError as e:
            self._raise_document_error(ErrorMessage.INVALID_FILE.format(error=e), e)

        content = self._serialize_pages(result.pages or [], file_path)
        return DocumentContent(content=content)

    def _build_extract_kwargs(self, mime_type: str) -> dict:
        kwargs: dict = {"config": self._config}
        if mime_type:
            kwargs["mime_type"] = mime_type
        return kwargs

    def _raise_document_error(self, message: str, e: Exception) -> None:
        logger.exception(message)
        raise DocumentReadError(message) from e

    def _raise_unsupported_format(self, e: Exception) -> None:
        logger.exception(ErrorMessage.UNSUPPORTED_FILE_FORMAT.format(error=e))
        raise UnsupportedFormatError(
            ErrorMessage.UNSUPPORTED_FILE_FORMAT.format(error=e)
        ) from e

    @staticmethod
    def _coerce_page_dict(page: object) -> dict:
        if isinstance(page, dict):
            return page
        if hasattr(page, "__dict__"):
            return dict(page)
        return {}

    @staticmethod
    def _serialize_table(t: object) -> dict:
        markdown = getattr(t, "markdown", None)
        if not markdown and isinstance(t, dict):
            markdown = t.get("markdown")
        return {"markdown": str(markdown or "")}

    def _serialize_pages(self, pages: list, file_path: str) -> list[dict]:
        content: list[dict] = []
        start_serialize = time.time()

        for page in pages:
            page_dict = self._coerce_page_dict(page)
            tables_raw = page_dict.get("tables") or []
            tables_serializable = [self._serialize_table(t) for t in tables_raw]
            content.append(
                {
                    "page_number": int(page_dict.get("page_number", 1) or 1),
                    "content": str(page_dict.get("content", "") or ""),
                    "tables": tables_serializable,
                }
            )

        serialize_ms = (time.time() - start_serialize) * 1000
        logger.info(
            LogMessage.KREUZBERG_SERIALIZATION_DURATION, file_path, serialize_ms
        )
        return content
