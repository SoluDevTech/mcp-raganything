"""Kreuzberg parser for RAGAnything — extracts documents via Kreuzberg with VLM OCR."""

import importlib.util
import logging
from typing import Any

from kreuzberg import ExtractionResult, KreuzbergError, extract_file_sync
from raganything.parser import Parser

from infrastructure.document_reader.kreuzberg_adapter import make_extraction_config

logger = logging.getLogger(__name__)


class KreuzbergRAGAnythingParser(Parser):
    """RAGAnything custom parser using Kreuzberg for all document extraction.

    Uses Kreuzberg's sync API (extract_file_sync) which supports:
    - Native text extraction from PDFs and Office docs (no API calls)
    - VLM-based OCR for scanned PDFs and images (via OpenRouter)
    - Tables extraction with markdown output
    - 91+ file formats

    Note: parse_pdf, parse_image, and parse_document accept output_dir,
    method, and lang for API compatibility with the Parser base class,
    but Kreuzberg performs all extraction in-memory so they are ignored.
    """

    def __init__(self) -> None:
        super().__init__()
        self._config = make_extraction_config()

    def check_installation(self) -> bool:
        return importlib.util.find_spec("kreuzberg") is not None

    def parse_pdf(
        self,
        pdf_path,
        output_dir=None,
        method="auto",
        lang=None,
        **kwargs,
    ) -> list[dict[str, Any]]:
        return self._extract_and_convert(str(pdf_path))

    def parse_image(
        self,
        image_path,
        output_dir=None,
        lang=None,
        **kwargs,
    ) -> list[dict[str, Any]]:
        return self._extract_and_convert(str(image_path))

    def parse_document(
        self,
        file_path,
        method="auto",
        output_dir=None,
        lang=None,
        **kwargs,
    ) -> list[dict[str, Any]]:
        return self._extract_and_convert(str(file_path))

    def _extract_and_convert(self, file_path: str) -> list[dict[str, Any]]:
        try:
            result: ExtractionResult = extract_file_sync(file_path, config=self._config)
        except KreuzbergError as e:
            logger.error("Kreuzberg extraction failed for %s: %s", file_path, e)
            raise ValueError(f"Kreuzberg extraction failed: {e}") from e

        return self._convert_result(result)

    @staticmethod
    def _convert_result(result: ExtractionResult) -> list[dict[str, Any]]:
        content_list: list[dict[str, Any]] = []

        if result.content and result.content.strip():
            content_list.append(
                {
                    "type": "text",
                    "text": result.content,
                    "page_idx": 0,
                }
            )

        for table in result.tables or []:
            content_list.append(_convert_table(table))

        return content_list


def _convert_table(table) -> dict[str, Any]:
    page_idx = getattr(table, "page_number", None) or 0
    return {
        "type": "table",
        "table_body": table.markdown or "",
        "table_caption": [],
        "table_footnote": [],
        "page_idx": page_idx,
    }
