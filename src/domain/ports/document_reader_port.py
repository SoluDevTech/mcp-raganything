from abc import ABC, abstractmethod

from pydantic import BaseModel


class DocumentMetadata(BaseModel):
    format_type: str = ""
    mime_type: str = ""


class TableData(BaseModel):
    markdown: str = ""


class ContentPages(BaseModel):
    page: int
    content: str


class DocumentContent(BaseModel):
    content: list
    metadata: DocumentMetadata | None = None
    tables: list[TableData] | None = None


class DocumentReaderPort(ABC):
    @abstractmethod
    async def extract_content(
        self, file_path: str, mime_type: str = ""
    ) -> DocumentContent:
        pass
