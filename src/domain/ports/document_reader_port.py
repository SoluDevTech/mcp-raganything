from abc import ABC, abstractmethod

from pydantic import BaseModel


class DocumentMetadata(BaseModel):
    format_type: str = ""
    mime_type: str = ""


class TableData(BaseModel):
    markdown: str = ""


class DocumentContent(BaseModel):
    content: str
    metadata: DocumentMetadata
    tables: list[TableData] = []


class DocumentReaderPort(ABC):
    @abstractmethod
    async def extract_content(self, file_path: str) -> DocumentContent:
        pass
