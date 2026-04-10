from pydantic import BaseModel, Field

from domain.ports.document_reader_port import DocumentMetadata, TableData


class FileInfoResponse(BaseModel):
    object_name: str
    size: int
    last_modified: str | None = None


class FileContentResponse(BaseModel):
    content: str
    metadata: DocumentMetadata
    tables: list[TableData] = Field(default_factory=list)
