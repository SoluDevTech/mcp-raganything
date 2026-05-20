from abc import ABC, abstractmethod

from pydantic import BaseModel, ConfigDict, Field


class BricksDocumentInfo(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    id: str
    file_name: str = Field(alias="fileName")
    url: str
    hash: str = ""
    mime_type: str = Field(default="", alias="mimeType")
    size: int = 0
    status: str = ""
    uploaded_at: str | None = Field(default=None, alias="uploadedAt")
    image_analysis_status: str | None = Field(default=None, alias="imageAnalysisStatus")
    image_analysis_confidence: float | None = Field(
        default=None, alias="imageAnalysisConfidence"
    )
    image_analysis_reasoning: str | None = Field(
        default=None, alias="imageAnalysisReasoning"
    )
    image_analysis_date: str | None = Field(default=None, alias="imageAnalysisDate")
    category: str | None = None
    category_confidence: float | None = Field(default=None, alias="categoryConfidence")
    category_source: str | None = Field(default=None, alias="categorySource")
    category_classified_at: str | None = Field(
        default=None, alias="categoryClassifiedAt"
    )
    is_ignored: bool = Field(default=False, alias="isIgnored")
    project_id: str = Field(default="", alias="projectId")


class SectionVersionResult(BaseModel):
    success: bool
    message: str = ""
    data: dict | None = None
    dry_run: bool = False
    payload_preview: dict | None = None


class BricksApiPort(ABC):
    @abstractmethod
    async def list_project_documents(self, project_id: str) -> list[BricksDocumentInfo]:
        pass

    @abstractmethod
    async def download_document(
        self,
        document_id: str,
        project_id: str,
    ) -> tuple[bytes, str]:
        pass

    @abstractmethod
    async def publish_section_version(self, payload: dict) -> SectionVersionResult:
        pass
