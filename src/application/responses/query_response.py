from pydantic import BaseModel, Field


class EntityResponse(BaseModel):
    entity_name: str
    entity_type: str
    description: str
    source_id: str
    file_path: str
    created_at: int


class RelationshipResponse(BaseModel):
    src_id: str
    tgt_id: str
    description: str
    keywords: str
    weight: float
    source_id: str
    file_path: str
    created_at: int


class ChunkResponse(BaseModel):
    reference_id: str
    content: str
    file_path: str
    chunk_id: str


class ReferenceResponse(BaseModel):
    reference_id: str
    file_path: str


class QueryDataResponse(BaseModel):
    entities: list[EntityResponse] = Field(default_factory=list)
    relationships: list[RelationshipResponse] = Field(default_factory=list)
    chunks: list[ChunkResponse] = Field(default_factory=list)
    references: list[ReferenceResponse] = Field(default_factory=list)


class KeywordsResponse(BaseModel):
    high_level: list[str] = Field(default_factory=list)
    low_level: list[str] = Field(default_factory=list)


class ProcessingInfoResponse(BaseModel):
    total_entities_found: int = 0
    total_relations_found: int = 0
    entities_after_truncation: int = 0
    relations_after_truncation: int = 0
    merged_chunks_count: int = 0
    final_chunks_count: int = 0


class QueryMetadataResponse(BaseModel):
    query_mode: str = ""
    keywords: KeywordsResponse | None = None
    processing_info: ProcessingInfoResponse | None = None


class QueryResponse(BaseModel):
    status: str
    message: str = ""
    data: QueryDataResponse = Field(default_factory=QueryDataResponse)
    metadata: QueryMetadataResponse | None = None
