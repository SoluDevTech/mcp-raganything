from typing import Literal

from pydantic import BaseModel, Field

QueryMode = Literal[
    "local", "global", "hybrid", "hybrid+", "naive", "mix", "bypass", "bm25"
]


class QueryRequest(BaseModel):
    working_dir: str = Field(
        ..., description="RAG workspace directory for this project"
    )
    query: str = Field(
        ...,
        description="The user's question or search query (e.g., 'What are the main findings?')",
    )
    mode: QueryMode = Field(
        default="naive",
        description=(
            "Search mode - 'naive' (default, vector only), 'local' (context-aware), "
            "'global' (document-level), 'hybrid' (local+global KG), "
            "'hybrid+' (BM25+vector parallel), 'mix' (automatic strategy), "
            "'bm25' (full-text only)."
        ),
    )
    top_k: int = Field(
        default=10,
        description=(
            "Number of chunks to retrieve (default 10, use 20 for broader search). "
            "Use 10 for fast, focused results; use 20 for comprehensive search."
        ),
    )


class MultimodalContentItem(BaseModel):
    type: Literal["image", "table", "equation"] = Field(
        ..., description="Type de contenu multimodal"
    )
    img_path: str | None = Field(
        default=None, description="Chemin vers un fichier image"
    )
    image_data: str | None = Field(
        default=None, description="Image encodée en base64 (alternative à img_path)"
    )
    table_data: str | None = Field(
        default=None, description="Données tabulaires au format CSV"
    )
    table_caption: str | None = Field(
        default=None, description="Légende décrivant la table"
    )
    latex: str | None = Field(default=None, description="Equation au format LaTeX")
    equation_caption: str | None = Field(
        default=None, description="Légende décrivant l'équation"
    )


class MultimodalQueryRequest(BaseModel):
    working_dir: str = Field(
        ..., description="RAG workspace directory for this project"
    )
    query: str = Field(..., description="The user's question or search query")
    mode: QueryMode = Field(
        default="hybrid",
        description="Search mode - 'hybrid' recommended for multimodal queries",
    )
    top_k: int = Field(default=10, description="Number of chunks to retrieve")
    multimodal_content: list[MultimodalContentItem] = Field(
        ..., description="Liste de contenus multimodaux à inclure dans la requête"
    )
