import logging

from fastmcp import FastMCP
from fastmcp.exceptions import ToolError

from application.responses.file_response import FileContentResponse
from dependencies import (
    get_list_bricks_documents_use_case,
    get_publish_section_version_use_case,
    get_read_bricks_document_use_case,
)

logger = logging.getLogger(__name__)

mcp_bricks = FastMCP("RAGAnythingBricks")


@mcp_bricks.tool()
async def list_bricks_documents(project_unique_id: str) -> list:
    """Liste les documents d'un projet Bricks.

    Args:
        project_unique_id: L'identifiant unique du projet Bricks

    Returns:
        Liste des documents du projet avec métadonnées
    """
    use_case = get_list_bricks_documents_use_case()
    try:
        return await use_case.execute(project_id=project_unique_id)
    except Exception:
        logger.exception(
            "Failed to list bricks documents for project %s", project_unique_id
        )
        raise ToolError("Failed to list bricks documents") from None


@mcp_bricks.tool()
async def read_bricks_document(file_url: str) -> FileContentResponse:
    """Télécharge un document Bricks et extrait son contenu textuel.

    Args:
        file_url: L'URL de téléchargement du document Bricks

    Returns:
        Contenu extrait avec métadonnées et tables détectées
    """
    use_case = get_read_bricks_document_use_case()
    try:
        result = await use_case.execute(file_url=file_url)
    except Exception:
        logger.exception("Failed to read bricks document: %s", file_url)
        raise ToolError("Failed to read bricks document") from None
    return FileContentResponse(
        content=result.content,
        metadata=result.metadata,
        tables=result.tables,
    )


@mcp_bricks.tool()
async def publish_section_version(
    project_unique_id: str,
    section_key: str,
    content: dict,
    workflow_id: str = "agent-haiku-files-v1",
    workflow_name: str = "haiku-files",
    workflow_metadata: dict | None = None,
) -> dict:
    """Publie la réponse structurée d'une section d'un projet Bricks.

    Args:
        project_unique_id: L'identifiant unique du projet
        section_key: La clé de la section à publier
        content: Le contenu structuré de la section
        workflow_id: L'identifiant du workflow
        workflow_name: Le nom du workflow
        workflow_metadata: Métadonnées additionnelles du workflow

    Returns:
        Résultat de la publication avec statut et aperçu du payload
    """
    use_case = get_publish_section_version_use_case()
    try:
        return await use_case.execute(
            project_unique_id=project_unique_id,
            section_key=section_key,
            content=content,
            workflow_id=workflow_id,
            workflow_name=workflow_name,
            workflow_metadata=workflow_metadata,
        )
    except Exception:
        logger.exception(
            "Failed to publish section version for project %s", project_unique_id
        )
        raise ToolError("Failed to publish section version") from None
