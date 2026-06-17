import logging
from datetime import UTC, datetime

from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from pydantic import BaseModel, Field

from application.responses.file_response import FileContentResponse
from dependencies import (
    get_list_bricks_documents_use_case,
    get_publish_section_version_use_case,
    get_read_bricks_document_use_case,
)

mcp_bricks = FastMCP("RAGAnythingBricks")

logger = logging.getLogger(__name__)

class DocumentSummary(BaseModel):
    id: str = Field(description="Document ID — pass this to read_bricks_document")
    file_name: str = Field(description="Nom du fichier")
    mime_type: str = Field(default="", description="Type MIME du fichier")
    size: int = Field(default=0, description="Taille en octets")
    status: str = Field(default="", description="Statut du document")


@mcp_bricks.tool()
async def list_bricks_documents(project_unique_id: str) -> list[DocumentSummary]:
    """Liste les documents d'un projet Bricks.

    IMPORTANT: This tool ONLY accepts project_unique_id.
    Do NOT pass any other parameters (e.g., limit, offset, etc.) — they are not supported.

    Args:
        project_unique_id: L'identifiant unique du projet Bricks

    Returns:
        Liste des documents du projet avec métadonnées
    """
    use_case = get_list_bricks_documents_use_case()
    try:
        documents = await use_case.execute(project_id=project_unique_id)
        logger.info(f"Documents found : {documents}")
    except PermissionError as e:
        logger.error(
            "List documents auth failed for project %s: %s", project_unique_id, e
        )
        raise ToolError(str(e)) from e
    except (ConnectionError, TimeoutError) as e:
        logger.error(
            "List documents network error for project %s: %s", project_unique_id, e
        )
        raise ToolError(str(e)) from e
    except Exception as e:
        logger.exception(
            "Failed to list bricks documents for project %s: %s", project_unique_id, e
        )
        raise ToolError(f"Failed to list bricks documents: {e}") from e
    return [
        DocumentSummary(
            id=doc.id,
            file_name=doc.file_name,
            mime_type=doc.mime_type,
            size=doc.size,
            status=doc.status,
        )
        for doc in documents
    ]


@mcp_bricks.tool()
async def read_bricks_document(
    document_id: str,
    project_unique_id: str,
) -> FileContentResponse:
    """Télécharge un document Bricks et extrait son contenu textuel.

    Résoud automatiquement l'URL pré-signée à partir du document_id et project_unique_id.

    IMPORTANT: This tool ONLY accepts document_id and project_unique_id.
    Do NOT pass any other parameters (e.g., limit, offset, etc.) — they are not supported.

    Args:
        document_id: L'identifiant du document (champ 'id' de list_bricks_documents)
        project_unique_id: L'identifiant du projet Bricks

    Returns:
        Contenu extrait avec métadonnées et tables détectées
    """
    use_case = get_read_bricks_document_use_case()
    try:
        logger.info(f"Reading document : {document_id}")
        result = await use_case.execute(
            document_id=document_id, project_id=project_unique_id
        )
        logger.info(f"Result for read document use case : {result}")
    except PermissionError as e:
        logger.error("Read document auth failed for %s: %s", document_id, e)
        raise ToolError(str(e)) from e
    except (ConnectionError, TimeoutError) as e:
        logger.error("Read document network error for %s: %s", document_id, e)
        raise ToolError(str(e)) from e
    except Exception as e:
        logger.exception(
            "Failed to read bricks document: %s in project %s: %s",
            document_id,
            project_unique_id,
            e,
        )
        raise ToolError(f"Failed to read bricks document: {e}") from e
    return FileContentResponse(
        content=result.content,
        metadata=result.metadata,
        tables=result.tables,
    )


@mcp_bricks.tool()
async def publish_section_version(
    project_unique_id: str, content: dict, field_sources: list[dict]
) -> dict:
    """Publie la réponse structurée d'une section d'un projet Bricks.

    IMPORTANT: This tool ONLY accepts project_unique_id, content and field_sources.
    Do NOT pass any other parameters (e.g., limit, offset, etc.) — they are not supported.

    Args:
        project_unique_id: L'identifiant unique du projet
        content: Le contenu structuré de la section à publier, typiquement un dict avec les données extraites et analysées
        field_sources: Sources justifiant chaque champ du contenu
    Returns:
        Résultat de la publication avec statut et aperçu du payload
    """
    use_case = get_publish_section_version_use_case()
    now = datetime.now(UTC)
    workflow_metadata = {
        "execution_id": f"run-{now.strftime('%Y-%m-%d')}-001",
        "executed_at": now.isoformat(),
        "version": "1.0.0",
    }
    try:
        return await use_case.execute(
            project_unique_id=project_unique_id,
            section_key="consolidated_data",
            content=content,
            field_sources=field_sources,
            workflow_id="agent-haiku-files-v1",
            workflow_name="agent-haiku-files-v1",
            workflow_metadata=workflow_metadata,
        )
    except PermissionError as e:
        logger.exception("Publish auth failed for project %s: %s", project_unique_id, e)
        raise ToolError(str(e)) from e
    except (ConnectionError, TimeoutError) as e:
        logger.exception(
            "Publish network error for project %s: %s", project_unique_id, e
        )
        raise ToolError(str(e)) from e
    except Exception as e:
        logger.exception(
            "Failed to publish section version for project %s: %s", project_unique_id, e
        )
        raise ToolError(f"Failed to publish section version: {e}") from e
