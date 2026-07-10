from domain.ports.bricks_api_port import BricksApiPort, BricksDocumentInfo


class ListBricksDocumentsUseCase:
    """List documents available in a Bricks project."""

    def __init__(self, bricks_api: BricksApiPort) -> None:
        self.bricks_api = bricks_api

    async def execute(self, project_id: str) -> list[BricksDocumentInfo]:
        """Retrieve the list of documents for the given Bricks project.

        Args:
            project_id: The Bricks project unique identifier.

        Returns:
            List of BricksDocumentInfo metadata objects.
        """
        return await self.bricks_api.list_project_documents(project_id=project_id)
