from domain.ports.bricks_api_port import BricksApiPort, BricksDocumentInfo


class ListBricksDocumentsUseCase:
    def __init__(self, bricks_api: BricksApiPort) -> None:
        self.bricks_api = bricks_api

    async def execute(self, project_id: str) -> list[BricksDocumentInfo]:
        return await self.bricks_api.list_project_documents(project_id=project_id)
