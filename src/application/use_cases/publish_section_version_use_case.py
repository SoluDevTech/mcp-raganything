from domain.ports.bricks_api_port import BricksApiPort, SectionVersionResult


class PublishSectionVersionUseCase:
    def __init__(
        self,
        bricks_api: BricksApiPort,
        dry_run: bool,
        target_url: str,
    ) -> None:
        self.bricks_api = bricks_api
        self.dry_run = dry_run
        self.target_url = target_url

    async def execute(
        self,
        project_unique_id: str,
        section_key: str,
        content: dict,
        workflow_id: str,
        workflow_name: str,
        workflow_metadata: dict | None = None,
    ) -> SectionVersionResult:
        payload = {
            "project_unique_id": project_unique_id,
            "section_key": section_key,
            "content": content,
            "workflow_id": workflow_id,
            "workflow_name": workflow_name,
            "workflow_metadata": workflow_metadata
            if workflow_metadata is not None
            else {},
        }

        if self.dry_run:
            return SectionVersionResult(
                success=True,
                message="DRY RUN — no API call made",
                dry_run=True,
                payload_preview=payload,
                target_url=self.target_url,
            )

        return await self.bricks_api.publish_section_version(payload=payload)
