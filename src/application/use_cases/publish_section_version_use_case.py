import logging

from domain.ports.bricks_api_port import BricksApiPort, SectionVersionResult

logger = logging.getLogger(__name__)


class PublishSectionVersionUseCase:
    def __init__(
        self,
        bricks_api: BricksApiPort,
        dry_run: bool,
    ) -> None:
        self.bricks_api = bricks_api
        self.dry_run = dry_run

    async def execute(
        self,
        project_unique_id: str,
        section_key: str,
        content: dict,
        field_sources: list[dict],
        workflow_id: str,
        workflow_name: str,
        workflow_metadata: dict | None = None,
    ) -> SectionVersionResult:
        payload = {
            "projectUniqueId": project_unique_id,
            "sectionKey": section_key,
            "content": content,
            "fieldSources": field_sources,
            "workflowId": workflow_id,
            "workflowName": workflow_name,
            "workflowMetadata": workflow_metadata
            if workflow_metadata is not None
            else {},
        }

        if self.dry_run:
            logger.info(f"Payload : {payload}")
            return SectionVersionResult(
                success=True,
                message="DRY RUN — no API call made",
                dry_run=True,
                payload_preview=payload,
            )

        return await self.bricks_api.publish_section_version(payload=payload)
