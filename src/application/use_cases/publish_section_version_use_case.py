import logging

from domain.logging.messages import LogMessage
from domain.ports.bricks_api_port import BricksApiPort, SectionVersionResult

logger = logging.getLogger(__name__)


class PublishSectionVersionUseCase:
    """Publish a section version to the Bricks API.

    In dry-run mode, logs the payload and returns a preview without
    making an HTTP call.
    """

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
        """Publish (or preview) a section version payload to Bricks.

        Args:
            project_unique_id: Bricks project identifier.
            section_key: Target section key within the project.
            content: Section content payload.
            field_sources: List of field source descriptors.
            workflow_id: Identifier of the workflow triggering the publish.
            workflow_name: Human-readable workflow name.
            workflow_metadata: Optional extra metadata for the workflow.

        Returns:
            SectionVersionResult with success flag and payload preview (dry-run).
        """
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
            logger.info(LogMessage.PAYLOAD_PREVIEW, payload)
            return SectionVersionResult(
                success=True,
                message="DRY RUN — no API call made",
                dry_run=True,
                payload_preview=payload,
            )

        return await self.bricks_api.publish_section_version(payload=payload)
