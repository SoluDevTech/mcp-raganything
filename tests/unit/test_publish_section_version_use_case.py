"""Tests for PublishSectionVersionUseCase — payload construction and dry_run logic."""

from unittest.mock import AsyncMock

import pytest

from application.use_cases.publish_section_version_use_case import (
    PublishSectionVersionUseCase,
)
from domain.ports.bricks_api_port import SectionVersionResult


class TestPublishSectionVersionUseCase:
    """Tests for PublishSectionVersionUseCase."""

    @pytest.fixture
    def use_case_dry_run(
        self, mock_bricks_api: AsyncMock
    ) -> PublishSectionVersionUseCase:
        """Create a use case with dry_run=True (default)."""
        return PublishSectionVersionUseCase(
            bricks_api=mock_bricks_api,
            dry_run=True,
        )

    @pytest.fixture
    def use_case_live(self, mock_bricks_api: AsyncMock) -> PublishSectionVersionUseCase:
        """Create a use case with dry_run=False."""
        return PublishSectionVersionUseCase(
            bricks_api=mock_bricks_api,
            dry_run=False,
        )

    async def test_dry_run_does_not_call_api(
        self,
        use_case_dry_run: PublishSectionVersionUseCase,
        mock_bricks_api: AsyncMock,
    ) -> None:
        """When dry_run=True, should NOT call bricks_api.publish_section_version."""
        result = await use_case_dry_run.execute(
            project_unique_id="proj-123",
            section_key="intro",
            content="Hello world",
            workflow_id="wf-1",
            workflow_name="draft",
            workflow_metadata={"source": "test"},
        )

        mock_bricks_api.publish_section_version.assert_not_called()
        assert result.dry_run is True

    async def test_dry_run_returns_payload_preview(
        self,
        use_case_dry_run: PublishSectionVersionUseCase,
    ) -> None:
        """When dry_run=True, should return payload_preview in the result."""
        result = await use_case_dry_run.execute(
            project_unique_id="proj-123",
            section_key="summary",
            content="Summary text",
            workflow_id="wf-2",
            workflow_name="review",
            workflow_metadata={"version": 1},
        )

        assert result.success is True
        assert result.dry_run is True
        assert result.payload_preview is not None
        assert result.payload_preview["projectUniqueId"] == "proj-123"
        assert result.payload_preview["sectionKey"] == "summary"
        assert result.payload_preview["content"] == "Summary text"
        assert result.payload_preview["workflowId"] == "wf-2"
        assert result.payload_preview["workflowName"] == "review"
        assert result.payload_preview["workflowMetadata"] == {"version": 1}

    async def test_live_calls_bricks_api(
        self,
        use_case_live: PublishSectionVersionUseCase,
        mock_bricks_api: AsyncMock,
    ) -> None:
        """When dry_run=False, should call bricks_api.publish_section_version."""
        mock_bricks_api.publish_section_version.return_value = SectionVersionResult(
            success=True,
            message="Published",
            data={"id": "sv-123"},
        )

        result = await use_case_live.execute(
            project_unique_id="proj-123",
            section_key="intro",
            content="Hello world",
            workflow_id="wf-1",
            workflow_name="draft",
            workflow_metadata={},
        )

        mock_bricks_api.publish_section_version.assert_called_once()
        assert result.success is True
        assert result.dry_run is False

    async def test_live_passes_correct_payload(
        self,
        use_case_live: PublishSectionVersionUseCase,
        mock_bricks_api: AsyncMock,
    ) -> None:
        """When dry_run=False, should pass the correctly constructed payload."""
        mock_bricks_api.publish_section_version.return_value = SectionVersionResult(
            success=True,
            message="OK",
        )

        await use_case_live.execute(
            project_unique_id="proj-abc",
            section_key="results",
            content="Final results",
            workflow_id="wf-final",
            workflow_name="final_review",
            workflow_metadata={"approved_by": "admin"},
        )

        call_args = mock_bricks_api.publish_section_version.call_args
        payload = call_args[1].get("payload", call_args[0][0] if call_args[0] else None)
        if payload is None:
            payload = call_args.kwargs.get("payload")
        assert payload["projectUniqueId"] == "proj-abc"
        assert payload["sectionKey"] == "results"
        assert payload["content"] == "Final results"
        assert payload["workflowId"] == "wf-final"
        assert payload["workflowName"] == "final_review"
        assert payload["workflowMetadata"] == {"approved_by": "admin"}

    async def test_workflow_metadata_defaults_to_empty_dict(
        self,
        use_case_dry_run: PublishSectionVersionUseCase,
    ) -> None:
        """When workflow_metadata is None, should send {} in the payload."""
        result = await use_case_dry_run.execute(
            project_unique_id="proj-123",
            section_key="intro",
            content="Hello",
            workflow_id="wf-1",
            workflow_name="draft",
        )

        assert result.payload_preview["workflowMetadata"] == {}

    async def test_workflow_metadata_none_in_live_payload(
        self,
        use_case_live: PublishSectionVersionUseCase,
        mock_bricks_api: AsyncMock,
    ) -> None:
        """When workflow_metadata is None in live mode, payload should contain {}."""
        mock_bricks_api.publish_section_version.return_value = SectionVersionResult(
            success=True,
        )

        await use_case_live.execute(
            project_unique_id="proj-123",
            section_key="intro",
            content="Hello",
            workflow_id="wf-1",
            workflow_name="draft",
        )

        call_args = mock_bricks_api.publish_section_version.call_args
        payload = call_args[1].get("payload", call_args[0][0] if call_args[0] else None)
        if payload is None:
            payload = call_args.kwargs.get("payload")
        assert payload["workflowMetadata"] == {}

    async def test_live_propagates_api_errors(
        self,
        mock_bricks_api: AsyncMock,
    ) -> None:
        """Should propagate errors from the API when publishing live."""
        mock_bricks_api.publish_section_version.side_effect = ConnectionError(
            "API connection failed"
        )
        use_case = PublishSectionVersionUseCase(
            bricks_api=mock_bricks_api,
            dry_run=False,
        )

        with pytest.raises(ConnectionError, match="API connection failed"):
            await use_case.execute(
                project_unique_id="proj-123",
                section_key="intro",
                content="Hello",
                workflow_id="wf-1",
                workflow_name="draft",
            )

    async def test_dry_run_returns_section_version_result(
        self,
        use_case_dry_run: PublishSectionVersionUseCase,
    ) -> None:
        """Dry run result should be a SectionVersionResult instance."""
        result = await use_case_dry_run.execute(
            project_unique_id="proj-123",
            section_key="intro",
            content="Hello",
            workflow_id="wf-1",
            workflow_name="draft",
        )

        assert isinstance(result, SectionVersionResult)