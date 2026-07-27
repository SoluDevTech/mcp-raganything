"""Startup rehydration of generated (openapi) MCP servers.

At application startup the registry is scanned for entries whose
``source_type == "openapi"``; each is re-mounted via the
:class:`GeneratedMcpRunnerPort` so the previously mounted servers become
reachable again after a restart. Failures are logged and skipped so a single
broken spec never blocks application startup.
"""

import logging
from typing import Any

from domain.logging.messages import LogMessage
from domain.ports.generated_mcp_runner import GeneratedMcpRunnerPort
from domain.ports.mcp_server_registry_store import McpServerRegistryStore

logger = logging.getLogger(__name__)


async def rehydrate_openapi_servers(
    store: McpServerRegistryStore,
    runner: GeneratedMcpRunnerPort,
    factory=None,
) -> None:
    """Re-mount every openapi entry from the store via the runner.

    Reads all entries once, filters to ``source_type == "openapi"``, and mounts
    each. Entries whose ``headers`` are populated pass them through to
    ``runner.mount``. Any mount failure is logged and skipped so a single broken
    spec never blocks startup.

    Args:
        store: The MCP server registry store (read-only here).
        runner: The generated MCP runner used to mount servers.
        factory: Optional :class:`OpenApiMcpFactory` used to fetch the spec for
            each entry before mounting. When omitted, the entry's
            ``openapi_url`` is passed as the spec placeholder (only suitable for
            tests/mocks that do not inspect the spec).
    """
    logger.info(LogMessage.MCP_REHYDRATION_STARTED)
    entries = await store.list_all()

    mounted = 0
    skipped = 0
    for entry in entries:
        if entry.source_type != "openapi":
            continue
        try:
            spec: Any
            if factory is not None:
                spec = await factory.fetch_spec(
                    entry.openapi_url or "", headers=entry.headers
                )
            else:
                spec = entry.openapi_url
            await runner.mount(entry.name, spec, headers=entry.headers)
            mounted += 1
        except Exception as e:  # noqa: BLE001 — one broken spec must not block startup
            skipped += 1
            logger.warning(LogMessage.MCP_REHYDRATION_ENTRY_FAILED, entry.name, e)

    logger.info(LogMessage.MCP_REHYDRATION_DONE, mounted, skipped)

    if skipped > 0:
        logger.error(LogMessage.MCP_REHYDRATION_SKIPPED_SUMMARY, skipped)
