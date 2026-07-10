from domain.ports.postgres_health_port import PostgresHealthPort
from domain.ports.storage_port import StoragePort


class LivenessCheckUseCase:
    """Health-check use case that pings PostgreSQL and object storage."""

    def __init__(
        self,
        storage: StoragePort,
        postgres_health: PostgresHealthPort,
        bucket: str,
    ) -> None:
        self._storage = storage
        self._postgres_health = postgres_health
        self._bucket = bucket

    async def execute(self) -> dict:
        """Ping PostgreSQL and MinIO and return aggregated health status.

        Returns:
            Dict with ``status`` ("healthy" or "degraded") and per-component
            ``checks`` mapping.
        """
        checks: dict[str, str] = {}

        pg_ok = await self._postgres_health.ping()
        checks["postgres"] = "ok" if pg_ok else "unreachable"

        minio_ok = await self._storage.ping(self._bucket)
        checks["minio"] = "ok" if minio_ok else "unreachable"

        healthy = all(v == "ok" for v in checks.values())
        return {"status": "healthy" if healthy else "degraded", "checks": checks}
