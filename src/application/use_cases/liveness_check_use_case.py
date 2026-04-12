from domain.ports.postgres_health_port import PostgresHealthPort
from domain.ports.storage_port import StoragePort


class LivenessCheckUseCase:
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
        checks: dict[str, str] = {}

        pg_ok = await self._postgres_health.ping()
        checks["postgres"] = "ok" if pg_ok else "unreachable"

        minio_ok = await self._storage.ping(self._bucket)
        checks["minio"] = "ok" if minio_ok else "unreachable"

        healthy = all(v == "ok" for v in checks.values())
        return {"status": "healthy" if healthy else "degraded", "checks": checks}
