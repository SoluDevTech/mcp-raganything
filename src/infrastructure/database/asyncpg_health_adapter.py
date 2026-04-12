import logging

import asyncpg

from config import DatabaseConfig

logger = logging.getLogger(__name__)


class AsyncpgHealthAdapter:
    """PostgreSQL health check using asyncpg direct connection."""

    def __init__(self, db_config: DatabaseConfig) -> None:
        self._db_url = db_config.DATABASE_URL.replace("+asyncpg", "")

    async def ping(self) -> bool:
        try:
            conn = await asyncpg.connect(self._db_url)
            try:
                await conn.fetchval("SELECT 1")
                return True
            finally:
                await conn.close()
        except Exception:
            logger.warning("PostgreSQL health check failed", exc_info=True)
            return False
