import logging

import asyncpg

from config import DatabaseConfig
from domain.logging.messages import LogMessage

logger = logging.getLogger(__name__)


class AsyncpgHealthAdapter:
    """PostgreSQL health check using asyncpg direct connection."""

    def __init__(self, db_config: DatabaseConfig) -> None:
        self._db_url = db_config.asyncpg_url
        self._statement_cache_size = db_config.POSTGRES_STATEMENT_CACHE_SIZE
        self._ssl_mode = db_config.ssl_mode

    async def ping(self) -> bool:
        try:
            connect_kwargs: dict = {}
            if self._statement_cache_size is not None:
                connect_kwargs["statement_cache_size"] = self._statement_cache_size
            if self._ssl_mode:
                import ssl as ssl_module

                ctx = ssl_module.create_default_context()
                ctx.check_hostname = False
                ctx.verify_mode = ssl_module.CERT_NONE
                connect_kwargs["ssl"] = ctx
            conn = await asyncpg.connect(self._db_url, **connect_kwargs)
            try:
                await conn.fetchval("SELECT 1")
                return True
            finally:
                await conn.close()
        except Exception:
            logger.warning(LogMessage.POSTGRES_HEALTH_CHECK_FAILED, exc_info=True)
            return False
