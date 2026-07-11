import logging
import ssl as ssl_module

import asyncpg

from config import DatabaseConfig
from domain.logging.messages import LogMessage

logger = logging.getLogger(__name__)

_SSL_MODES_NO_VERIFY = frozenset({"require", "prefer"})


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
            ctx = self._build_ssl_context()
            if ctx is not None:
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

    def _build_ssl_context(self) -> ssl_module.SSLContext | None:
        """Build an SSL context appropriate for the configured sslmode.

        - ``require`` / ``prefer``: encryption without certificate verification
          (CERT_NONE, check_hostname disabled); a warning is logged so operators
          know verification is intentionally skipped.
        - ``verify-ca``: verify the server certificate against the CA but skip
          hostname verification (CERT_REQUIRED, check_hostname disabled).
        - ``verify-full``: full verification using ``create_default_context``
          defaults (CERT_REQUIRED, check_hostname enabled).
        - ``disable`` or ``None``: let asyncpg handle the connection natively,
          without a custom SSL context.
        """
        if not self._ssl_mode or self._ssl_mode == "disable":
            return None

        if self._ssl_mode in _SSL_MODES_NO_VERIFY:
            logger.warning(
                "SSL mode '%s' disables certificate verification for the "
                "PostgreSQL health check; use 'verify-ca' or 'verify-full' "
                "in production to prevent MITM attacks.",
                self._ssl_mode,
            )

        ctx = ssl_module.create_default_context()

        if self._ssl_mode in _SSL_MODES_NO_VERIFY:
            ctx.check_hostname = False  # NOSONAR — sslmode=require/prefer means encryption without verification by PostgreSQL spec
            ctx.verify_mode = ssl_module.CERT_NONE  # NOSONAR — see above; warning logged above
        elif self._ssl_mode == "verify-ca":
            ctx.check_hostname = False
        # verify-full (and any unknown sslmode) keeps full verification defaults.

        return ctx
