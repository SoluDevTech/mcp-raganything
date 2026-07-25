"""Centralized logging configuration.

This module is the single place where logging is configured for the whole
application. Log *message templates* live in ``domain/logging/messages.py``;
this module only handles formatting, levels, handler wiring and the per-request
correlation id (``request_id``) propagation.
"""

import logging
import uuid
from contextvars import ContextVar

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

from config import AppConfig

# Per-request correlation id. Populated by RequestIdMiddleware and read by the
# RequestIdFilter so every log line within a request carries the same id.
request_id_ctx: ContextVar[str] = ContextVar("request_id", default="-")

# Log format including the correlation id slot.
_LOG_FORMAT = (
    "%(asctime)s | %(levelname)-8s | %(name)s | request_id=%(request_id)s | %(message)s"
)
_LOG_DATEFMT = "%Y-%m-%d %H:%M:%S"

# Third-party loggers that should be quieter than the application level.
_NOISY_LOGGERS = (
    "uvicorn",
    "uvicorn.access",
    "uvicorn.error",
    "sqlalchemy.engine",
    "httpx",
    "httpcore",
    "asyncio",
    "cachetools",
)


class RequestIdFilter(logging.Filter):
    """Inject the current ``request_id`` context var into every log record."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_ctx.get()
        return True


def configure_logging(app_config: AppConfig) -> None:
    """Configure root logging for the application.

    Idempotent: safe to call from ``main.py`` and tests.

    Args:
        app_config: Application config (reads ``LOG_LEVEL``).
    """
    level = getattr(logging, app_config.LOG_LEVEL.upper(), logging.INFO)

    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_LOG_DATEFMT))
    handler.addFilter(RequestIdFilter())

    root = logging.getLogger()
    # Remove existing handlers so repeated calls (e.g. tests) stay clean.
    root.handlers = [handler]
    root.setLevel(level)

    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)


class RequestIdMiddleware(BaseHTTPMiddleware):
    """Assign/propagate a correlation id for every HTTP request.

    Reads an incoming ``X-Request-ID`` header if present, otherwise generates a
    new one. The id is stored in the ``request_id_ctx`` context var (so it lands
    in every log line) and echoed back on the response.
    """

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex
        token = request_id_ctx.set(request_id)
        try:
            response = await call_next(request)
        finally:
            request_id_ctx.reset(token)
        response.headers["X-Request-ID"] = request_id
        return response
