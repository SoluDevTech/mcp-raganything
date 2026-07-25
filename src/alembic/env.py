"""Alembic migration environment for mcp-raganything.

Runs migrations asynchronously against the same PostgreSQL database used by the
application (``DATABASE_URL`` from :class:`DatabaseConfig`, normalized to
``postgresql+asyncpg://``). Migrations are written as raw SQL via ``op.execute``
— there is no SQLAlchemy ORM metadata to autogenerate from.

The migration state is tracked in the ``raganything_alembic_version`` table
(separate from composable-agents' ``alembic_version`` table) so both services
can share the same database without colliding on Alembic's bookkeeping.
"""

import asyncio

from sqlalchemy import pool
from sqlalchemy.ext.asyncio import async_engine_from_config

from alembic import context
from config import AppConfig, DatabaseConfig
from infrastructure.logging import configure_logging

# Alembic context objects.
config = context.config

# Configure logging once (idempotent) so migration logs respect LOG_LEVEL.
configure_logging(AppConfig())

# No ORM metadata — migrations are hand-written SQL.
target_metadata = None


def get_url() -> str:
    """Build the SQLAlchemy database URL from application settings.

    ``DatabaseConfig`` normalizes the scheme to ``postgresql+asyncpg://`` and
    validates the host, so the URL is engine-ready.
    """
    return DatabaseConfig().DATABASE_URL


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode (emit SQL to stdout, no DB connection)."""
    url = get_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        version_table="raganything_alembic_version",
    )

    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection) -> None:
    """Run migrations within a synchronous connection callback."""
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        version_table="raganything_alembic_version",
    )

    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """Run migrations in 'online' mode with an async engine."""
    configuration = config.get_section(config.config_ini_section, {})
    configuration["sqlalchemy.url"] = get_url()

    connectable = async_engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode."""
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
