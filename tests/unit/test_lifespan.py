"""Tests for FastAPI lifespan management."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestLifespan:
    """Tests for lifespan context managers in main.py."""

    @pytest.mark.asyncio
    async def test_db_lifespan_runs_migrations_on_startup(self):
        """Should run Alembic migrations on startup."""
        from main import db_lifespan

        mock_app = MagicMock()

        with (
            patch("main.bm25_adapter", None),
            patch("main.asyncio.to_thread") as mock_to_thread,
        ):
            mock_to_thread.return_value = None
            async with db_lifespan(mock_app):
                pass
            mock_to_thread.assert_called_once()

    @pytest.mark.asyncio
    async def test_db_lifespan_closes_bm25_pool_on_shutdown(self):
        """Should close BM25 adapter connection pool on shutdown."""
        from main import db_lifespan

        mock_app = MagicMock()
        mock_bm25 = AsyncMock()

        with patch("main.bm25_adapter", mock_bm25), patch("main.asyncio.to_thread"):
            async with db_lifespan(mock_app):
                pass
            mock_bm25.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_db_lifespan_handles_no_bm25_adapter(self):
        """Should handle gracefully when bm25_adapter is None."""
        from main import db_lifespan

        mock_app = MagicMock()

        with patch("main.bm25_adapter", None), patch("main.asyncio.to_thread"):
            async with db_lifespan(mock_app):
                pass

    @pytest.mark.asyncio
    async def test_db_lifespan_raises_on_migration_failure(self):
        """Should raise if migrations fail — refusing to start with broken schema."""
        from main import db_lifespan

        mock_app = MagicMock()

        with (
            patch("main.bm25_adapter", None),
            patch("main.asyncio.to_thread") as mock_to_thread,
        ):
            mock_to_thread.side_effect = Exception("Migration failed")
            with pytest.raises(Exception, match="Migration failed"):
                async with db_lifespan(mock_app):
                    pass
            mock_to_thread.assert_called_once()

    @pytest.mark.asyncio
    async def test_db_lifespan_handles_close_failure(self):
        """Should not crash if BM25 close fails."""
        from main import db_lifespan

        mock_app = MagicMock()
        mock_bm25 = AsyncMock()
        mock_bm25.close = AsyncMock(side_effect=Exception("Close failed"))

        with patch("main.bm25_adapter", mock_bm25), patch("main.asyncio.to_thread"):
            async with db_lifespan(mock_app):
                pass
            mock_bm25.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_alembic_upgrade_calls_command(self):
        """Should call alembic command.upgrade with head."""
        with (
            patch("main.command.upgrade") as mock_upgrade,
            patch("main.Config") as mock_config_cls,
        ):
            mock_cfg = MagicMock()
            mock_config_cls.return_value = mock_cfg
            from main import _run_alembic_upgrade

            _run_alembic_upgrade()
            mock_upgrade.assert_called_once_with(mock_cfg, "head")
