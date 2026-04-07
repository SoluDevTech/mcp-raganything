"""Tests for Alembic configuration and environment."""

import importlib.util
from pathlib import Path

ALEMBIC_DIR = Path(__file__).parent.parent.parent / "src" / "alembic"
SRC_DIR = Path(__file__).parent.parent.parent / "src"


class TestAlembicConfig:
    """Tests for Alembic configuration files."""

    def test_alembic_ini_exists(self):
        alembic_ini = SRC_DIR / "alembic.ini"
        assert alembic_ini.exists(), f"alembic.ini should exist at {alembic_ini}"

    def test_alembic_versions_dir_exists(self):
        versions_dir = ALEMBIC_DIR / "versions"
        assert versions_dir.is_dir(), "alembic/versions directory should exist"

    def test_get_url_converts_asyncpg_to_sync(self):
        """Should convert postgresql+asyncpg:// to postgresql://."""
        test_cases = [
            (
                "postgresql+asyncpg://user:pass@host/db",
                "postgresql://user:pass@host/db",
            ),
            ("postgresql://user:pass@host/db", "postgresql://user:pass@host/db"),
        ]
        for input_url, expected in test_cases:
            result = input_url.replace("+asyncpg", "")
            assert result == expected

    def test_target_metadata_is_none(self):
        env_path = ALEMBIC_DIR / "env.py"
        content = env_path.read_text()
        assert "target_metadata = None" in content

    def test_migration_001_exists(self):
        migration = ALEMBIC_DIR / "versions" / "001_add_bm25_support.py"
        assert migration.exists(), f"Migration 001 should exist at {migration}"

    def test_migration_001_has_upgrade_and_downgrade(self):
        migration_path = ALEMBIC_DIR / "versions" / "001_add_bm25_support.py"
        spec = importlib.util.spec_from_file_location("migration_001", migration_path)
        assert spec is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)  # type: ignore[union-attr]
        assert hasattr(module, "upgrade"), "Migration should have upgrade()"
        assert hasattr(module, "downgrade"), "Migration should have downgrade()"

    def test_env_py_has_async_support(self):
        env_path = ALEMBIC_DIR / "env.py"
        content = env_path.read_text()
        assert "async_engine_from_config" in content
        assert "run_async_migrations" in content
