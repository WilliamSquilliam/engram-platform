"""SQLAlchemy engine/session + schema migration entrypoint. SQLite locally,
Postgres on AWS via DATABASE_URL. Schema is owned by Alembic (../alembic): startup
runs `alembic upgrade head`, so SQLite and Postgres get identical, versioned DDL."""
import logging
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from .config import DATABASE_URL

logger = logging.getLogger(__name__)

connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
# pool_pre_ping: transparently replace connections the DB closed while idle
# (RDS failover, idle timeout) instead of surfacing them as request errors.
engine = create_engine(DATABASE_URL, connect_args=connect_args, pool_pre_ping=True, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


class Base(DeclarativeBase):
    pass


def init_db() -> None:
    """Bring the database schema to the latest Alembic revision."""
    from alembic import command
    from alembic.config import Config
    from sqlalchemy import inspect

    from . import models  # noqa: F401  (register mappers)

    backend_dir = Path(__file__).resolve().parents[1]  # platform/backend
    cfg = Config(str(backend_dir / "alembic.ini"))
    cfg.set_main_option("script_location", str(backend_dir / "alembic"))
    cfg.set_main_option("sqlalchemy.url", DATABASE_URL.replace("%", "%%"))

    # A DB created before Alembic (old create_all path) has the tables but no
    # alembic_version. Stamp it at the baseline so upgrade doesn't try to re-create
    # existing tables; the old _ensure_columns kept it at the 0001 schema.
    tables = set(inspect(engine).get_table_names())
    if tables and "alembic_version" not in tables:
        logger.info("Stamping pre-Alembic database at baseline 0001_initial")
        command.stamp(cfg, "0001_initial")

    logger.info("Running database migrations (alembic upgrade head)")
    command.upgrade(cfg, "head")
