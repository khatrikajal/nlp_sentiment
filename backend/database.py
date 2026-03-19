"""
database.py — SQLAlchemy async database setup
"""

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm          import DeclarativeBase

# ── SQLite URL (async) ────────────────────────────────────────────────────────
DATABASE_URL = "sqlite+aiosqlite:///./sentiment_history.db"

# ── Async engine ──────────────────────────────────────────────────────────────
engine = create_async_engine(
    DATABASE_URL,
    echo=False,          # set True to log SQL queries
    connect_args={"check_same_thread": False},
)

# ── Session factory ───────────────────────────────────────────────────────────
AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
)

# ── Base class for ORM models ─────────────────────────────────────────────────
class Base(DeclarativeBase):
    pass


async def init_db() -> None:
    """Create all tables on startup."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_db():
    """FastAPI dependency — yields an async DB session."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
