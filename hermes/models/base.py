"""SQLAlchemy declarative base and session management."""

from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from config import get_settings


class Base(DeclarativeBase):
    """Declarative base for all ORM models."""

    pass


def get_engine():
    """Create the async SQLAlchemy engine.

    Returns:
        AsyncEngine instance configured from settings.
    """
    settings = get_settings()
    # Convert postgresql:// to postgresql+asyncpg://
    db_url = str(settings.database_url).replace(
        "postgresql://", "postgresql+asyncpg://"
    )
    return create_async_engine(
        db_url,
        echo=settings.debug,
        pool_size=5,
        max_overflow=10,
        pool_pre_ping=True,
    )


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    """Create an async session factory.

    Returns:
        async_sessionmaker bound to the engine.
    """
    engine = get_engine()
    return async_sessionmaker(engine, expire_on_commit=False)


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """Yield an async database session.

    Yields:
        AsyncSession for database operations.
    """
    factory = get_session_factory()
    async with factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
