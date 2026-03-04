"""FastAPI application factory and lifespan management."""

import logging
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import get_settings
from hermes.api import health_router, metrics_router
from hermes.websocket.handler import websocket_router

logger = structlog.get_logger(__name__)


def setup_logging() -> None:
    """Configure structured logging for the application."""
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
            if get_settings().is_production
            else structlog.dev.ConsoleRenderer(),
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan events.

    Args:
        app: FastAPI application instance.

    Yields:
        None
    """
    settings = get_settings()
    logger.info(
        "startup",
        app_name=settings.app_name,
        environment=settings.app_env,
        debug=settings.debug,
    )

    # Startup: Initialize services
    try:
        # Initialize vector database connection
        from hermes.services.vector_db import VectorDB

        app.state.vector_db = VectorDB()
        await app.state.vector_db.connect()
        logger.info("vector_db_connected")

        # Initialize Redis connection
        import redis.asyncio as redis

        app.state.redis = redis.from_url(str(settings.redis_url))
        await app.state.redis.ping()
        logger.info("redis_connected")

        yield

    except Exception as e:
        logger.error("startup_failed", error=str(e))
        raise

    finally:
        # Shutdown: Cleanup resources
        logger.info("shutdown")

        if hasattr(app.state, "vector_db"):
            await app.state.vector_db.disconnect()
            logger.info("vector_db_disconnected")

        if hasattr(app.state, "redis"):
            await app.state.redis.close()
            logger.info("redis_disconnected")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application.

    Returns:
        FastAPI: Configured FastAPI application.
    """
    settings = get_settings()
    setup_logging()

    app = FastAPI(
        title=settings.app_name,
        version="0.1.0",
        description="AI-powered voice support service",
        debug=settings.debug,
        lifespan=lifespan,
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include routers
    app.include_router(health_router, tags=["health"])
    app.include_router(metrics_router, prefix="/metrics", tags=["metrics"])
    app.include_router(websocket_router, prefix="/stream", tags=["websocket"])

    return app


# Application instance
app = create_app()

if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "hermes.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.is_development,
        workers=settings.workers if settings.is_production else 1,
    )
