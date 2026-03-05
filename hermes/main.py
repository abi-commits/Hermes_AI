"""FastAPI application factory and lifespan management."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import get_settings
from hermes import __version__
from hermes.api import health_router, metrics_router
from hermes.api.metrics import APP_INFO, MetricsCollector
from hermes.services.rag import ChromaRAGService
from hermes.services.tts import ChatterboxTTSService
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
        version=__version__,
    )

    # Startup: Initialize services
    try:
        # Initialize ChromaRAG service
        app.state.rag_service = ChromaRAGService()
        logger.info("rag_service_initialized")

        # Initialize TTS service with thread pool
        tts_executor = ThreadPoolExecutor(
            max_workers=settings.chatterbox_num_workers,
            thread_name_prefix="tts-worker",
        )
        app.state.tts_executor = tts_executor
        app.state.tts_service = ChatterboxTTSService(
            device=settings.chatterbox_device,
            watermark_key=(
                bytes.fromhex(settings.chatterbox_watermark_key)
                if settings.chatterbox_watermark_key
                else None
            ),
            num_workers=settings.chatterbox_num_workers,
        )
        app.state.tts_service.set_executor(tts_executor)
        logger.info("tts_service_initialized", device=settings.chatterbox_device)

        # Update Prometheus app info
        APP_INFO.info({"version": __version__, "name": settings.app_name})

        yield

    except Exception as e:
        logger.error("startup_failed", error=str(e))
        raise

    finally:
        # Shutdown: Cleanup resources
        logger.info("shutdown")

        if hasattr(app.state, "tts_executor"):
            app.state.tts_executor.shutdown(wait=False)
            logger.info("tts_executor_shutdown")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application.

    Returns:
        FastAPI: Configured FastAPI application.
    """
    settings = get_settings()
    setup_logging()

    app = FastAPI(
        title=settings.app_name,
        version=__version__,
        description="AI-powered voice support service",
        debug=settings.debug,
        lifespan=lifespan,
    )

    # CORS middleware — restrict origins in production
    allowed_origins = ["*"] if settings.is_development else [f"https://{settings.host}"]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
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
