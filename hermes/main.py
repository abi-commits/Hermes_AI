"""FastAPI application factory and lifespan management."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import get_settings
from hermes import __version__
from hermes.api import calls_router, health_router, knowledge_router, metrics_router, twilio_router
from hermes.api.metrics import APP_INFO, MetricsCollector
from hermes.websocket.manager import connection_manager

logger = structlog.get_logger(__name__)


def build_tts_service(settings):
    """Create the configured TTS service implementation."""
    if settings.tts_provider == "modal_remote":
        from hermes.services.tts import ModalRemoteTTSService
        service = ModalRemoteTTSService(
            app_name=settings.modal_tts_app_name,
            class_name=settings.modal_tts_class_name,
            sample_rate=settings.modal_tts_sample_rate,
            default_chunk_size=settings.modal_tts_chunk_size,
            default_embed_watermark=settings.modal_tts_embed_watermark,
        )
        logger.info(
            "tts_service_selected",
            provider="modal_remote",
            modal_app=settings.modal_tts_app_name,
            modal_class=settings.modal_tts_class_name,
        )
        return service

    # Local chatterbox requires torch/torchaudio
    from hermes.services.tts import ChatterboxTTSService
    service = ChatterboxTTSService(
        device=settings.chatterbox_device,
        watermark_key=(
            bytes.fromhex(settings.chatterbox_watermark_key)
            if settings.chatterbox_watermark_key
            else None
        ),
        num_workers=settings.chatterbox_num_workers,
    )
    logger.info(
        "tts_service_selected",
        provider="chatterbox",
        device=settings.chatterbox_device,
    )
    return service


def setup_logging() -> None:
    """Configure structured logging with JSON output in production."""
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
    """Initialise and shut down all application services."""
    settings = get_settings()
    settings.validate_production_requirements()
    logger.info(
        "startup",
        app_name=settings.app_name,
        environment=settings.app_env,
        debug=settings.debug,
        version=__version__,
    )

    # Startup: Initialize services
    try:
        # Import services locally to keep the main process lightweight
        from hermes.core.orchestrator import CallOrchestrator, ServiceBundle
        from hermes.models.llm import LLMConfig
        from hermes.prompts.prompt_manager import PromptManager
        from hermes.services.llm import GeminiLLMService
        from hermes.services.rag import ChromaRAGService
        from hermes.services.stt import DeepgramSTTService

        # 1. RAG — warm up eagerly so the first call pays no cold-start cost
        app.state.rag_service = ChromaRAGService()
        try:
            await app.state.rag_service.warm_up()
            logger.info("rag_service_initialized")
        except Exception as rag_exc:
            logger.warning("rag_warm_up_failed", error=str(rag_exc))

        # 2. TTS — Chatterbox with dedicated thread pool
        tts_executor = ThreadPoolExecutor(
            max_workers=settings.chatterbox_num_workers,
            thread_name_prefix="tts-worker",
        )
        app.state.tts_executor = tts_executor
        app.state.tts_service = build_tts_service(settings)
        app.state.tts_service.set_executor(tts_executor)
        logger.info("tts_service_initialized", provider=settings.tts_provider)

        # 3. LLM — shared Gemini instance with prompt management (stateless per-request)
        prompt_manager = PromptManager()
        app.state.llm_service = GeminiLLMService(
            api_key=settings.gemini_api_key,
            config=LLMConfig(
                model_name=settings.gemini_model,
                temperature=settings.gemini_temperature,
                max_output_tokens=settings.gemini_max_tokens,
            ),
            prompt_manager=prompt_manager,
            prompt_name=getattr(settings, "llm_prompt_name", "default"),
        )
        logger.info("llm_service_initialized", model=settings.gemini_model)

        # 4. Build the service bundle and orchestrator
        bundle = ServiceBundle(
            stt_factory=DeepgramSTTService,
            llm_service=app.state.llm_service,
            tts_service=app.state.tts_service,
            rag_service=app.state.rag_service,
        )
        app.state.orchestrator = CallOrchestrator(bundle)
        connection_manager.set_orchestrator(app.state.orchestrator)
        app.state.connection_manager = connection_manager
        logger.info("orchestrator_initialized")

        # Update Prometheus app info
        APP_INFO.info({"version": __version__, "name": settings.app_name})

        yield

    except Exception as e:
        logger.error("startup_failed", error=str(e))
        raise

    finally:
        # Shutdown: gracefully terminate active calls, then release resources
        logger.info("shutdown")

        if hasattr(app.state, "orchestrator"):
            await app.state.orchestrator.shutdown()
            logger.info("orchestrator_shutdown")

        if hasattr(app.state, "tts_executor"):
            app.state.tts_executor.shutdown(wait=False)
            logger.info("tts_executor_shutdown")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()
    setup_logging()

    app = FastAPI(
        title=settings.app_name,
        version=__version__,
        description="AI-powered voice support service",
        debug=settings.debug,
        lifespan=lifespan,
    )

    # Import router here to ensure it uses the lightweight websocket_router
    from hermes.websocket.handler import websocket_router

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
    app.include_router(twilio_router, tags=["twilio"])
    app.include_router(calls_router, tags=["calls"])
    app.include_router(knowledge_router, tags=["knowledge"])
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
