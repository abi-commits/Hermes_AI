"""FastAPI application factory and lifespan management."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import get_settings
from hermes import __version__
from hermes.api import (
    calls_router,
    health_router,
    knowledge_router,
    metrics_router,
    tts_router,
    twilio_router,
)
from hermes.api.metrics import APP_INFO, MetricsCollector
from hermes.utils.logging import configure_logging
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


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise and shut down all application services."""
    settings = get_settings()
    settings.validate_production_requirements()

    # Configure unified structured logging
    configure_logging(environment=settings.app_env, log_level=settings.log_level)

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

        # 1. Thread Pool
        executor = ThreadPoolExecutor(
            max_workers=settings.thread_pool_workers, thread_name_prefix="hermes-worker"
        )
        app.state.executor = executor

        # 2. Prompts
        prompt_manager = PromptManager(
            system_prompts_dir=settings.prompt_dir_system,
            few_shot_dir=settings.prompt_dir_few_shot,
        )

        # 3. RAG Service
        rag_service = ChromaRAGService(
            persist_directory=settings.chroma_persist_dir,
            collection_name=settings.chroma_collection_name,
        )
        # ── FIX: Warm up in background to avoid blocking lifespan ──
        if not settings.debug:
            asyncio.create_task(rag_service.warm_up(), name="rag-warmup")
            logger.info("rag_warm_up_task_started")

        # 4. LLM Service
        llm_config = LLMConfig(
            model=settings.gemini_model,
            temperature=settings.gemini_temperature,
            max_output_tokens=settings.gemini_max_tokens,
        )
        llm_service = GeminiLLMService(
            api_key=settings.gemini_api_key,
            config=llm_config,
            prompt_manager=prompt_manager,
        )

        # 5. TTS Service
        tts_service = build_tts_service(settings)
        tts_service.set_executor(executor)

        # 6. Call Orchestrator
        bundle = ServiceBundle(
            stt_factory=lambda: DeepgramSTTService(),
            llm_service=llm_service,
            tts_service=tts_service,
            rag_service=rag_service,
        )
        orchestrator = CallOrchestrator(bundle)
        app.state.orchestrator = orchestrator

        # Attach orchestrator to connection manager for WebSocket routing
        connection_manager.set_orchestrator(orchestrator)
        app.state.connection_manager = connection_manager

        # Set app info for metrics
        APP_INFO.info(
            {
                "version": __version__,
                "environment": settings.app_env,
                "tts_provider": settings.tts_provider,
            }
        )

        yield

    finally:
        # Shutdown
        logger.info("shutdown_starting")
        if hasattr(app.state, "orchestrator"):
            await app.state.orchestrator.shutdown()
        if hasattr(app.state, "executor"):
            app.state.executor.shutdown(wait=True)
        logger.info("shutdown_complete")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="Hermes AI",
        description="Low-latency voice AI orchestration platform.",
        version=__version__,
        lifespan=lifespan,
    )

    # Middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include routers (tags are defined internally in each router)
    from hermes.websocket.handler import websocket_router

    app.include_router(health_router)
    app.include_router(metrics_router)
    app.include_router(twilio_router)
    app.include_router(calls_router)
    app.include_router(knowledge_router)
    app.include_router(tts_router)
    app.include_router(websocket_router, prefix="/stream")

    return app


# Create the default app instance
app = create_app()
