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

# ── TOP LEVEL IMPORTS TO PREVENT COLD-START DELAYS ──
from hermes.core.orchestrator import CallOrchestrator, ServiceBundle
from hermes.models.llm import LLMConfig
from hermes.prompts.prompt_manager import PromptManager
from hermes.services.llm import GeminiLLMService
from hermes.services.rag import ChromaRAGService
from hermes.services.stt import DeepgramSTTService

logger = structlog.get_logger(__name__)


def build_tts_service(settings):
    """Create the configured TTS service implementation."""
    if settings.tts_provider == "modal_remote":
        from hermes.services.tts import ModalRemoteTTSService

        return ModalRemoteTTSService(
            app_name=settings.modal_tts_app_name,
            class_name=settings.modal_tts_class_name,
            sample_rate=settings.modal_tts_sample_rate,
            default_chunk_size=settings.modal_tts_chunk_size,
            default_embed_watermark=settings.modal_tts_embed_watermark,
        )

    from hermes.services.tts import ChatterboxTTSService
    return ChatterboxTTSService(
        device=settings.chatterbox_device,
        num_workers=settings.chatterbox_num_workers,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise and shut down all application services."""
    settings = get_settings()
    configure_logging(environment=settings.app_env, log_level=settings.log_level)

    logger.info("lifespan_starting", version=__version__)

    # 1. Critical plumbing
    executor = ThreadPoolExecutor(
        max_workers=settings.thread_pool_workers, thread_name_prefix="hermes-worker"
    )
    app.state.executor = executor

    async def _async_init():
        """Heavy initialization in background."""
        try:
            # 2. Service Initialization
            prompt_manager = PromptManager()
            rag_service = ChromaRAGService()
            
            llm_service = GeminiLLMService(
                api_key=settings.gemini_api_key,
                config=LLMConfig(
                    model_name=settings.gemini_model,
                    temperature=settings.gemini_temperature,
                ),
                prompt_manager=prompt_manager,
            )

            tts_service = build_tts_service(settings)
            tts_service.set_executor(executor)

            # 3. Call Orchestrator
            bundle = ServiceBundle(
                stt_factory=lambda: DeepgramSTTService(),
                llm_service=llm_service,
                tts_service=tts_service,
                rag_service=rag_service,
            )
            orchestrator = CallOrchestrator(bundle)
            
            # 4. ── ATTACH IMMEDIATELY ──
            # This makes the orchestrator available to the WebSocket handler
            # BEFORE we do the slow RAG warmup.
            app.state.orchestrator = orchestrator
            connection_manager.set_orchestrator(orchestrator)
            logger.info("orchestrator_attached_and_ready")
            
            # 5. Final slow warm-up
            if not settings.debug:
                await rag_service.warm_up()
            
            logger.info("background_initialization_complete")
        except Exception as e:
            logger.exception("background_initialization_failed", error=str(e))

    # Launch initialization in background
    init_task = asyncio.create_task(_async_init())
    
    yield

    # Shutdown
    init_task.cancel()
    if hasattr(app.state, "orchestrator"):
        await app.state.orchestrator.shutdown()
    executor.shutdown(wait=False)
    logger.info("lifespan_ended")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="Hermes AI",
        version=__version__,
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    from hermes.websocket.handler import websocket_router
    app.include_router(health_router)
    app.include_router(metrics_router)
    app.include_router(twilio_router)
    app.include_router(calls_router)
    app.include_router(knowledge_router)
    app.include_router(tts_router)
    app.include_router(websocket_router, prefix="/stream")

    return app

app = create_app()
