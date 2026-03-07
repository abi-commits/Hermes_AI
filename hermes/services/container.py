"""Service container for dependency injection.

Provides a centralized registry for service instances with proper
lifecycle management. Services are created once and shared across
the application rather than being instantiated per-call.
"""

from __future__ import annotations

import httpx
import structlog
from concurrent.futures import ThreadPoolExecutor

from config import get_settings
from hermes.core.exceptions import ConfigurationError

logger = structlog.get_logger(__name__)


class ServiceContainer:
    """Centralized service container with shared, pooled instances.

    Usage::

        container = ServiceContainer()
        await container.start()

        llm = container.llm_service
        stt = container.stt_service

        await container.stop()
    """

    def __init__(self) -> None:
        """Initialize the service container (services created on start)."""
        self.settings = get_settings()
        self._started = False

        # Shared HTTP client pool
        self._http_client: httpx.AsyncClient | None = None
        self._tts_executor: ThreadPoolExecutor | None = None

        # Service instances (lazy, set on start)
        self._stt_service = None
        self._llm_service = None
        self._tts_service = None
        self._rag_service = None
        self._vector_db = None

    # -- Lifecycle -------------------------------------------------------------

    async def start(self) -> None:
        """Initialize all services and shared resources."""
        if self._started:
            return

        logger.info("service_container_starting")

        # Shared HTTP client with connection pooling
        self._http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(60.0, connect=10.0),
            limits=httpx.Limits(
                max_connections=100,
                max_keepalive_connections=20,
                keepalive_expiry=30,
            ),
        )

        # TTS Executor
        self._tts_executor = ThreadPoolExecutor(
            max_workers=self.settings.chatterbox_num_workers,
            thread_name_prefix="tts-worker",
        )

        # Import here to avoid circular imports and allow lazy dependency loading
        from hermes.services.llm import GeminiLLMService
        from hermes.services.rag import ChromaRAGService
        from hermes.services.stt import DeepgramSTTService
        from hermes.models.llm import LLMConfig
        from hermes.prompts.prompt_manager import PromptManager

        # 1. RAG
        self._rag_service = ChromaRAGService()
        try:
            await self._rag_service.warm_up()
        except Exception as rag_exc:
            logger.warning("rag_warm_up_failed", error=str(rag_exc))

        # 2. TTS - Use conditional imports to avoid torch in API layer
        if self.settings.tts_provider == "modal_remote":
            from hermes.services.tts import ModalRemoteTTSService
            self._tts_service = ModalRemoteTTSService(
                app_name=self.settings.modal_tts_app_name,
                class_name=self.settings.modal_tts_class_name,
                sample_rate=self.settings.modal_tts_sample_rate,
                default_chunk_size=self.settings.modal_tts_chunk_size,
                default_embed_watermark=self.settings.modal_tts_embed_watermark,
            )
        else:
            # Local chatterbox requires torch/torchaudio
            from hermes.services.tts import ChatterboxTTSService
            self._tts_service = ChatterboxTTSService(
                device=self.settings.chatterbox_device,
                watermark_key=(
                    bytes.fromhex(self.settings.chatterbox_watermark_key)
                    if self.settings.chatterbox_watermark_key
                    else None
                ),
                num_workers=self.settings.chatterbox_num_workers,
            )
        
        if hasattr(self._tts_service, "set_executor"):
            self._tts_service.set_executor(self._tts_executor)

        # 3. LLM
        prompt_manager = PromptManager()
        self._llm_service = GeminiLLMService(
            api_key=self.settings.gemini_api_key,
            config=LLMConfig(
                model_name=self.settings.gemini_model,
                temperature=self.settings.gemini_temperature,
                max_output_tokens=self.settings.gemini_max_tokens,
            ),
            prompt_manager=prompt_manager,
        )

        # 4. STT (Factory/Class reference)
        self._stt_service = DeepgramSTTService

        self._started = True
        logger.info("service_container_started")

    async def stop(self) -> None:
        """Shut down all services and release resources."""
        if not self._started:
            return

        logger.info("service_container_stopping")

        if self._tts_executor:
            self._tts_executor.shutdown(wait=False)

        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None

        self._started = False
        logger.info("service_container_stopped")

    # -- Accessors -------------------------------------------------------------

    def _ensure_started(self) -> None:
        if not self._started:
            raise ConfigurationError(
                "ServiceContainer has not been started. Call await container.start() first."
            )

    @property
    def http_client(self) -> httpx.AsyncClient:
        """Shared HTTP client with connection pooling."""
        self._ensure_started()
        return self._http_client  # type: ignore[return-value]

    @property
    def stt_service(self):
        """Speech-to-Text service class (factory)."""
        self._ensure_started()
        return self._stt_service

    @property
    def llm_service(self):
        """LLM service instance."""
        self._ensure_started()
        return self._llm_service

    @property
    def tts_service(self):
        """TTS service instance."""
        self._ensure_started()
        return self._tts_service

    @property
    def rag_service(self):
        """RAG service instance."""
        self._ensure_started()
        return self._rag_service
