"""Service container for dependency injection.

Provides a centralized registry for service instances with proper
lifecycle management. Services are created once and shared across
the application rather than being instantiated per-call.
"""

from __future__ import annotations

import httpx
import structlog

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

        # Import here to avoid circular imports
        from hermes.services.llm import LLMService
        from hermes.services.rag import RAGService
        from hermes.services.stt import STTService
        from hermes.services.tts import TTSService
        from hermes.services.vector_db import VectorDB

        # Vector DB (needed by RAG)
        self._vector_db = VectorDB(http_client=self._http_client)

        # Individual services
        self._stt_service = STTService(http_client=self._http_client)
        self._llm_service = LLMService(http_client=self._http_client)
        self._tts_service = TTSService(http_client=self._http_client)
        self._rag_service = RAGService(
            vector_db=self._vector_db,
            http_client=self._http_client,
        )

        self._started = True
        logger.info("service_container_started")

    async def stop(self) -> None:
        """Shut down all services and release resources."""
        if not self._started:
            return

        logger.info("service_container_stopping")

        if self._vector_db:
            await self._vector_db.disconnect()

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
        """Speech-to-Text service instance."""
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

    @property
    def vector_db(self):
        """Vector database instance."""
        self._ensure_started()
        return self._vector_db
