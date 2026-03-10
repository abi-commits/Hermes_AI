"""Call orchestration — creates, routes, and tears down voice calls."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import structlog

from hermes.api.metrics import MetricsCollector
from hermes.core.exceptions import CallError

if TYPE_CHECKING:
    from fastapi import WebSocket

    from hermes.core.call import Call
    from hermes.services.llm.base import AbstractLLMService
    from hermes.services.rag.base import AbstractRAGService
    from hermes.services.stt.base import AbstractSTTService
    from hermes.services.tts.base import AbstractTTSService

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Configuration data-classes
# ---------------------------------------------------------------------------


@dataclass
class ServiceBundle:
    """Services injected into every new call. STT is a factory (per-call WebSocket)."""

    stt_factory: Callable[[], "AbstractSTTService"]
    llm_service: "AbstractLLMService"
    tts_service: "AbstractTTSService"
    rag_service: "AbstractRAGService"


@dataclass(frozen=True)
class CallConfig:
    """Per-call configuration options."""

    persona: str = "default"
    greeting: str | None = None
    initial_prompt: str | None = None
    rag_metadata_filter: dict[str, Any] | None = None
    max_history: int = 20
    fallback_phrase: str = "I'm sorry, I had a problem. Could you repeat that?"

    language: str | None = None


# ---------------------------------------------------------------------------
# OrchestratorHooks protocol  (for testability / extension)
# ---------------------------------------------------------------------------


class OrchestratorHooks:
    """Optional lifecycle callbacks; all methods are no-ops by default."""

    async def on_call_started(self, call_sid: str) -> None:  # noqa: B027
        """Called after a call reaches LISTENING state."""

    async def on_call_ended(self, call_sid: str, duration_s: float) -> None:  # noqa: B027
        """Called after a call has fully ended."""

    async def on_interrupt(self, call_sid: str) -> None:  # noqa: B027
        """Called when a barge-in interrupt is applied."""

    async def on_error(self, call_sid: str, error: Exception) -> None:  # noqa: B027
        """Called when an unrecoverable error occurs inside a call."""


# ---------------------------------------------------------------------------
# CallOrchestrator
# ---------------------------------------------------------------------------


class CallOrchestrator:
    """Central coordinator for all active voice calls."""

    def __init__(
        self,
        bundle: ServiceBundle,
        hooks: OrchestratorHooks | None = None,
    ) -> None:
        self._bundle = bundle
        self._hooks = hooks or OrchestratorHooks()
        self._active_calls: dict[str, "Call"] = {}
        self._lock = asyncio.Lock()
        self._logger = structlog.get_logger(__name__)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def create_call(
        self,
        websocket: "WebSocket",
        call_sid: str,
        stream_sid: str,
        account_sid: str,
        config: CallConfig | None = None,
    ) -> "Call":
        """Create and start a new managed call; raises ``CallError`` if already active."""
        cfg = config or CallConfig()

        async with self._lock:
            if call_sid in self._active_calls:
                raise CallError(
                    f"Call {call_sid} is already active.",
                    error_code="DUPLICATE_CALL",
                )

        call = self._build_call(
            websocket=websocket,
            call_sid=call_sid,
            stream_sid=stream_sid,
            account_sid=account_sid,
            config=cfg,
        )

        async with self._lock:
            self._active_calls[call_sid] = call

        try:
            await call.start(greeting=cfg.greeting, initial_prompt=cfg.initial_prompt)
        except Exception as exc:
            async with self._lock:
                self._active_calls.pop(call_sid, None)
            await self._hooks.on_error(call_sid, exc)
            raise

        self._logger.info(
            "call_created",
            call_sid=call_sid,
            stream_sid=stream_sid,
            persona=cfg.persona,
            active_calls=len(self._active_calls),
        )
        await self._hooks.on_call_started(call_sid)
        return call

    async def interrupt_call(self, call_sid: str) -> bool:
        """Apply a barge-in interrupt; returns ``True`` if applied."""
        from hermes.models.call import CallState

        call = self.get_call(call_sid)
        if call is None:
            self._logger.warning("interrupt_call_not_found", call_sid=call_sid)
            return False

        if call.state not in (CallState.SPEAKING, CallState.PROCESSING):
            return False

        await call.interrupt()
        await self._hooks.on_interrupt(call_sid)
        self._logger.info("call_interrupted", call_sid=call_sid)
        MetricsCollector.record_call_interrupted(call_sid)
        return True

    async def terminate_call(
        self,
        call_sid: str,
        reason: str = "hangup",
    ) -> None:
        """Stop a call and remove it from the active registry."""
        async with self._lock:
            call = self._active_calls.pop(call_sid, None)

        if call is None:
            self._logger.debug("terminate_call_not_found", call_sid=call_sid)
            return

        status = "failed" if reason == "error" else "completed"

        try:
            await call.stop(status=status)
        except Exception as exc:
            self._logger.warning(
                "call_stop_error", call_sid=call_sid, error=str(exc)
            )

        duration = call.duration_seconds
        self._logger.info(
            "call_terminated",
            call_sid=call_sid,
            reason=reason,
            duration_s=round(duration, 2),
            active_calls=len(self._active_calls),
        )
        await self._hooks.on_call_ended(call_sid, duration)

    async def shutdown(self) -> None:
        """Gracefully terminate all active calls concurrently."""
        async with self._lock:
            call_sids = list(self._active_calls.keys())

        if not call_sids:
            return

        self._logger.info("orchestrator_shutdown", active_calls=len(call_sids))
        await asyncio.gather(
            *[self.terminate_call(sid, reason="server_shutdown") for sid in call_sids],
            return_exceptions=True,
        )
        self._logger.info("orchestrator_shutdown_complete")

    def get_call(self, call_sid: str) -> "Call | None":
        """Return the active ``Call`` for *call_sid*, or ``None``."""
        return self._active_calls.get(call_sid)

    @property
    def active_calls(self) -> dict[str, "Call"]:
        """Shallow copy of the active-call registry."""
        return dict(self._active_calls)

    @property
    def active_call_count(self) -> int:
        """Number of calls currently active."""
        return len(self._active_calls)

    # ------------------------------------------------------------------
    # Error recovery
    # ------------------------------------------------------------------

    async def handle_call_error(
        self,
        call_sid: str,
        error: Exception,
        *,
        attempt_recovery: bool = False,
    ) -> None:
        """Log an unrecoverable call error and terminate the affected call."""
        call = self.get_call(call_sid)
        if call is None:
            return

        self._logger.error(
            "call_error_recovery",
            call_sid=call_sid,
            error_type=type(error).__name__,
            error=str(error),
        )
        await self._hooks.on_error(call_sid, error)

        if attempt_recovery and call._running:  # noqa: SLF001
            self._logger.info(
                "call_error_recovery_skipped",
                call_sid=call_sid,
                reason="background_task_failure_requires_termination",
            )

        await self.terminate_call(call_sid, reason="error")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_call(
        self,
        websocket: "WebSocket",
        call_sid: str,
        stream_sid: str,
        account_sid: str,
        config: CallConfig,
    ) -> "Call":
        """Construct a ``Call`` with injected services (not yet started)."""
        from hermes.core.call import Call

        # STT is per-call (stateful WebSocket connection)
        stt_instance = self._bundle.stt_factory()

        return Call(
            call_sid=call_sid,
            stream_sid=stream_sid,
            websocket=websocket,
            account_sid=account_sid,
            stt_service=stt_instance,
            llm_service=self._bundle.llm_service,
            tts_service=self._bundle.tts_service,
            rag_service=self._bundle.rag_service,
            task_error_handler=self.handle_call_error,
            persona=config.persona,
            rag_metadata_filter=config.rag_metadata_filter,
            max_history=config.max_history,
            fallback_phrase=config.fallback_phrase,
        )
