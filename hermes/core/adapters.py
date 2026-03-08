"""Service adapters — thin wrappers that add consistent timing, interrupt support,
timeout enforcement, and error recording around the raw service interfaces.

This layer sits between ``Call`` (pipeline logic) and the service backends
(STT, LLM, TTS, RAG), so ``Call`` never has to scatter timing or error
boilerplate across its tasks.

Architecture::

    Call._stt_task()  →  STTAdapter  →  AbstractSTTService
    Call._llm_task()  →  LLMAdapter  →  AbstractLLMService
    Call._tts_task()  →  TTSAdapter  →  AbstractTTSService
    Call._build_context() →  RAGAdapter  →  AbstractRAGService

Each adapter exposes the same method signatures as its underlying service,
so swapping a concrete backend has no impact on ``Call``.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

import structlog

from hermes.api.metrics import MetricsCollector

if TYPE_CHECKING:
    from hermes.services.llm.base import AbstractLLMService
    from hermes.services.rag.base import AbstractRAGService
    from hermes.services.stt.base import AbstractSTTService
    from hermes.services.tts.base import AbstractTTSService

logger = structlog.get_logger(__name__)

# Default timeout for RAG retrieval (seconds)
_RAG_TIMEOUT_S: float = 5.0


# ---------------------------------------------------------------------------
# Base helper
# ---------------------------------------------------------------------------


class _AdapterBase:
    """Shared helper for logging and error reporting."""

    def __init__(self, call_sid: str, interrupt_event: asyncio.Event) -> None:
        self._call_sid = call_sid
        self._interrupt_event = interrupt_event
        self._logger = structlog.get_logger(__name__).bind(call_sid=call_sid)

    @property
    def interrupted(self) -> bool:
        """Return ``True`` if a barge-in interrupt has been signalled."""
        return self._interrupt_event.is_set()

    @asynccontextmanager
    async def _timed(self, record_fn):  # type: ignore[no-untyped-def]
        """Context manager that calls *record_fn(elapsed)* on exit."""
        t0 = time.perf_counter()
        try:
            yield
        finally:
            record_fn(time.perf_counter() - t0)


# ---------------------------------------------------------------------------
# STT Adapter
# ---------------------------------------------------------------------------


class STTAdapter(_AdapterBase):
    """Wraps ``AbstractSTTService`` with per-transcript timing, interrupt
    support, and error recording."""

    def __init__(
        self,
        service: "AbstractSTTService",
        call_sid: str,
        interrupt_event: asyncio.Event,
    ) -> None:
        super().__init__(call_sid, interrupt_event)
        self._service = service

    async def stream_transcribe(
        self,
        tensor_queue: asyncio.Queue,
    ) -> AsyncGenerator[str, None]:
        """Stream transcripts, recording per-transcript latency.

        Stops yielding immediately when the interrupt event is set.
        """
        segment_start = time.perf_counter()
        try:
            async for transcript in self._service.stream_transcribe(tensor_queue):
                if self.interrupted:
                    self._logger.debug("stt_interrupted")
                    return
                if transcript.strip():
                    MetricsCollector.record_stt_latency(
                        time.perf_counter() - segment_start
                    )
                    yield transcript
                    segment_start = time.perf_counter()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            MetricsCollector.record_stt_error(type(exc).__name__)
            self._logger.error("stt_error", error=str(exc))
            raise


# ---------------------------------------------------------------------------
# LLM Adapter
# ---------------------------------------------------------------------------


class LLMAdapter(_AdapterBase):
    """Wraps ``AbstractLLMService`` with end-to-end latency timing, interrupt
    support, and error recording."""

    def __init__(
        self,
        service: "AbstractLLMService",
        call_sid: str,
        interrupt_event: asyncio.Event,
    ) -> None:
        super().__init__(call_sid, interrupt_event)
        self._service = service

    async def stream_sentences(
        self,
        prompt: str,
        context: str | None = None,
    ) -> AsyncGenerator[str, None]:
        """Stream LLM sentences, stopping on interrupt.

        Latency is recorded as total time from first call to last sentence.
        """
        t0 = time.perf_counter()
        sentences_yielded = 0
        try:
            async for chunk in self._service.stream_sentences(
                prompt=prompt,
                context=context,
                call_sid=self._call_sid,
            ):
                if self.interrupted:
                    self._logger.debug("llm_interrupted", sentences=sentences_yielded)
                    return
                sentence = str(chunk)
                sentences_yielded += 1
                yield sentence
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            MetricsCollector.record_llm_error(type(exc).__name__)
            self._logger.error("llm_error", error=str(exc))
            raise
        finally:
            if sentences_yielded:
                MetricsCollector.record_llm_latency(time.perf_counter() - t0)


# ---------------------------------------------------------------------------
# TTS Adapter
# ---------------------------------------------------------------------------


class TTSAdapter(_AdapterBase):
    """Wraps ``AbstractTTSService`` with per-sentence latency timing, interrupt
    support (chunk-level), and error recording."""

    def __init__(
        self,
        service: "AbstractTTSService",
        call_sid: str,
        interrupt_event: asyncio.Event,
    ) -> None:
        super().__init__(call_sid, interrupt_event)
        self._service = service

    @property
    def sample_rate(self) -> int:
        """Delegate sample rate to the underlying service."""
        return self._service.sample_rate

    async def generate_stream(self, text: str) -> AsyncGenerator[bytes, None]:
        """Stream PCM chunks for *text*, stopping immediately on interrupt.

        Latency is recorded per-sentence (from call to last chunk).
        """
        t0 = time.perf_counter()
        chunks_yielded = 0
        try:
            async for chunk_bytes in self._service.generate_stream(text):
                if self.interrupted:
                    self._logger.debug(
                        "tts_interrupted",
                        text=text[:60],
                        chunks=chunks_yielded,
                    )
                    return
                chunks_yielded += 1
                yield chunk_bytes
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            MetricsCollector.record_tts_error(type(exc).__name__)
            self._logger.error("tts_error", text=text[:60], error=str(exc))
            raise
        finally:
            if chunks_yielded:
                MetricsCollector.record_tts_latency(time.perf_counter() - t0)


# ---------------------------------------------------------------------------
# RAG Adapter
# ---------------------------------------------------------------------------


class RAGAdapter(_AdapterBase):
    """Wraps ``AbstractRAGService`` with latency timing, a unified timeout, and
    error recording. Always returns ``[]`` on timeout rather than raising."""

    def __init__(
        self,
        service: "AbstractRAGService | None",
        call_sid: str,
        interrupt_event: asyncio.Event,
        timeout_s: float = _RAG_TIMEOUT_S,
    ) -> None:
        super().__init__(call_sid, interrupt_event)
        self._service = service
        self._timeout_s = timeout_s

    async def retrieve(
        self,
        query: str,
        where: dict[str, Any] | None = None,
    ) -> list[str]:
        """Retrieve relevant documents for *query*.

        Returns ``[]`` if the service is unavailable, times out, or errors.
        Latency is recorded on success.
        """
        if self._service is None:
            return []

        t0 = time.perf_counter()
        try:
            results = await asyncio.wait_for(
                self._service.retrieve(query, where=where),
                timeout=self._timeout_s,
            )
            self._logger.debug(
                "rag_retrieved",
                query=query[:60],
                results=len(results),
                elapsed=round(time.perf_counter() - t0, 3),
            )
            return results

        except asyncio.TimeoutError:
            self._logger.warning(
                "rag_timeout",
                query=query[:60],
                timeout_s=self._timeout_s,
            )
            return []
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._logger.warning("rag_error", query=query[:60], error=str(exc))
            return []


# ---------------------------------------------------------------------------
# Bundle
# ---------------------------------------------------------------------------


class ServiceAdapters:
    """Bundle of all four service adapters for a single call.

    Build via :py:meth:`build` rather than constructing the adapters manually.
    """

    def __init__(
        self,
        stt: STTAdapter,
        llm: LLMAdapter,
        tts: TTSAdapter,
        rag: RAGAdapter,
    ) -> None:
        self.stt = stt
        self.llm = llm
        self.tts = tts
        self.rag = rag

    @classmethod
    def build(
        cls,
        *,
        call_sid: str,
        interrupt_event: asyncio.Event,
        stt_service: "AbstractSTTService | None",
        llm_service: "AbstractLLMService | None",
        tts_service: "AbstractTTSService | None",
        rag_service: "AbstractRAGService | None",
        rag_timeout_s: float = _RAG_TIMEOUT_S,
    ) -> "ServiceAdapters":
        """Construct all four adapters from the raw service references."""
        return cls(
            stt=STTAdapter(stt_service, call_sid, interrupt_event),  # type: ignore[arg-type]
            llm=LLMAdapter(llm_service, call_sid, interrupt_event),  # type: ignore[arg-type]
            tts=TTSAdapter(tts_service, call_sid, interrupt_event),  # type: ignore[arg-type]
            rag=RAGAdapter(rag_service, call_sid, interrupt_event, timeout_s=rag_timeout_s),
        )
