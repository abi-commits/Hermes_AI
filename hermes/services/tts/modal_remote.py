"""Modal-backed remote TTS adapter for Hermes."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import AsyncIterator, Iterable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import structlog

from hermes.core.exceptions import ServiceUnavailableError, TTSGenerationError
from hermes.services.tts.base import AbstractTTSService

logger = structlog.get_logger(__name__)


class ModalRemoteTTSService(AbstractTTSService):
    """Adapter that makes a remote Modal TTS worker look like a local Hermes TTS service."""

    def __init__(
        self,
        app_name: str,
        class_name: str,
        sample_rate: int = 24_000,
        default_chunk_size: int = 50,
        default_embed_watermark: bool = False,
    ) -> None:
        self.app_name = app_name
        self.class_name = class_name
        self._sample_rate = sample_rate
        self._default_chunk_size = default_chunk_size
        self._default_embed_watermark = default_embed_watermark
        self._executor: ThreadPoolExecutor | None = None
        self._remote_cls: Any = None
        self._logger = structlog.get_logger(__name__).bind(
            remote_app=app_name,
            remote_class=class_name
        )

    async def _lookup_remote_cls(self) -> Any:
        """Resolve the Modal class handle lazily and safely."""
        if self._remote_cls is not None:
            return self._remote_cls

        try:
            import modal
        except ImportError as exc:
            raise ServiceUnavailableError(
                "ModalRemoteTTS",
                "modal package is not installed",
            ) from exc

        self._logger.info("modal_lookup_starting")
        
        try:
            # Run the lookup in an executor to avoid blocking the event loop
            loop = asyncio.get_running_loop()
            
            def _do_lookup():
                # Correct for newer Modal SDKs
                return modal.Cls.from_name(self.app_name, self.class_name)

            self._remote_cls = await loop.run_in_executor(None, _do_lookup)
            self._logger.info("modal_lookup_success")
            return self._remote_cls
        except Exception as exc:
            self._logger.error("modal_lookup_failed", error=str(exc))
            raise TTSGenerationError(
                f"Failed to look up Modal TTS class {self.app_name}.{self.class_name}: {exc}"
            ) from exc

    async def _maybe_await(self, value: Any) -> Any:
        """Await *value* if it is awaitable; otherwise return it as-is."""
        if inspect.isawaitable(value):
            return await value
        return value

    async def _get_remote_instance(self) -> Any:
        """Create or fetch the remote Modal class instance."""
        remote_cls = await self._lookup_remote_cls()
        try:
            instance = remote_cls()
        except TypeError:
            instance = remote_cls
        return await self._maybe_await(instance)

    async def _call_remote(self, method: Any, **kwargs: Any) -> Any:
        """Invoke a non-generator Modal method across client API variants."""
        remote = getattr(method, "remote", None)
        if remote is None:
            raise TTSGenerationError("Modal remote method does not expose .remote()")

        if hasattr(remote, "aio"):
            return await self._maybe_await(remote.aio(**kwargs))

        return await self._maybe_await(remote(**kwargs))

    async def _iter_remote_stream(self, stream: Any) -> AsyncIterator[bytes]:
        """Normalize sync or async Modal generators into a byte stream."""
        self._logger.debug("_iter_remote_stream_entered", stream_type=type(stream).__name__)

        if hasattr(stream, "__aiter__"):
            self._logger.debug("stream_is_async_iterator")
            chunk_count = 0
            async for chunk in stream:
                chunk_count += 1
                if chunk_count <= 3:
                    self._logger.debug("yielding_async_chunk", idx=chunk_count, size=len(bytes(chunk)))
                yield bytes(chunk)
            self._logger.debug("async_iterator_complete", total_chunks=chunk_count)
            return

        if isinstance(stream, (bytes, bytearray, memoryview)):
            self._logger.debug("stream_is_bytes_buffer", size=len(bytes(stream)))
            yield bytes(stream)
            return

        if isinstance(stream, Iterable):
            self._logger.debug("stream_is_sync_iterable")
            chunk_count = 0
            for chunk in stream:
                chunk_count += 1
                if chunk_count <= 3:
                    self._logger.debug("yielding_sync_chunk", idx=chunk_count, size=len(bytes(chunk)))
                yield bytes(chunk)
            self._logger.debug("sync_iterator_complete", total_chunks=chunk_count)
            return

        raise TTSGenerationError(f"Modal TTS stream did not return an iterable generator (got {type(stream).__name__})")

    async def _call_remote_gen(self, method: Any, **kwargs: Any) -> AsyncIterator[bytes]:
        """Invoke a generator Modal method across client API variants."""
        remote_gen = getattr(method, "remote_gen", None)
        if remote_gen is None:
            raise TTSGenerationError("Modal remote generator does not expose .remote_gen()")

        if hasattr(remote_gen, "aio"):
            stream = await self._maybe_await(remote_gen.aio(**kwargs))
        else:
            stream = await self._maybe_await(remote_gen(**kwargs))

        async for chunk in self._iter_remote_stream(stream):
            yield chunk

    def generate_stream(
        self,
        text: str,
        audio_prompt_path: str | Path | None = None,
        embed_watermark: bool = True,
        chunk_size: int | None = None,
    ) -> AsyncIterator[bytes]:
        """Stream native-rate PCM bytes from the remote Modal TTS worker."""
        return self._do_generate_stream(
            text=text,
            audio_prompt_path=audio_prompt_path,
            embed_watermark=embed_watermark,
            chunk_size=chunk_size
        )

    async def _do_generate_stream(
        self,
        text: str,
        audio_prompt_path: str | Path | None = None,
        embed_watermark: bool = True,
        chunk_size: int | None = None,
    ) -> AsyncIterator[bytes]:
        """Internal async generator for streaming."""
        self._logger.info("remote_synthesis_stream_requested", text_len=len(text), chunk_size=chunk_size)

        instance = await self._get_remote_instance()
        prompt = str(audio_prompt_path) if audio_prompt_path else None
        effective_chunk_size = chunk_size if chunk_size is not None else self._default_chunk_size

        self._logger.debug("got_remote_instance", instance_type=type(instance).__name__)

        try:
            self._logger.debug("calling_remote_generator", method="generate_stream")
            chunk_count = 0
            async for chunk in self._call_remote_gen(
                instance.generate_stream,
                text=text,
                audio_prompt_path=prompt,
                embed_watermark=embed_watermark
                if embed_watermark is not None
                else self._default_embed_watermark,
                chunk_size=effective_chunk_size,
            ):
                chunk_count += 1
                if chunk_count <= 3:
                    self._logger.debug("yielding_chunk_from_remote", idx=chunk_count, size=len(bytes(chunk)))
                yield bytes(chunk)

            self._logger.info("remote_synthesis_stream_complete", total_chunks=chunk_count)
        except Exception as exc:
            self._logger.error("modal_remote_tts_stream_failed", error=str(exc), error_type=type(exc).__name__)
            raise TTSGenerationError(f"Remote Modal TTS streaming failed: {exc}") from exc

    async def generate(
        self,
        text: str,
        audio_prompt_path: str | Path | None = None,
        embed_watermark: bool = True,
    ) -> bytes:
        """Request full native-rate PCM audio from the remote Modal TTS worker."""
        self._logger.info("remote_synthesis_full_requested", text_len=len(text))
        
        instance = await self._get_remote_instance()
        prompt = str(audio_prompt_path) if audio_prompt_path else None

        try:
            result = await self._call_remote(
                instance.generate,
                text=text,
                audio_prompt_path=prompt,
                embed_watermark=embed_watermark
                if embed_watermark is not None
                else self._default_embed_watermark,
            )
            return bytes(await self._maybe_await(result))
        except Exception as exc:
            self._logger.error("modal_remote_tts_generate_failed", error=str(exc))
            raise TTSGenerationError(f"Remote Modal TTS generation failed: {exc}") from exc

    def set_executor(self, executor: ThreadPoolExecutor) -> None:
        """No-op hook to satisfy the Hermes TTS service interface."""
        self._executor = executor

    async def ping(self) -> bool:
        """Ping the remote worker to verify it is responsive."""
        try:
            instance = await self._get_remote_instance()
            await self._call_remote(instance.get_sample_rate)
            return True
        except Exception as exc:
            self._logger.warning("modal_remote_tts_ping_failed", error=str(exc))
            return False

    @property
    def sample_rate(self) -> int:
        """Native sample rate of the remote worker output."""
        return self._sample_rate
