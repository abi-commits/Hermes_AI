"""Modal-backed remote TTS adapter for Hermes."""

from __future__ import annotations

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
        self._logger = structlog.get_logger(__name__)

    def _lookup_remote_cls(self) -> Any:
        """Resolve the Modal class handle lazily."""
        if self._remote_cls is not None:
            return self._remote_cls

        try:
            import modal
        except ImportError as exc:
            raise ServiceUnavailableError(
                "ModalRemoteTTS",
                "modal package is not installed",
            ) from exc

        try:
            self._remote_cls = modal.Cls.lookup(self.app_name, self.class_name)
            return self._remote_cls
        except Exception as exc:
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
        remote_cls = self._lookup_remote_cls()
        try:
            instance = remote_cls()
        except TypeError:
            instance = remote_cls
        return await self._maybe_await(instance)

    async def _iter_remote_stream(self, stream: Any) -> AsyncIterator[bytes]:
        """Normalize sync or async remote generators into an async byte stream."""
        if hasattr(stream, "__aiter__"):
            async for chunk in stream:
                yield bytes(chunk)
            return

        if isinstance(stream, Iterable):
            for chunk in stream:
                yield bytes(chunk)
            return

        raise TTSGenerationError("Modal TTS stream did not return an iterable generator")

    async def generate_stream(
        self,
        text: str,
        audio_prompt_path: str | Path | None = None,
        embed_watermark: bool = True,
        chunk_size: int | None = None,
    ) -> AsyncIterator[bytes]:
        """Stream native-rate PCM bytes from the remote Modal TTS worker."""
        instance = await self._get_remote_instance()
        prompt = str(audio_prompt_path) if audio_prompt_path else None
        effective_chunk_size = chunk_size if chunk_size is not None else self._default_chunk_size

        try:
            remote_stream = instance.generate_stream.remote_gen.aio(
                text=text,
                audio_prompt_path=prompt,
                embed_watermark=embed_watermark
                if embed_watermark is not None
                else self._default_embed_watermark,
                chunk_size=effective_chunk_size,
            )
            async for chunk in self._iter_remote_stream(remote_stream):
                yield chunk
        except Exception as exc:
            self._logger.error(
                "modal_remote_tts_stream_failed",
                app_name=self.app_name,
                class_name=self.class_name,
                error=str(exc),
            )
            raise TTSGenerationError(f"Remote Modal TTS streaming failed: {exc}") from exc

    async def generate(
        self,
        text: str,
        audio_prompt_path: str | Path | None = None,
        embed_watermark: bool = True,
    ) -> bytes:
        """Request full native-rate PCM audio from the remote Modal TTS worker."""
        instance = await self._get_remote_instance()
        prompt = str(audio_prompt_path) if audio_prompt_path else None

        try:
            result = await instance.generate.remote.aio(
                text=text,
                audio_prompt_path=prompt,
                embed_watermark=embed_watermark
                if embed_watermark is not None
                else self._default_embed_watermark,
            )
            return bytes(await self._maybe_await(result))
        except Exception as exc:
            self._logger.error(
                "modal_remote_tts_generate_failed",
                app_name=self.app_name,
                class_name=self.class_name,
                error=str(exc),
            )
            raise TTSGenerationError(f"Remote Modal TTS generation failed: {exc}") from exc

    def set_executor(self, executor: ThreadPoolExecutor) -> None:
        """No-op hook to satisfy the Hermes TTS service interface."""
        self._executor = executor

    async def ping(self) -> bool:
        """Ping the remote worker to verify it is responsive."""
        try:
            instance = await self._get_remote_instance()
            # Making a fast, lightweight remote call
            await self._maybe_await(instance.get_sample_rate.remote.aio())
            return True
        except Exception as exc:
            self._logger.warning("modal_remote_tts_ping_failed", error=str(exc))
            return False

    @property
    def sample_rate(self) -> int:
        """Native sample rate of the remote worker output."""
        return self._sample_rate
