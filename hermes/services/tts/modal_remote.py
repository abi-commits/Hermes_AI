"""Modal-backed remote TTS adapter for Hermes."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import AsyncIterator
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
        self._remote_instance: Any = None
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
                return modal.Cls.from_name(self.app_name, self.class_name)

            self._remote_cls = await loop.run_in_executor(None, _do_lookup)
            self._logger.info("modal_lookup_success")
            return self._remote_cls
        except Exception as exc:
            self._logger.error("modal_lookup_failed", error=str(exc))
            raise TTSGenerationError(
                f"Failed to look up Modal TTS class {self.app_name}.{self.class_name}: {exc}"
            ) from exc

    async def _get_remote_instance(self) -> Any:
        """Fetch or create the remote Modal class instance (cached)."""
        if self._remote_instance is not None:
            return self._remote_instance

        remote_cls = await self._lookup_remote_cls()
        # Instantiate once and cache
        self._remote_instance = remote_cls()
        return self._remote_instance

    async def _call_remote_gen(self, method: Any, **kwargs: Any) -> AsyncIterator[bytes]:
        """Invoke a generator Modal method using the definitive aio pattern."""
        if not hasattr(method, "remote_gen"):
            raise TTSGenerationError(f"Method {method} does not expose .remote_gen")

        try:
            async for chunk in method.remote_gen.aio(**kwargs):
                yield bytes(chunk)
        except Exception as e:
            self._logger.error("modal_remote_gen_failed", error=str(e))
            raise

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

        try:
            chunk_count = 0
            async for chunk in self._call_remote_gen(
                instance.generate_stream,
                text=text,
                audio_prompt_path=prompt,
                embed_watermark=embed_watermark if embed_watermark is not None else self._default_embed_watermark,
                chunk_size=effective_chunk_size,
            ):
                if chunk_count == 0:
                    self._logger.info("tts_first_chunk_received", text_len=len(text))
                
                chunk_count += 1
                yield chunk

            self._logger.info("remote_synthesis_stream_complete", total_chunks=chunk_count)
        except Exception as exc:
            self._logger.error("modal_remote_tts_stream_failed", error=str(exc))
            raise TTSGenerationError(f"Remote Modal TTS streaming failed: {exc}") from exc

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
            # Consistent aio pattern for unary calls
            result = await instance.generate.aio(
                text=text,
                audio_prompt_path=prompt,
                embed_watermark=embed_watermark if embed_watermark is not None else self._default_embed_watermark,
            )
            return bytes(result)
        except Exception as exc:
            self._logger.error("modal_remote_tts_generate_failed", error=str(exc))
            raise TTSGenerationError(f"Remote Modal TTS generation failed: {exc}") from exc

    def set_executor(self, executor: ThreadPoolExecutor) -> None:
        """No-op. Remote service doesn't require a local executor, but satisfies interface."""
        self._executor = executor

    async def ping(self) -> bool:
        """Ping the remote worker to verify it is responsive."""
        try:
            instance = await self._get_remote_instance()
            # Consistent aio pattern
            await instance.get_sample_rate.aio()
            return True
        except Exception:
            return False

    @property
    def sample_rate(self) -> int:
        """Native sample rate of the remote worker output."""
        return self._sample_rate
