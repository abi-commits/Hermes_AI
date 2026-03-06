"""Chatterbox Streaming TTS service implementation."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import structlog
import torch

from hermes.core.exceptions import TTSGenerationError
from hermes.services.tts.base import AbstractTTSService

logger = structlog.get_logger(__name__)

# Default number of speech tokens synthesised per streaming chunk.
_DEFAULT_CHUNK_SIZE: int = 50


class ChatterboxTTSService(AbstractTTSService):
    """TTS service backed by Chatterbox Streaming with optional Perth watermarking."""

    def __init__(
        self,
        model_name: str = "chatterbox-streaming",
        device: str | None = None,
        watermark_key: bytes | None = None,
        num_workers: int = 1,
        chunk_size: int = _DEFAULT_CHUNK_SIZE,
    ) -> None:
        """Load the Chatterbox model."""
        self.model_name = model_name
        _auto = not device or device == "auto"
        self.device = (
            (
                "cuda"
                if torch.cuda.is_available()
                else "mps"
                if torch.backends.mps.is_available()
                else "cpu"
            )
            if _auto
            else device
        )
        self.watermark_key = watermark_key
        self.num_workers = num_workers
        self.chunk_size = chunk_size
        self._logger = structlog.get_logger(__name__)

        self._model = None
        self._load_model()

        self._executor: ThreadPoolExecutor | None = None

        self._watermarker = None
        if watermark_key:
            try:
                import perth

                self._watermarker = perth.PerthImplicitWatermarker()
            except ImportError:
                self._logger.warning(
                    "perth_not_installed",
                    hint="Install 'perth' to enable audio watermarking",
                )

        self._logger.info(
            "chatterbox_tts_initialised",
            device=self.device,
            watermark_enabled=self._watermarker is not None,
            chunk_size=self.chunk_size,
        )

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        """Load the Chatterbox Streaming model (called once at init)."""
        try:
            from chatterbox.tts import ChatterboxTTS

            self._model = ChatterboxTTS.from_pretrained(device=self.device)
            self._logger.info("chatterbox_model_loaded", device=self.device)
        except Exception as exc:
            self._logger.error("chatterbox_model_load_failed", error=str(exc))
            raise TTSGenerationError(f"Model loading failed: {exc}") from exc

    # ------------------------------------------------------------------
    # Streaming synthesis
    # ------------------------------------------------------------------

    async def generate_stream(
        self,
        text: str,
        audio_prompt_path: str | Path | None = None,
        embed_watermark: bool = True,
        chunk_size: int | None = None,
        cancel_event: threading.Event | None = None,
    ) -> AsyncIterator[bytes]:
        """Synthesise *text* and yield 16-bit PCM chunks; raises ``TTSGenerationError`` on failure.

        Args:
            cancel_event: When set by the caller (e.g. on barge-in), the
                background synthesis thread will stop between chunks, preventing
                wasted CPU/GPU work after the consumer has already moved on.
        """
        if self._model is None:
            raise TTSGenerationError("Chatterbox model is not loaded")

        _chunk = chunk_size if chunk_size is not None else self.chunk_size
        prompt = str(audio_prompt_path) if audio_prompt_path else None
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[torch.Tensor | None | Exception] = asyncio.Queue()
        model = self._model

        def _run_stream() -> None:
            try:
                with torch.no_grad():
                    for audio_chunk, _metrics in model.generate_stream(
                        text,
                        audio_prompt_path=prompt,
                        chunk_size=_chunk,
                        print_metrics=False,
                    ):
                        # Stop early if the consumer signalled cancellation
                        # (e.g. barge-in interrupt) to avoid wasted synthesis.
                        if cancel_event is not None and cancel_event.is_set():
                            break
                        loop.call_soon_threadsafe(queue.put_nowait, audio_chunk)
                loop.call_soon_threadsafe(queue.put_nowait, None)  # sentinel
            except Exception as exc:
                loop.call_soon_threadsafe(queue.put_nowait, exc)

        future = loop.run_in_executor(self._executor, _run_stream)
        chunks_yielded = 0

        try:
            while True:
                item = await queue.get()
                if item is None:
                    break
                if isinstance(item, Exception):
                    raise item

                wav: torch.Tensor = item
                audio_np = wav.cpu().numpy().squeeze()
                if audio_np.ndim == 0 or audio_np.size == 0:
                    continue

                if embed_watermark and self._watermarker and self._model:
                    try:
                        audio_np = self._watermarker.apply_watermark(
                            audio_np.astype(np.float32),
                            sample_rate=self._model.sr,
                        )
                    except Exception as exc:
                        self._logger.warning("watermark_embed_failed", error=str(exc))

                yield (audio_np * 32_767).astype(np.int16).tobytes()
                chunks_yielded += 1

        except Exception as exc:
            self._logger.exception("tts_stream_failed", text_preview=text[:50])
            raise TTSGenerationError(f"Streaming synthesis error: {exc}") from exc
        finally:
            await asyncio.shield(future)

        self._logger.debug(
            "tts_stream_complete", chunks=chunks_yielded, text_preview=text[:50]
        )

    # ------------------------------------------------------------------
    # Full-audio synthesis
    # ------------------------------------------------------------------

    async def generate(
        self,
        text: str,
        audio_prompt_path: str | Path | None = None,
        embed_watermark: bool = True,
    ) -> bytes:
        """Synthesise *text* and return the complete audio in a single buffer."""
        chunks: list[bytes] = []
        async for chunk in self.generate_stream(text, audio_prompt_path, embed_watermark):
            chunks.append(chunk)
        return b"".join(chunks)

    # ------------------------------------------------------------------
    # Executor management
    # ------------------------------------------------------------------

    def set_executor(self, executor: ThreadPoolExecutor) -> None:
        """Attach a shared thread-pool executor for synthesis work."""
        self._executor = executor

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def sample_rate(self) -> int:
        """Native output sample rate of the loaded model in Hz."""
        if self._model is not None:
            return self._model.sr
        return 24_000  # Chatterbox default
