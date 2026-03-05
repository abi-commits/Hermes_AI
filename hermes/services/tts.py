"""TTS Service Module for Hermes.

Chatterbox Turbo is a state-of-the-art open-source TTS model from Resemble AI
that delivers near-human speech quality with low latency.
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import structlog
import torch
import torchaudio

from hermes.core.exceptions import TTSGenerationError

logger = structlog.get_logger(__name__)


class ChatterboxTTSService:
    """Service class for Chatterbox Turbo TTS.

    Loads the model once and provides an async ``generate`` method that runs
    the CPU/GPU-bound synthesis in a thread pool to avoid blocking the event loop.
    """

    def __init__(
        self,
        model_name: str = "chatterbox-turbo",
        device: str | None = None,
        watermark_key: bytes | None = None,
        num_workers: int = 1,
    ) -> None:
        """Initialize the TTS service.
        """
        self.model_name = model_name
        self.device = device or (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        self.watermark_key = watermark_key
        self.num_workers = num_workers
        self._logger = structlog.get_logger(__name__)

        # Load the model (blocking, but done at startup)
        self._model = None
        self._load_model()

        # Thread pool executor for offloading generate calls
        self._executor: ThreadPoolExecutor | None = None  # set via set_executor / lifespan

        # Perth watermarker instance
        self._watermarker = None
        if watermark_key:
            try:
                import perth  # Resemble's Perth watermarker

                self._watermarker = perth.PerthImplicitWatermarker(secret_key=watermark_key)
            except ImportError:
                self._logger.warning(
                    "perth_not_installed",
                    msg="Perth watermarking requested but 'perth' package is not installed",
                )

        self._logger.info(
            "tts_service_initialized",
            device=self.device,
            model_name=self.model_name,
            watermark_enabled=self._watermarker is not None,
        )

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        """Load the Chatterbox Turbo model (synchronous)."""
        try:
            from chatterbox.tts_turbo import ChatterboxTurboTTS

            self._model = ChatterboxTurboTTS.from_pretrained(device=self.device)
            self._logger.info("chatterbox_turbo_model_loaded", device=self.device)
        except Exception as exc:
            self._logger.error("chatterbox_turbo_model_load_failed", error=str(exc))
            raise TTSGenerationError(f"Model loading failed: {exc}") from exc

    # ------------------------------------------------------------------
    # Public async API
    # ------------------------------------------------------------------

    async def generate(
        self,
        text: str,
        audio_prompt_path: str | Path | None = None,
        embed_watermark: bool = True,
    ) -> bytes:
        """Generate speech audio from text.

        Args:
            text: Input text to synthesize.
            audio_prompt_path: Path to a reference WAV file for voice cloning.
            embed_watermark: Whether to embed a Perth watermark (if service configured).

        Returns:
            Raw audio bytes in 16-bit PCM format at the model's native sample rate.
        """
        if not self._model:
            raise TTSGenerationError("Model not loaded")

        loop = asyncio.get_running_loop()
        try:
            wav_tensor = await loop.run_in_executor(
                self._executor,
                self._synthesize,
                text,
                audio_prompt_path,
            )
        except Exception as exc:
            self._logger.exception(
                "tts_generation_failed",
                text_preview=text[:50],
                error=str(exc),
            )
            raise TTSGenerationError(f"Synthesis error: {exc}") from exc

        # Convert tensor to 16-bit PCM bytes
        audio_np = wav_tensor.cpu().numpy().squeeze()
        audio_int16 = (audio_np * 32767).astype(np.int16)
        audio_bytes = audio_int16.tobytes()

        # Embed watermark if requested and configured
        if embed_watermark and self._watermarker:
            try:
                watermarked = self._watermarker.embed_watermark(
                    audio_np.astype(np.float32),
                    sample_rate=self._model.sr,
                )
                audio_int16 = (watermarked * 32767).astype(np.int16)
                audio_bytes = audio_int16.tobytes()
                self._logger.debug("watermark_embedded")
            except Exception as exc:
                self._logger.warning("watermark_embed_failed", error=str(exc))

        return audio_bytes

    # ------------------------------------------------------------------
    # Synchronous synthesis (runs in thread pool)
    # ------------------------------------------------------------------

    def _synthesize(
        self,
        text: str,
        audio_prompt_path: str | Path | None,
    ) -> torch.Tensor:
        """Synchronous synthesis call dispatched via the thread-pool executor.

        Returns:
            Tensor of shape ``(1, T)`` at ``model.sr``.
        """
        prompt = str(audio_prompt_path) if audio_prompt_path else None
        with torch.no_grad():
            wav = self._model.generate(text, audio_prompt_path=prompt)
        return wav

    # ------------------------------------------------------------------
    # Audio format helpers
    # ------------------------------------------------------------------

    def resample_to_8khz(self, audio_bytes: bytes, orig_sr: int) -> bytes:
        """Resample 16-bit PCM audio to 8 kHz.

        Args:
            audio_bytes: Raw 16-bit PCM bytes at *orig_sr*.
            orig_sr: Original sample rate in Hz.

        Returns:
            16-bit PCM bytes resampled to 8 000 Hz.
        """
        audio_tensor = torch.from_numpy(
            np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32767.0
        ).unsqueeze(0)
        resampler = torchaudio.transforms.Resample(orig_sr, 8000)
        resampled = resampler(audio_tensor)
        resampled_int16 = (resampled.squeeze().numpy() * 32767).astype(np.int16)
        return resampled_int16.tobytes()

    @staticmethod
    def convert_to_ulaw(pcm16_bytes: bytes) -> bytes:
        """Convert 16-bit PCM audio to 8-bit µ-law (Twilio format).

        Args:
            pcm16_bytes: Raw 16-bit PCM bytes.

        Returns:
            µ-law encoded bytes.
        """
        audio_np = np.frombuffer(pcm16_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        tensor = torch.from_numpy(audio_np)
        encoded = torchaudio.functional.mu_law_encoding(tensor, quantization_channels=256)
        return encoded.to(torch.uint8).numpy().tobytes()

    # ------------------------------------------------------------------
    # Executor management
    # ------------------------------------------------------------------

    def set_executor(self, executor: ThreadPoolExecutor) -> None:
        """Set the thread pool executor (called during app startup).

        Args:
            executor: ``ThreadPoolExecutor`` to use for synthesis calls.
        """
        self._executor = executor

    @property
    def sample_rate(self) -> int:
        """Native sample rate of the loaded model."""
        if self._model is not None:
            return self._model.sr
        return 24_000  # Chatterbox Turbo default


# ---------------------------------------------------------------------------
# Worker pool for higher concurrency
# ---------------------------------------------------------------------------

class TTSWorkerPool:
    """Manages a pool of ``ChatterboxTTSService`` instances.

    Each worker runs its own model (on the same or different devices) and jobs
    are dispatched via round-robin.
    """

    def __init__(
        self,
        num_workers: int = 2,
        device_ids: list[str] | None = None,
        watermark_key: bytes | None = None,
    ) -> None:
        """Initialize the worker pool.

        Args:
            num_workers: Number of TTS service instances.
            device_ids: List of devices (e.g. ``['cuda:0', 'cuda:1']``). Auto-detect if *None*.
            watermark_key: Shared watermark key.
        """
        self.workers: list[ChatterboxTTSService] = []
        self._next_worker = 0
        self._logger = structlog.get_logger(__name__)

        for i in range(num_workers):
            device = device_ids[i] if device_ids and i < len(device_ids) else None
            worker = ChatterboxTTSService(
                device=device,
                watermark_key=watermark_key,
                num_workers=1,
            )
            self.workers.append(worker)

        self._logger.info("tts_worker_pool_initialized", num_workers=num_workers)

    async def submit(
        self,
        call_sid: str,
        turn_id: int,
        text: str,
        audio_prompt_path: str | Path | None = None,
    ) -> "asyncio.Future[bytes]":
        """Submit a TTS job to the pool.

        Uses round-robin to select a worker and returns an ``asyncio.Future``
        that resolves to the raw audio bytes.

        Args:
            call_sid: Call session identifier.
            turn_id: Conversation turn identifier.
            text: Text to synthesize.
            audio_prompt_path: Optional reference audio for voice cloning.

        Returns:
            Future that resolves to 16-bit PCM bytes.
        """
        worker = self.workers[self._next_worker % len(self.workers)]
        self._next_worker += 1

        future: asyncio.Future[bytes] = asyncio.get_running_loop().create_future()
        asyncio.create_task(
            self._run_job(worker, call_sid, turn_id, text, audio_prompt_path, future)
        )
        return future

    async def _run_job(
        self,
        worker: ChatterboxTTSService,
        call_sid: str,
        turn_id: int,
        text: str,
        audio_prompt_path: str | Path | None,
        future: "asyncio.Future[bytes]",
    ) -> None:
        """Execute a single TTS job and resolve the future."""
        try:
            audio_bytes = await worker.generate(text, audio_prompt_path)
            if not future.cancelled():
                future.set_result(audio_bytes)
        except Exception as exc:
            if not future.cancelled():
                future.set_exception(exc)

    async def cancel_jobs_for_call(self, call_sid: str) -> None:
        """Cancel all pending jobs for *call_sid*.

        .. note:: Full per-call tracking is not yet implemented.
        """
        self._logger.debug("cancel_jobs_requested", call_sid=call_sid)

    def set_executor(self, executor: ThreadPoolExecutor) -> None:
        """Propagate a thread-pool executor to all workers.

        Args:
            executor: ``ThreadPoolExecutor`` to use.
        """
        for w in self.workers:
            w.set_executor(executor)


# ---------------------------------------------------------------------------
# Mock service for tests
# ---------------------------------------------------------------------------

class MockTTSService(ChatterboxTTSService):
    """Mock TTS service that returns a deterministic sine wave.

    Bypasses model loading entirely and is safe to instantiate without
    a GPU or the ``chatterbox`` package.
    """

    def __init__(self, duration_seconds: float = 1.0) -> None:
        """Initialize mock service.

        Args:
            duration_seconds: Duration of mock audio in seconds.
        """
        # Intentionally skip ChatterboxTTSService.__init__ to avoid model load
        self.duration_seconds = duration_seconds
        self.model_name = "mock"
        self.device = "cpu"
        self.watermark_key = None
        self.num_workers = 1
        self._model = None
        self._executor = None
        self._watermarker = None
        self._logger = structlog.get_logger(__name__)

    async def generate(
        self,
        text: str,
        audio_prompt_path: str | Path | None = None,
        embed_watermark: bool = False,
    ) -> bytes:
        """Return mock audio (440 Hz sine wave) as 16-bit PCM bytes."""
        sr = 16_000
        duration_samples = int(self.duration_seconds * sr)
        t = np.linspace(0, self.duration_seconds, duration_samples)
        audio = np.sin(2 * np.pi * 440 * t).astype(np.float32) * 0.3
        audio_int16 = (audio * 32767).astype(np.int16)

        self._logger.debug("mock_tts_generated", text=text[:50])
        return audio_int16.tobytes()

    @property
    def sample_rate(self) -> int:  # noqa: D102
        return 16_000
