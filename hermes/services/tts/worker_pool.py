"""Round-robin worker pool for concurrent TTS generation."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import structlog

from hermes.services.tts.chatterbox import ChatterboxTTSService

logger = structlog.get_logger(__name__)


class TTSWorkerPool:
    """Round-robin pool of ``ChatterboxTTSService`` instances for concurrent synthesis."""

    def __init__(
        self,
        num_workers: int = 2,
        device_ids: list[str] | None = None,
        watermark_key: bytes | None = None,
    ) -> None:
        """Initialise the worker pool and load all models."""
        self.workers: list[ChatterboxTTSService] = []
        self._next_worker = 0
        self._logger = structlog.get_logger(__name__)

        for i in range(num_workers):
            device = device_ids[i] if device_ids and i < len(device_ids) else None
            self.workers.append(
                ChatterboxTTSService(
                    device=device,
                    watermark_key=watermark_key,
                )
            )

        # Per-call tracking for cancellation.
        # _call_tasks:        call_sid → set of in-flight full-audio asyncio Tasks
        # _call_stream_flags: call_sid → list of threading.Events for streaming jobs
        # All accessed from the asyncio event loop (single thread), so no extra
        # locking is required beyond what the event loop already provides.
        self._call_tasks: dict[str, set[asyncio.Task]] = {}
        self._call_stream_flags: dict[str, list[threading.Event]] = {}

        self._logger.info("tts_worker_pool_initialised", num_workers=num_workers)

    # ------------------------------------------------------------------
    # Job submission — full audio
    # ------------------------------------------------------------------

    async def submit(
        self,
        call_sid: str,
        turn_id: int,
        text: str,
        audio_prompt_path: str | Path | None = None,
    ) -> asyncio.Future[bytes]:
        """Submit a full-audio TTS job; returns a future for the 16-bit PCM result."""
        worker = self._pick_worker()
        future: asyncio.Future[bytes] = asyncio.get_running_loop().create_future()
        task = asyncio.create_task(
            self._run_job(worker, call_sid, turn_id, text, audio_prompt_path, future),
            name=f"tts-job-{call_sid}-{turn_id}",
        )
        # Track the task so cancel_jobs_for_call can cancel it.
        self._call_tasks.setdefault(call_sid, set()).add(task)
        task.add_done_callback(
            lambda t: self._call_tasks.get(call_sid, set()).discard(t)
        )
        return future

    # ------------------------------------------------------------------
    # Job submission — streaming
    # ------------------------------------------------------------------

    async def submit_stream(
        self,
        call_sid: str,
        turn_id: int,
        text: str,
        audio_prompt_path: str | Path | None = None,
        chunk_size: int | None = None,
    ) -> AsyncIterator[bytes]:
        """Submit a streaming TTS job; returns an async generator of 16-bit PCM bytes.

        A ``threading.Event`` is created for each stream and stored per call_sid.
        Calling :py:meth:`cancel_jobs_for_call` sets the event, signalling the
        background synthesis thread to stop between chunks.
        """
        worker = self._pick_worker()

        # Create a cancel flag and register it before starting the stream so
        # cancel_jobs_for_call can always find it, even in a tight race.
        cancel_flag = threading.Event()
        self._call_stream_flags.setdefault(call_sid, []).append(cancel_flag)

        self._logger.debug(
            "tts_stream_job_submitted", call_sid=call_sid, turn_id=turn_id
        )
        return worker.generate_stream(
            text,
            audio_prompt_path=audio_prompt_path,
            chunk_size=chunk_size,
            cancel_event=cancel_flag,
        )

    # ------------------------------------------------------------------
    # Cancellation
    # ------------------------------------------------------------------

    async def cancel_jobs_for_call(self, call_sid: str) -> None:
        """Cancel all in-flight synthesis jobs for *call_sid*.

        - Full-audio jobs: the backing asyncio ``Task`` is cancelled, which also
          cancels the ``Future`` returned to the caller.
        - Streaming jobs: the ``threading.Event`` cancel flag is set, causing the
          synthesis thread to exit its chunk loop at the next iteration boundary.
        """
        # --- Full-audio tasks ---
        tasks = self._call_tasks.pop(call_sid, set())
        for task in tasks:
            task.cancel()

        # --- Streaming thread flags ---
        flags = self._call_stream_flags.pop(call_sid, [])
        for flag in flags:
            flag.set()

        if tasks or flags:
            self._logger.debug(
                "tts_jobs_cancelled",
                call_sid=call_sid,
                tasks_cancelled=len(tasks),
                streams_stopped=len(flags),
            )

    # ------------------------------------------------------------------
    # Executor management
    # ------------------------------------------------------------------

    def set_executor(self, executor: ThreadPoolExecutor) -> None:
        """Propagate *executor* to every worker in the pool."""
        for worker in self.workers:
            worker.set_executor(executor)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _pick_worker(self) -> ChatterboxTTSService:
        """Return the next worker via round-robin."""
        worker = self.workers[self._next_worker % len(self.workers)]
        self._next_worker += 1
        return worker

    async def _run_job(
        self,
        worker: ChatterboxTTSService,
        call_sid: str,
        turn_id: int,
        text: str,
        audio_prompt_path: str | Path | None,
        future: asyncio.Future[bytes],
    ) -> None:
        """Run a full-audio synthesis job and resolve or reject *future*."""
        try:
            audio_bytes = await worker.generate(text, audio_prompt_path)
            if not future.cancelled():
                future.set_result(audio_bytes)
                self._logger.debug(
                    "tts_job_complete", call_sid=call_sid, turn_id=turn_id
                )
        except asyncio.CancelledError:
            if not future.cancelled():
                future.cancel()
            raise
        except Exception as exc:
            if not future.cancelled():
                future.set_exception(exc)
            self._logger.error(
                "tts_job_failed", call_sid=call_sid, turn_id=turn_id, error=str(exc)
            )
