"""Background TTS worker for generating audio asynchronously.

This module provides a worker that consumes TTS jobs from a queue
and generates audio using the TTS service. This decouples TTS from
the main WebSocket task and allows for pooling/batching.
"""

import asyncio
import json
from collections.abc import Callable
from typing import Any

import structlog

from config import get_settings
from hermes.services.tts import MockTTSService, TTSService

logger = structlog.get_logger(__name__)


class TTSJob:
    """Represents a TTS job."""

    def __init__(
        self,
        text: str,
        callback: Callable[[bytes], Any] | None = None,
        priority: int = 0,
        metadata: dict | None = None,
    ):
        """Initialize a TTS job.

        Args:
            text: Text to synthesize.
            callback: Callback function to call with audio bytes.
            priority: Job priority (higher = more important).
            metadata: Additional metadata.
        """
        self.text = text
        self.callback = callback
        self.priority = priority
        self.metadata = metadata or {}
        self.created_at = asyncio.get_event_loop().time()
        self.audio: bytes | None = None

    @property
    def age(self) -> float:
        """Time since job creation."""
        return asyncio.get_event_loop().time() - self.created_at


class TTSWorker:
    """Background worker for TTS generation.

    This worker consumes TTS jobs from a Redis queue or in-memory queue
    and processes them using the TTS service.
    """

    def __init__(
        self,
        queue: "asyncio.Queue[TTSJob] | None" = None,
        max_workers: int = 2,
        use_redis: bool = False,
    ) -> None:
        """Initialize the TTS worker.

        Args:
            queue: Queue to consume jobs from. Creates one if None.
            max_workers: Number of concurrent workers.
            use_redis: Whether to use Redis for job queue.
        """
        self.settings = get_settings()
        self._logger = structlog.get_logger(__name__)
        self.max_workers = max_workers
        self.use_redis = use_redis

        self._queue = queue or asyncio.Queue()
        self._running = False
        self._workers: set[asyncio.Task] = set()
        self._tts_service: TTSService | None = None

        # Metrics
        self.jobs_processed = 0
        self.jobs_failed = 0
        self.total_latency = 0.0

    async def start(self) -> None:
        """Start the TTS worker."""
        self._running = True
        self._tts_service = TTSService()

        self._logger.info(
            "tts_worker_started",
            max_workers=self.max_workers,
            use_redis=self.use_redis,
        )

        # Start worker tasks
        for i in range(self.max_workers):
            task = asyncio.create_task(
                self._worker_loop(i),
                name=f"tts-worker-{i}",
            )
            self._workers.add(task)
            task.add_done_callback(self._workers.discard)

    async def stop(self) -> None:
        """Stop the TTS worker."""
        self._running = False
        self._logger.info("tts_worker_stopping")

        # Cancel all workers
        for worker in self._workers:
            worker.cancel()

        # Wait for workers to finish
        if self._workers:
            await asyncio.gather(*self._workers, return_exceptions=True)

        self._logger.info(
            "tts_worker_stopped",
            jobs_processed=self.jobs_processed,
            jobs_failed=self.jobs_failed,
        )

    async def _worker_loop(self, worker_id: int) -> None:
        """Main worker loop.

        Args:
            worker_id: Worker identifier.
        """
        self._logger.info("tts_worker_ready", worker_id=worker_id)

        while self._running:
            try:
                # Get job from queue
                job = await asyncio.wait_for(
                    self._queue.get(),
                    timeout=1.0,
                )

                # Process job
                await self._process_job(job)

            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                self._logger.info("tts_worker_cancelled", worker_id=worker_id)
                break
            except Exception as e:
                self._logger.error("tts_worker_error", error=str(e), worker_id=worker_id)

    async def _process_job(self, job: TTSJob) -> None:
        """Process a TTS job.

        Args:
            job: The TTS job to process.
        """
        if not self._tts_service:
            return

        start_time = asyncio.get_event_loop().time()

        try:
            self._logger.debug(
                "tts_job_processing",
                text=job.text[:50],
                priority=job.priority,
            )

            # Generate audio
            audio_tensor = await self._tts_service.synthesize(job.text)

            # Convert tensor to bytes
            import numpy as np

            audio_bytes = (audio_tensor.numpy() * 32767).astype(np.int16).tobytes()
            job.audio = audio_bytes

            # Call callback if provided
            if job.callback:
                await job.callback(audio_bytes)

            # Update metrics
            self.jobs_processed += 1
            latency = asyncio.get_event_loop().time() - start_time
            self.total_latency += latency

            self._logger.debug(
                "tts_job_completed",
                text=job.text[:50],
                latency=latency,
                audio_bytes=len(audio_bytes),
            )

        except Exception as e:
            self.jobs_failed += 1
            self._logger.error("tts_job_failed", error=str(e), text=job.text[:50])
            raise

    async def submit(self, job: TTSJob) -> None:
        """Submit a TTS job.

        Args:
            job: The TTS job to submit.
        """
        await self._queue.put(job)
        self._logger.debug("tts_job_submitted", text=job.text[:50])

    def get_metrics(self) -> dict:
        """Get worker metrics.

        Returns:
            Dictionary with worker metrics.
        """
        avg_latency = (
            self.total_latency / self.jobs_processed
            if self.jobs_processed > 0
            else 0.0
        )

        return {
            "jobs_processed": self.jobs_processed,
            "jobs_failed": self.jobs_failed,
            "avg_latency_seconds": avg_latency,
            "queue_size": self._queue.qsize(),
            "active_workers": len(self._workers),
        }


async def main() -> None:
    """Run the TTS worker as a standalone process."""
    worker = TTSWorker()

    try:
        await worker.start()

        # Keep running until interrupted
        while True:
            await asyncio.sleep(1)
            metrics = worker.get_metrics()
            logger.info("tts_worker_metrics", **metrics)

    except KeyboardInterrupt:
        logger.info("shutdown_signal_received")
    finally:
        await worker.stop()


if __name__ == "__main__":
    asyncio.run(main())
