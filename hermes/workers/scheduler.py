"""Periodic task scheduler for background jobs.

This module provides a simple scheduler for running periodic tasks
like metrics collection, database cleanup, etc.
"""

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class ScheduledTask:
    """Represents a scheduled task."""

    name: str
    func: Callable[..., Any]
    interval_seconds: float
    args: tuple = ()
    kwargs: dict | None = None

    def __post_init__(self) -> None:
        """Initialize default kwargs."""
        if self.kwargs is None:
            self.kwargs = {}


class Scheduler:
    """Simple periodic task scheduler.

    This scheduler runs tasks at specified intervals using asyncio.
    """

    def __init__(self) -> None:
        """Initialize the scheduler."""
        self._tasks: dict[str, ScheduledTask] = {}
        self._running_tasks: dict[str, asyncio.Task] = {}
        self._running = False
        self._logger = structlog.get_logger(__name__)

    def add_task(
        self,
        name: str,
        func: Callable[..., Any],
        interval_seconds: float,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Add a task to the scheduler.

        Args:
            name: Task name.
            func: Function to run.
            interval_seconds: Interval between runs.
            *args: Positional arguments for the function.
            **kwargs: Keyword arguments for the function.
        """
        self._tasks[name] = ScheduledTask(
            name=name,
            func=func,
            interval_seconds=interval_seconds,
            args=args,
            kwargs=kwargs,
        )
        self._logger.debug("task_added", name=name, interval=interval_seconds)

    def remove_task(self, name: str) -> None:
        """Remove a task from the scheduler.

        Args:
            name: Task name.
        """
        if name in self._tasks:
            del self._tasks[name]
            self._logger.debug("task_removed", name=name)

        if name in self._running_tasks:
            self._running_tasks[name].cancel()
            del self._running_tasks[name]

    async def start(self) -> None:
        """Start the scheduler."""
        self._running = True
        self._logger.info("scheduler_started", task_count=len(self._tasks))

        for name, task in self._tasks.items():
            self._running_tasks[name] = asyncio.create_task(
                self._run_task(task),
                name=f"scheduler-{name}",
            )

    async def stop(self) -> None:
        """Stop the scheduler."""
        self._running = False
        self._logger.info("scheduler_stopping")

        for name, task in self._running_tasks.items():
            task.cancel()
            self._logger.debug("task_cancelled", name=name)

        # Wait for tasks to finish
        if self._running_tasks:
            await asyncio.gather(*self._running_tasks.values(), return_exceptions=True)

        self._running_tasks.clear()
        self._logger.info("scheduler_stopped")

    async def _run_task(self, task: ScheduledTask) -> None:
        """Run a scheduled task loop.

        Args:
            task: The scheduled task.
        """
        self._logger.info("task_started", name=task.name)

        while self._running:
            try:
                start_time = asyncio.get_event_loop().time()

                # Run the task
                if asyncio.iscoroutinefunction(task.func):
                    await task.func(*task.args, **(task.kwargs or {}))
                else:
                    task.func(*task.args, **(task.kwargs or {}))

                # Calculate next run time
                elapsed = asyncio.get_event_loop().time() - start_time
                sleep_time = max(0, task.interval_seconds - elapsed)

                self._logger.debug(
                    "task_completed",
                    name=task.name,
                    elapsed=elapsed,
                    next_run_in=sleep_time,
                )

                await asyncio.sleep(sleep_time)

            except asyncio.CancelledError:
                self._logger.info("task_cancelled", name=task.name)
                break
            except Exception as e:
                self._logger.error("task_error", name=task.name, error=str(e))
                await asyncio.sleep(task.interval_seconds)


# Example usage
async def example_task():
    """Example scheduled task."""
    logger.info("example_task_running")


async def main():
    """Run the scheduler example."""
    scheduler = Scheduler()

    # Add example tasks
    scheduler.add_task("metrics", example_task, 60.0)  # Every minute
    scheduler.add_task("cleanup", example_task, 300.0)  # Every 5 minutes

    try:
        await scheduler.start()
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        logger.info("shutdown_signal_received")
    finally:
        await scheduler.stop()


if __name__ == "__main__":
    asyncio.run(main())
