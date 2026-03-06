"""WebSocket connection manager for active calls."""

import asyncio
from typing import TYPE_CHECKING

import structlog
from fastapi import WebSocket

from hermes.core.call import Call

if TYPE_CHECKING:
    from hermes.core.orchestrator import CallConfig, CallOrchestrator
    from hermes.websocket.schemas import MediaMessage, StartMessage

logger = structlog.get_logger(__name__)


class ConnectionManager:
    """Manages active WebSocket connections and routes Twilio media frames to calls."""

    def __init__(self) -> None:
        """Initialise the connection manager."""
        # call_sid → Call  (always maintained for fast lookups)
        self._active_calls: dict[str, Call] = {}
        # stream_sid → call_sid  (fast audio routing)
        self._stream_to_call: dict[str, str] = {}
        self._lock = asyncio.Lock()
        self._orchestrator: "CallOrchestrator | None" = None
        self._logger = structlog.get_logger(__name__)
        self._services = services

    # ------------------------------------------------------------------
    # Orchestrator integration
    # ------------------------------------------------------------------

    def set_orchestrator(self, orchestrator: "CallOrchestrator") -> None:
        """Attach a ``CallOrchestrator``; must be called before any calls are handled."""
        self._orchestrator = orchestrator
        self._logger.info("orchestrator_attached")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def connect(
        self,
        websocket: WebSocket,
        start_message: "StartMessage",
        config: "CallConfig | None" = None,
    ) -> Call:
        """Handle a new Twilio WebSocket connection and start the call."""
        call_sid = start_message.start.call_sid
        stream_sid = start_message.start.stream_sid
        account_sid = start_message.start.account_sid

        if self._orchestrator is not None:
            call = await self._orchestrator.create_call(
                websocket=websocket,
                call_sid=call_sid,
                stream_sid=stream_sid,
                account_sid=account_sid,
                config=config,
            )
        else:
            # Standalone mode (no orchestrator) — create call directly
            async with self._lock:
                call = Call(
                    call_sid=call_sid,
                    stream_sid=stream_sid,
                    websocket=websocket,
                    account_sid=account_sid,
                )
                self._active_calls[call_sid] = call
            await call.start()

        async with self._lock:
            # Always register the stream→call mapping for audio routing
            self._stream_to_call[stream_sid] = call_sid
            # When an orchestrator is active it is the single source of truth for
            # active-call state; do NOT mirror into ConnectionManager._active_calls
            # to avoid the two dicts diverging under error conditions.

        self._logger.info(
            "call_connected",
            call_sid=call_sid,
            stream_sid=stream_sid,
            active_calls=self.active_calls,
        )
        return call

    async def disconnect(self, stream_sid: str) -> None:
        """Stop the call associated with *stream_sid* and clean up routing entries."""
        standalone_call: Call | None = None
        async with self._lock:
            call_sid = self._stream_to_call.pop(stream_sid, None)
            if call_sid:
                # Capture the call object before removing it so we can stop it
                # in standalone mode (orchestrator handles its own teardown).
                standalone_call = self._active_calls.pop(call_sid, None)

        if call_sid is None:
            self._logger.debug("disconnect_unknown_stream", stream_sid=stream_sid)
            return

        if self._orchestrator is not None:
            await self._orchestrator.terminate_call(call_sid, reason="hangup")
        elif standalone_call is not None:
            # Standalone mode — stop the call directly; there is no orchestrator
            # to do it, so we must call stop() to cancel background tasks cleanly.
            try:
                await standalone_call.stop()
            except Exception as exc:
                self._logger.warning(
                    "standalone_call_stop_error", call_sid=call_sid, error=str(exc)
                )

        self._logger.info(
            "call_disconnected",
            call_sid=call_sid,
            stream_sid=stream_sid,
            active_calls=self.active_calls,
        )

    # ------------------------------------------------------------------
    # Media routing
    # ------------------------------------------------------------------

    async def handle_media(self, message: "MediaMessage") -> None:
        """Route an incoming Twilio audio frame to the correct call."""
        stream_sid = message.stream_sid
        call_sid = self._stream_to_call.get(stream_sid)

        if not call_sid:
            self._logger.warning(
                "media_received_for_unknown_stream", stream_sid=stream_sid
            )
            return

        # Use get_call() so that, when an orchestrator is present, we always read
        # from its registry (the single source of truth) rather than a stale
        # local mirror.
        call = self.get_call(call_sid)
        if not call:
            self._logger.warning(
                "media_received_for_unknown_call",
                call_sid=call_sid,
                stream_sid=stream_sid,
            )
            return

        await call.process_audio_chunk(message.media.payload)

    # ------------------------------------------------------------------
    # Lookups
    # ------------------------------------------------------------------

    def get_call(self, call_sid: str) -> Call | None:
        """Return the active ``Call`` for *call_sid*, or ``None``."""
        if self._orchestrator is not None:
            return self._orchestrator.get_call(call_sid)
        return self._active_calls.get(call_sid)

    def get_call_by_stream(self, stream_sid: str) -> Call | None:
        """Return the active ``Call`` for *stream_sid*, or ``None``."""
        call_sid = self._stream_to_call.get(stream_sid)
        return self.get_call(call_sid) if call_sid else None

    @property
    def active_calls(self) -> int:
        """Number of currently active calls."""
        if self._orchestrator is not None:
            return self._orchestrator.active_call_count
        return len(self._active_calls)

    @property
    def active_calls_list(self) -> list[str]:
        """List of active call SIDs."""
        if self._orchestrator is not None:
            return list(self._orchestrator.active_calls.keys())
        return list(self._active_calls.keys())

    async def broadcast_metrics(self) -> None:
        """Log a structured snapshot of all active calls (for monitoring)."""
        if self._orchestrator is not None:
            calls_data = self._orchestrator.active_calls
        else:
            calls_data = self._active_calls

        metrics = {
            "active_calls": len(calls_data),
            "calls": [
                {
                    "call_sid": call.call_sid,
                    "duration_seconds": round(call.duration_seconds, 1),
                    "state": call.state.name,
                }
                for call in calls_data.values()
            ],
        }
        self._logger.debug("active_calls_metrics", **metrics)


def get_connection_manager(services: "ServiceContainer | None" = None) -> ConnectionManager:
    """Create a ConnectionManager with the given service container.

    Args:
        services: Service container for dependency injection.

    Returns:
        A new ConnectionManager instance.
    """
    return ConnectionManager(services=services)
