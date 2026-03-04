"""WebSocket connection manager for active calls."""

import asyncio
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

import structlog
from fastapi import WebSocket, WebSocketDisconnect

from hermes.core.call import Call

if TYPE_CHECKING:
    from hermes.services.container import ServiceContainer
    from hermes.websocket.schemas import MediaMessage, StartMessage

logger = structlog.get_logger(__name__)


class ConnectionManager:
    """Manage active WebSocket connections for voice calls.

    This class handles:
    - Active WebSocket connections keyed by call SID
    - Call state machines with injected services
    - Broadcasting messages to connections
    """

    def __init__(self, services: "ServiceContainer | None" = None) -> None:
        """Initialize the connection manager.

        Args:
            services: Service container for dependency injection into calls.
        """
        # Map of call_sid -> Call objects
        self._active_calls: dict[str, Call] = {}
        # Map of stream_sid -> call_sid for lookup
        self._stream_to_call: dict[str, str] = {}
        self._lock = asyncio.Lock()
        self._logger = structlog.get_logger(__name__)
        self._services = services

    async def connect(
        self,
        websocket: WebSocket,
        start_message: "StartMessage",
    ) -> Call:
        """Handle a new WebSocket connection.

        Args:
            websocket: The WebSocket connection.
            start_message: The Twilio start message with stream parameters.

        Returns:
            Call: The created call state machine.
        """
        call_sid = start_message.start.call_sid
        stream_sid = start_message.start.stream_sid

        async with self._lock:
            # Create new call state machine with injected services
            call = Call(
                call_sid=call_sid,
                stream_sid=stream_sid,
                websocket=websocket,
                account_sid=start_message.start.account_sid,
                stt_service=self._services.stt_service if self._services else None,
                llm_service=self._services.llm_service if self._services else None,
                tts_service=self._services.tts_service if self._services else None,
                rag_service=self._services.rag_service if self._services else None,
            )

            self._active_calls[call_sid] = call
            self._stream_to_call[stream_sid] = call_sid

            await call.start()

            self._logger.info(
                "call_connected",
                call_sid=call_sid,
                stream_sid=stream_sid,
                active_calls=len(self._active_calls),
            )

            return call

    async def disconnect(self, stream_sid: str) -> None:
        """Handle WebSocket disconnection.

        Args:
            stream_sid: The stream SID from Twilio.
        """
        async with self._lock:
            call_sid = self._stream_to_call.pop(stream_sid, None)
            if call_sid:
                call = self._active_calls.pop(call_sid, None)
                if call:
                    await call.stop()
                    self._logger.info(
                        "call_disconnected",
                        call_sid=call_sid,
                        stream_sid=stream_sid,
                        active_calls=len(self._active_calls),
                    )

    async def handle_media(self, message: "MediaMessage") -> None:
        """Handle incoming audio media.

        Args:
            message: The Twilio media message.
        """
        stream_sid = message.stream_sid
        call_sid = self._stream_to_call.get(stream_sid)

        if not call_sid:
            self._logger.warning(
                "media_received_for_unknown_stream",
                stream_sid=stream_sid,
            )
            return

        call = self._active_calls.get(call_sid)
        if not call:
            self._logger.warning(
                "media_received_for_unknown_call",
                call_sid=call_sid,
                stream_sid=stream_sid,
            )
            return

        # Decode and queue audio for processing
        await call.process_audio_chunk(message.media.payload)

    def get_call(self, call_sid: str) -> Call | None:
        """Get an active call by SID.

        Args:
            call_sid: The call SID.

        Returns:
            Call or None: The call if active, None otherwise.
        """
        return self._active_calls.get(call_sid)

    def get_call_by_stream(self, stream_sid: str) -> Call | None:
        """Get an active call by stream SID.

        Args:
            stream_sid: The stream SID.

        Returns:
            Call or None: The call if active, None otherwise.
        """
        call_sid = self._stream_to_call.get(stream_sid)
        if call_sid:
            return self._active_calls.get(call_sid)
        return None

    @property
    def active_calls(self) -> int:
        """Number of active calls."""
        return len(self._active_calls)

    @property
    def active_calls_list(self) -> list[str]:
        """List of active call SIDs."""
        return list(self._active_calls.keys())

    async def broadcast_metrics(self) -> None:
        """Broadcast metrics for all active calls (for monitoring)."""
        metrics = {
            "active_calls": self.active_calls,
            "calls": [
                {
                    "call_sid": call.call_sid,
                    "duration_seconds": call.duration_seconds,
                    "state": call.state.value,
                }
                for call in self._active_calls.values()
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
