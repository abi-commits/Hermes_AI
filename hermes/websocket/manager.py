"""WebSocket connection manager for active calls."""

import asyncio
from typing import TYPE_CHECKING

import structlog
from fastapi import WebSocket

from hermes.core.call import Call

if TYPE_CHECKING:
    from hermes.core.orchestrator import CallConfig, CallOrchestrator
    from hermes.websocket.schemas import StartMessage


class ConnectionManager:
    """Registry of active media stream connections."""

    def __init__(self) -> None:
        self._active_calls: dict[str, Call] = {}
        self._stream_map: dict[str, str] = {}  # stream_sid -> call_sid
        self._orchestrator: CallOrchestrator | None = None
        self._logger = structlog.get_logger(__name__)

    @property
    def orchestrator(self) -> "CallOrchestrator | None":
        """Access the current call orchestrator."""
        return self._orchestrator

    def set_orchestrator(self, orchestrator: "CallOrchestrator") -> None:
        """Attach the call orchestrator for lifecycle management."""
        self._orchestrator = orchestrator

    async def connect(
        self,
        websocket: WebSocket,
        start_msg: "StartMessage",
        config: "CallConfig | None" = None,
    ) -> Call:
        """Register a new active call and media stream."""
        call_sid = start_msg.start.call_sid
        stream_sid = start_msg.start.stream_sid
        account_sid = start_msg.start.account_sid

        self._logger.info(
            "manager_connect_starting",
            call_sid=call_sid,
            stream_sid=stream_sid,
            has_orchestrator=bool(self._orchestrator),
            has_greeting=bool(config.greeting) if config else False,
        )

        try:
            # If we have an orchestrator, use it to manage the call lifecycle
            if self._orchestrator:
                self._logger.debug("using_orchestrator_for_call", call_sid=call_sid)
                call = await self._orchestrator.create_call(
                    websocket=websocket,
                    call_sid=call_sid,
                    stream_sid=stream_sid,
                    account_sid=account_sid,
                    config=config,
                )
            else:
                # Fallback for standalone/test usage
                self._logger.debug("using_standalone_call", call_sid=call_sid)
                call = Call(
                    call_sid=call_sid,
                    stream_sid=stream_sid,
                    websocket=websocket,
                    account_sid=account_sid,
                    persona=config.persona if config else "default",
                )
                # Standalone mode must still respect the greeting
                await call.start(greeting=config.greeting if config else None)

            self._active_calls[call_sid] = call
            self._stream_map[stream_sid] = call_sid

            self._logger.info(
                "websocket_connected",
                call_sid=call_sid,
                stream_sid=stream_sid,
                total_active=len(self._active_calls),
                call_state=call.state.name if hasattr(call.state, 'name') else str(call.state),
            )
            return call
        except Exception as e:
            self._logger.exception("manager_connect_failed", call_sid=call_sid, error=str(e))
            raise

    async def disconnect(self, stream_sid: str) -> None:
        """Unregister a stream and signal the orchestrator to terminate the call."""
        call_sid = self._stream_map.pop(stream_sid, None)
        if not call_sid:
            return

        call = self._active_calls.pop(call_sid, None)
        if not call:
            return

        # If orchestrated, the orchestrator handles the stop() call
        if self._orchestrator:
            await self._orchestrator.terminate_call(call_sid, reason="disconnect")
        else:
            await call.stop(status="completed")

        self._logger.info(
            "websocket_disconnected",
            call_sid=call_sid,
            stream_sid=stream_sid,
            total_active=len(self._active_calls),
        )

    async def handle_media(self, message: "MediaMessage") -> None:
        """Route an inbound media chunk to the correct call processor."""
        call_sid = self._stream_map.get(message.stream_sid)
        if not call_sid:
            return

        call = self._active_calls.get(call_sid)
        if call:
            await call.process_audio_chunk(message.media.payload)

    def get_call(self, call_sid: str) -> Call | None:
        """Return the active call for *call_sid*."""
        return self._active_calls.get(call_sid)

    def get_stats(self) -> dict:
        """Return basic connection statistics."""
        metrics = {
            "active_calls": len(self._active_calls),
            "stream_mappings": len(self._stream_map),
        }
        self._logger.debug("connection_stats", **metrics)
        return metrics


# Module-level singleton — shared by main.py, api/calls.py, and the WebSocket handler.
# The orchestrator is attached at startup via ``connection_manager.set_orchestrator()``.
connection_manager = ConnectionManager()
