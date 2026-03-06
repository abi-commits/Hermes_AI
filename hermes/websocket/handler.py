"""Main WebSocket endpoint handler for Twilio media streams."""

import json

import structlog
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

from hermes.websocket.manager import connection_manager
from hermes.websocket.schemas import (
    ConnectedMessage,
    MediaMessage,
    StartMessage,
    StopMessage,
)

logger = structlog.get_logger(__name__)
websocket_router = APIRouter()


@websocket_router.websocket("/{call_sid}")
async def handle_websocket(websocket: WebSocket, call_sid: str) -> None:
    """Handle a Twilio media stream WebSocket connection."""
    await websocket.accept()
    logger.info("websocket_accepted", call_sid=call_sid)

    call = None

    try:
        while True:
            # Receive message from Twilio
            data = await websocket.receive_text()
            message = json.loads(data)

            event_type = message.get("event")

            if event_type == "connected":
                # Connected event - just log it
                connected = ConnectedMessage(**message)
                logger.info(
                    "twilio_connected",
                    protocol=connected.protocol,
                    version=connected.version,
                )

            elif event_type == "start":
                # Start event - initialize the call
                start_msg = StartMessage(**message)

                # Verify call_sid matches
                if start_msg.start.call_sid != call_sid:
                    logger.error(
                        "call_sid_mismatch",
                        expected=call_sid,
                        received=start_msg.start.call_sid,
                    )
                    await websocket.close(code=4001, reason="Call SID mismatch")
                    return

                # Create call instance
                call = await connection_manager.connect(websocket, start_msg)

            elif event_type == "media":
                # Media event - process audio
                if call is None:
                    logger.warning("media_before_start", call_sid=call_sid)
                    continue

                media_msg = MediaMessage(**message)
                await connection_manager.handle_media(media_msg)

            elif event_type == "dtmf":
                # DTMF event - handle touch tones
                dtmf_digit = message.get("dtmf", {}).get("digit")
                logger.info("dtmf_received", call_sid=call_sid, digit=dtmf_digit)

                if call:
                    await call.handle_dtmf(dtmf_digit)

            elif event_type == "stop":
                # Stop event - end the call
                stop_msg = StopMessage(**message)
                logger.info("call_stopping", stream_sid=stop_msg.stream_sid)
                await connection_manager.disconnect(stop_msg.stream_sid)
                break

            elif event_type == "mark":
                # Mark event - handle timing markers
                logger.debug("mark_received", message=message)

            elif event_type == "clear":
                # Clear event - audio cleared
                logger.debug("clear_received", message=message)

            else:
                logger.warning("unknown_event_type", event=event_type, message=message)

    except WebSocketDisconnect:
        logger.info("websocket_disconnected", call_sid=call_sid)
    except json.JSONDecodeError as e:
        logger.error("json_decode_error", error=str(e), data=data[:200])
    except Exception as e:
        logger.exception("websocket_error", error=str(e), call_sid=call_sid)
    finally:
        # Ensure cleanup
        if call:
            await connection_manager.disconnect(call.stream_sid)


@websocket_router.get("/test-client")
async def get_test_client() -> HTMLResponse:
    """Serve a simple WebSocket test client (for development only)."""
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Hermes WebSocket Test</title>
    </head>
    <body>
        <h1>Hermes WebSocket Test Client</h1>
        <div id="status">Disconnected</div>
        <button onclick="connect()">Connect</button>
        <button onclick="disconnect()">Disconnect</button>
        <div id="messages"></div>

        <script>
            let ws = null;

            function connect() {
                const callSid = 'test-' + Math.random().toString(36).substr(2, 9);
                ws = new WebSocket(`ws://localhost:8000/stream/${callSid}`);

                ws.onopen = function() {
                    document.getElementById('status').textContent = 'Connected';
                    log('Connected to ' + callSid);

                    // Send start message
                    ws.send(JSON.stringify({
                        event: 'start',
                        sequenceNumber: 1,
                        start: {
                            callSid: callSid,
                            accountSid: 'test-account',
                            streamSid: 'test-stream-' + Date.now()
                        }
                    }));
                };

                ws.onmessage = function(event) {
                    log('Received: ' + event.data);
                };

                ws.onclose = function() {
                    document.getElementById('status').textContent = 'Disconnected';
                    log('Disconnected');
                };

                ws.onerror = function(error) {
                    log('Error: ' + error);
                };
            }

            function disconnect() {
                if (ws) {
                    ws.close();
                }
            }

            function log(message) {
                const div = document.getElementById('messages');
                div.innerHTML += '<p>' + new Date().toISOString() + ': ' + message + '</p>';
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html)
