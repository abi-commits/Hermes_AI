"""Twilio webhook endpoints for inbound calls and status updates."""

from __future__ import annotations

import structlog
from fastapi import APIRouter, Form, HTTPException, Request, status
from fastapi.responses import Response

from config import get_settings

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/twilio", tags=["Telephony"])


def _build_stream_twiml(stream_url: str) -> str:
    """Return TwiML that opens a bidirectional media stream to *stream_url*."""
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        "<Response>"
        "<Connect>"
        f'<Stream url="{stream_url}" />'
        "</Connect>"
        "</Response>"
    )


async def _validate_twilio_signature(request: Request, settings) -> bool:
    """Validate the Twilio request signature; returns ``True`` in dev mode (no auth token set)."""
    if not settings.twilio_auth_token:
        logger.warning("twilio_auth_token_not_set_skipping_validation")
        return True

    try:
        from twilio.request_validator import RequestValidator

        validator = RequestValidator(settings.twilio_auth_token)
        signature = request.headers.get("X-Twilio-Signature", "")
        
        # Reconstruct the original requested URL to match Twilio's signature.
        # This is critical when running behind a proxy like ngrok.
        scheme = request.headers.get("X-Forwarded-Proto", request.url.scheme)
        host = request.headers.get("X-Forwarded-Host") or request.url.netloc
        
        # Include query string if present
        query_string = request.url.query
        path = request.url.path
        url = f"{scheme}://{host}{path}"
        if query_string:
            url += f"?{query_string}"
        
        # POST params must be sorted for validation, read form data
        form = await request.form()
        params = dict(form)
        return validator.validate(url, params, signature)
    except Exception as exc:
        logger.error("twilio_signature_validation_error", error=str(exc))
        return False


@router.post(
    "/voice",
    response_class=Response,
    summary="Voice Entrypoint",
    description=(
        "Primary webhook called by Twilio when an inbound call arrives. "
        "Instructs Twilio to open a WebSocket media stream to Hermes."
    ),
)
async def twilio_voice_webhook(
    request: Request,
    CallSid: str = Form(...),
    AccountSid: str = Form(...),
    From: str = Form(default=""),
    To: str = Form(default=""),
    CallStatus: str = Form(default=""),
) -> Response:
    """Handle Twilio inbound call and return TwiML to start a media stream."""
    settings = get_settings()

    # --- Signature validation (production guard) ---
    if settings.is_production:
        is_valid = await _validate_twilio_signature(request, settings)
        if not is_valid:
            logger.warning(
                "twilio_invalid_signature",
                call_sid=CallSid,
                from_=From,
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Invalid Twilio signature",
            )

    logger.info(
        "twilio_call_received",
        call_sid=CallSid,
        account_sid=AccountSid,
        from_=From,
        to=To,
        status=CallStatus,
    )

    # Build the WebSocket URL from the incoming request host so this works
    # behind any proxy / ngrok tunnel without needing a hard-coded PUBLIC_URL.
    host = request.headers.get("X-Forwarded-Host") or request.url.hostname
    
    # Twilio always streams over wss:// in production, use protocol matching for dev
    scheme = "wss" if request.url.scheme == "https" else "ws"
    stream_url = f"{scheme}://{host}/stream/{CallSid}"

    twiml = _build_stream_twiml(stream_url)

    logger.info(
        "twilio_stream_twiml_sent",
        call_sid=CallSid,
        stream_url=stream_url,
    )

    return Response(content=twiml, media_type="application/xml")


@router.post(
    "/status",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Status Callback",
    description="Asynchronous hook receiving call lifecycle updates from Twilio (ringing, answered, etc.).",
)
async def twilio_status_callback(
    CallSid: str = Form(...),
    CallStatus: str = Form(...),
    From: str = Form(default=""),
    To: str = Form(default=""),
    CallDuration: str | None = Form(default=None),
) -> None:
    """Log Twilio call status updates."""
    logger.info(
        "twilio_status_update",
        call_sid=CallSid,
        status=CallStatus,
        from_=From,
        to=To,
        duration_s=CallDuration,
    )
