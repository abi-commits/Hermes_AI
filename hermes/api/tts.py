"""API endpoints for direct Text-to-Speech synthesis."""

from __future__ import annotations

import structlog
from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import Response
from pydantic import BaseModel, Field

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/tts", tags=["Speech Synthesis"])


class SynthesisRequest(BaseModel):
    """Request model for direct TTS synthesis."""

    text: str = Field(..., min_length=1, example="Hello, this is a test of the Hermes TTS system.")


@router.post(
    "/synthesize",
    response_class=Response,
    summary="Synthesize Audio",
    description=(
        "Directly converts text to high-quality audio using the configured TTS engine. "
        "If TTS_PROVIDER is set to 'modal_remote', this will call the remote GPU server. "
        "Returns raw 16-bit PCM audio bytes at the engine's native sample rate."
    ),
)
async def synthesize_speech(
    body: SynthesisRequest, request: Request
) -> Response:
    """Synthesize text into audio bytes."""
    tts = request.app.state.tts_service
    
    if not tts:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="TTS service is not initialised.",
        )

    try:
        provider_name = type(tts).__name__
        logger.info("direct_synthesis_requested", text_len=len(body.text), provider=provider_name)
        
        # Generate full audio bytes (hits GPU if provider is modal_remote)
        audio_bytes = await tts.generate(body.text)
        
        # Return as raw octet-stream
        return Response(
            content=audio_bytes,
            media_type="audio/l16",
            headers={
                "X-Sample-Rate": str(tts.sample_rate),
                "X-TTS-Provider": provider_name,
                "Content-Disposition": 'attachment; filename="synthesis.pcm"'
            }
        )
        
    except Exception as exc:
        logger.error("direct_synthesis_failed", error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Synthesis failed: {exc}",
        )
