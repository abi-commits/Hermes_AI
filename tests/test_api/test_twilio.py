"""Tests for Twilio webhook handling."""

import httpx
import pytest
from unittest.mock import patch

from fastapi import FastAPI

from config import get_settings
from hermes.api.twilio import router


def _build_app() -> FastAPI:
    app = FastAPI()
    app.include_router(router)
    return app


@pytest.mark.asyncio
async def test_voice_webhook_validates_using_form_body(monkeypatch):
    """Signature validation should use POSTed Twilio fields, not query params."""
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("TWILIO_AUTH_TOKEN", "test-token")
    get_settings.cache_clear()

    captured: dict[str, object] = {}

    class DummyValidator:
        def __init__(self, token: str) -> None:
            captured["token"] = token

        def validate(
            self,
            url: str,
            params: dict[str, str | list[str]],
            signature: str,
        ) -> bool:
            captured["url"] = url
            captured["params"] = params
            captured["signature"] = signature
            return True

    with patch("twilio.request_validator.RequestValidator", DummyValidator):
        transport = httpx.ASGITransport(app=_build_app())
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            response = await client.post(
                "/twilio/voice?foo=bar",
                data={
                    "CallSid": "CA123",
                    "AccountSid": "AC123",
                    "From": "+15550000001",
                    "To": "+15550000002",
                    "CallStatus": "ringing",
                },
                headers={
                    "X-Twilio-Signature": "sig",
                    "X-Forwarded-Proto": "https",
                    "X-Forwarded-Host": "voice.example.com",
                },
            )

    assert response.status_code == 200
    assert captured["token"] == "test-token"
    assert captured["signature"] == "sig"
    assert captured["url"] == "https://voice.example.com/twilio/voice?foo=bar"
    assert captured["params"] == {
        "CallSid": "CA123",
        "AccountSid": "AC123",
        "From": "+15550000001",
        "To": "+15550000002",
        "CallStatus": "ringing",
    }

    get_settings.cache_clear()


@pytest.mark.asyncio
async def test_voice_webhook_rejects_invalid_signature(monkeypatch):
    """Production webhook requests should fail closed on invalid signatures."""
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("TWILIO_AUTH_TOKEN", "test-token")
    get_settings.cache_clear()

    class DummyValidator:
        def __init__(self, token: str) -> None:  # noqa: ARG002
            pass

        def validate(
            self,
            url: str,  # noqa: ARG002
            params: dict[str, str | list[str]],  # noqa: ARG002
            signature: str,  # noqa: ARG002
        ) -> bool:
            return False

    with patch("twilio.request_validator.RequestValidator", DummyValidator):
        transport = httpx.ASGITransport(app=_build_app())
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            response = await client.post(
                "/twilio/voice",
                data={
                    "CallSid": "CA123",
                    "AccountSid": "AC123",
                },
                headers={"X-Twilio-Signature": "sig"},
            )

    assert response.status_code == 403
    assert response.json()["detail"] == "Invalid Twilio signature"

    get_settings.cache_clear()
