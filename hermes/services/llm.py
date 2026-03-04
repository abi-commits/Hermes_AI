"""Language Model service integration using Gemini."""

import json
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

import httpx
import structlog
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from config import get_settings
from hermes.core.exceptions import LLMError, ServiceUnavailableError

if TYPE_CHECKING:
    pass

logger = structlog.get_logger(__name__)


class LLMService:
    """Language Model service using Google's Gemini API.

    This class provides streaming text generation with conversation
    context management and optional RAG integration.
    """

    def __init__(self, http_client: httpx.AsyncClient | None = None) -> None:
        """Initialize the LLM service.

        Args:
            http_client: Shared HTTP client for connection pooling.
                         Creates a per-request client if not provided.
        """
        self.settings = get_settings()
        self._logger = structlog.get_logger(__name__)
        self._api_base = "https://generativelanguage.googleapis.com/v1beta"
        self._http_client = http_client

        if not self.settings.gemini_api_key:
            self._logger.warning("gemini_api_key_not_set")

    @retry(
        retry=retry_if_exception_type((LLMError, ConnectionError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    async def generate(
        self,
        context: str,
        query: str,
        system_prompt: str | None = None,
    ) -> str:
        """Generate a response to a query.

        Args:
            context: Additional context for the response.
            query: The user's query.
            system_prompt: Optional system prompt override.

        Returns:
            Generated response text.
        """
        if not self.settings.gemini_api_key:
            raise ServiceUnavailableError("Gemini", "API key not configured")

        # Build the prompt
        if system_prompt:
            system_instruction = system_prompt
        else:
            system_instruction = (
                "You are a helpful AI assistant for a voice support system. "
                "Keep responses concise and natural for spoken conversation. "
                "Avoid markdown formatting and special characters."
            )

        # Build content
        contents = []

        if context:
            contents.append({
                "role": "user",
                "parts": [{"text": f"Context:\n{context}"}],
            })
            contents.append({
                "role": "model",
                "parts": [{"text": "I understand the context. How can I help?"}],
            })

        contents.append({
            "role": "user",
            "parts": [{"text": query}],
        })

        # Build request
        request_data = {
            "contents": contents,
            "system_instruction": {
                "parts": [{"text": system_instruction}],
            },
            "generation_config": {
                "temperature": self.settings.gemini_temperature,
                "maxOutputTokens": self.settings.gemini_max_tokens,
                "topP": 0.95,
                "topK": 40,
            },
        }

        try:
            client = self._http_client or httpx.AsyncClient(timeout=60.0)
            try:
                response = await client.post(
                    f"{self._api_base}/models/{self.settings.gemini_model}:generateContent",
                    headers={
                        "Content-Type": "application/json",
                        "x-goog-api-key": self.settings.gemini_api_key,
                    },
                    json=request_data,
                    timeout=60.0,
                )

                if response.status_code != 200:
                    error_text = response.text
                    raise LLMError(
                        f"Gemini API error: {response.status_code} - {error_text}"
                    )

                data = response.json()

                # Extract text from response
                candidates = data.get("candidates", [])
                if not candidates:
                    raise LLMError("No response generated")

                content = candidates[0].get("content", {})
                parts = content.get("parts", [])

                if not parts:
                    raise LLMError("Empty response from model")

                text = "".join(part.get("text", "") for part in parts)

                self._logger.debug(
                    "llm_response_generated",
                    query=query[:50],
                    response=text[:100],
                )

                return text
            finally:
                if not self._http_client:
                    await client.aclose()

        except httpx.TimeoutException as e:
            raise LLMError(f"Gemini API timeout: {e}")
        except Exception as e:
            self._logger.error("llm_generation_failed", error=str(e))
            raise LLMError(f"LLM generation failed: {e}")

    async def generate_stream(
        self,
        context: str,
        query: str,
        system_prompt: str | None = None,
    ) -> AsyncIterator[str]:
        """Generate a streaming response.

        Args:
            context: Additional context for the response.
            query: The user's query.
            system_prompt: Optional system prompt override.

        Yields:
            Response text chunks as they are generated.
        """
        if not self.settings.gemini_api_key:
            raise ServiceUnavailableError("Gemini", "API key not configured")

        # Build the prompt
        if system_prompt:
            system_instruction = system_prompt
        else:
            system_instruction = (
                "You are a helpful AI assistant for a voice support system. "
                "Keep responses concise and natural for spoken conversation. "
                "Avoid markdown formatting and special characters."
            )

        # Build content
        contents = []

        if context:
            contents.append({
                "role": "user",
                "parts": [{"text": f"Context:\n{context}"}],
            })
            contents.append({
                "role": "model",
                "parts": [{"text": "I understand the context. How can I help?"}],
            })

        contents.append({
            "role": "user",
            "parts": [{"text": query}],
        })

        # Build request
        request_data = {
            "contents": contents,
            "system_instruction": {
                "parts": [{"text": system_instruction}],
            },
            "generation_config": {
                "temperature": self.settings.gemini_temperature,
                "maxOutputTokens": self.settings.gemini_max_tokens,
                "topP": 0.95,
                "topK": 40,
            },
        }

        client = self._http_client or httpx.AsyncClient(timeout=60.0)
        try:
            async with client.stream(
                "POST",
                f"{self._api_base}/models/{self.settings.gemini_model}:streamGenerateContent",
                headers={
                    "Content-Type": "application/json",
                    "x-goog-api-key": self.settings.gemini_api_key,
                },
                json=request_data,
                timeout=60.0,
            ) as response:
                    if response.status_code != 200:
                        error_text = await response.aread()
                        raise LLMError(
                            f"Gemini API error: {response.status_code} - {error_text}"
                        )

                    async for line in response.aiter_lines():
                        if line.startswith("data: "):
                            data_str = line[6:]
                            if data_str == "[DONE]":
                                break

                            try:
                                data = json.loads(data_str)
                                candidates = data.get("candidates", [])
                                if candidates:
                                    content = candidates[0].get("content", {})
                                    parts = content.get("parts", [])
                                    for part in parts:
                                        text = part.get("text", "")
                                        if text:
                                            yield text
                            except json.JSONDecodeError:
                                continue

        except httpx.TimeoutException as e:
            raise LLMError(f"Gemini API timeout: {e}")
        except Exception as e:
            self._logger.error("llm_stream_failed", error=str(e))
            raise LLMError(f"LLM stream generation failed: {e}")
        finally:
            if not self._http_client:
                await client.aclose()


class MockLLMService(LLMService):
    """Mock LLM service for testing."""

    def __init__(self, responses: list[str] | None = None) -> None:
        """Initialize mock service.

        Args:
            responses: List of mock responses to cycle through.
        """
        self.responses = responses or ["This is a mock response from the LLM."]
        self._index = 0
        self._logger = structlog.get_logger(__name__)

    async def generate(
        self,
        context: str,
        query: str,
        system_prompt: str | None = None,
    ) -> str:
        """Return mock response."""
        response = self.responses[self._index % len(self.responses)]
        self._index += 1
        self._logger.debug("mock_llm_response", response=response)
        return response

    async def generate_stream(
        self,
        context: str,
        query: str,
        system_prompt: str | None = None,
    ) -> AsyncIterator[str]:
        """Yield mock response as stream."""
        response = await self.generate(context, query, system_prompt)
        # Yield words one at a time to simulate streaming
        for word in response.split():
            yield word + " "
