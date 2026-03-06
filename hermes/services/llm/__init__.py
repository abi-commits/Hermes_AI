"""LLM service package.

Public API
----------
``GeminiLLMService``
    Production LLM backed by Google Gemini 2.5 Flash.

``MockGeminiLLMService``
    Deterministic stub for tests and local development.

``AbstractLLMService``
    ABC defining the interface all implementations must fulfil.

``create_function_tool``
    Decorator factory for registering Gemini function-calling tools.
"""

from hermes.services.llm.base import AbstractLLMService
from hermes.services.llm.gemini import GeminiLLMService
from hermes.services.llm.mock import MockGeminiLLMService
from hermes.services.llm.tools import create_function_tool

__all__ = [
    "AbstractLLMService",
    "GeminiLLMService",
    "MockGeminiLLMService",
    "create_function_tool",
]
