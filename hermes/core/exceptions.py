"""Custom exceptions for Hermes."""


class HermesError(Exception):
    """Base exception for all Hermes errors."""

    def __init__(self, message: str, error_code: str | None = None) -> None:
        """Initialize the exception.

        Args:
            message: Error message.
            error_code: Optional error code for categorization.
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code

    def __str__(self) -> str:
        """String representation."""
        if self.error_code:
            return f"[{self.error_code}] {self.message}"
        return self.message


class CallError(HermesError):
    """Errors related to call management."""

    pass


class STTError(HermesError):
    """Errors from speech-to-text services."""

    pass


class LLMError(HermesError):
    """Errors from language model services."""

    pass


class TTSError(HermesError):
    """Errors from text-to-speech services."""

    pass


class TTSGenerationError(TTSError):
    """Errors during TTS audio generation."""

    pass


class RAGError(HermesError):
    """Errors from retrieval-augmented generation."""

    pass


class VectorDBError(HermesError):
    """Errors from vector database operations."""

    pass


class AudioProcessingError(HermesError):
    """Errors during audio processing."""

    pass


class ConfigurationError(HermesError):
    """Errors related to configuration."""

    pass


class ValidationError(HermesError):
    """Errors related to input validation."""

    pass


class WebSocketError(HermesError):
    """Errors related to WebSocket communication."""

    pass


class ServiceUnavailableError(HermesError):
    """Errors when external services are unavailable."""

    def __init__(self, service: str, message: str | None = None) -> None:
        """Initialize the exception.

        Args:
            service: Name of the unavailable service.
            message: Optional custom message.
        """
        msg = message or f"Service '{service}' is currently unavailable"
        super().__init__(msg, error_code="SERVICE_UNAVAILABLE")
        self.service = service
