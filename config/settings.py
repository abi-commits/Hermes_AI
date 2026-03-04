"""Application settings using Pydantic BaseSettings."""

from functools import lru_cache
from typing import Literal

from pydantic import Field, PostgresDsn, RedisDsn, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ==========================================================================
    # Application Settings
    # ==========================================================================
    app_name: str = Field(default="hermes", description="Application name")
    app_env: Literal["development", "staging", "production"] = Field(
        default="development",
        description="Application environment",
    )
    debug: bool = Field(default=False, description="Debug mode")
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        description="Logging level",
    )

    # ==========================================================================
    # Server Settings
    # ==========================================================================
    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=8000, description="Server port")
    workers: int = Field(default=1, description="Number of worker processes")

    # ==========================================================================
    # Twilio Settings
    # ==========================================================================
    twilio_account_sid: str | None = Field(
        default=None,
        description="Twilio Account SID",
    )
    twilio_auth_token: str | None = Field(
        default=None,
        description="Twilio Auth Token",
    )
    twilio_phone_number: str | None = Field(
        default=None,
        description="Twilio phone number",
    )

    # ==========================================================================
    # Speech-to-Text (Deepgram)
    # ==========================================================================
    deepgram_api_key: str | None = Field(
        default=None,
        description="Deepgram API key",
    )
    deepgram_model: str = Field(
        default="nova-2",
        description="Deepgram model to use",
    )
    deepgram_language: str = Field(
        default="en-US",
        description="Language code for transcription",
    )

    # ==========================================================================
    # LLM (Gemini)
    # ==========================================================================
    gemini_api_key: str | None = Field(
        default=None,
        description="Gemini API key",
    )
    gemini_model: str = Field(
        default="gemini-1.5-flash",
        description="Gemini model to use",
    )
    gemini_temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Sampling temperature",
    )
    gemini_max_tokens: int = Field(
        default=1024,
        ge=1,
        description="Maximum tokens to generate",
    )

    # ==========================================================================
    # Text-to-Speech (Chatterbox)
    # ==========================================================================
    chatterbox_api_url: str = Field(
        default="http://localhost:8001",
        description="Chatterbox TTS API URL",
    )
    chatterbox_voice: str = Field(
        default="default",
        description="Voice to use for TTS",
    )
    chatterbox_speed: float = Field(
        default=1.0,
        ge=0.5,
        le=2.0,
        description="Speech speed multiplier",
    )

    # Alternative TTS providers
    openai_api_key: str | None = Field(
        default=None,
        description="OpenAI API key for TTS",
    )
    elevenlabs_api_key: str | None = Field(
        default=None,
        description="ElevenLabs API key for TTS",
    )

    # ==========================================================================
    # Vector Database
    # ==========================================================================
    vector_db_provider: Literal["chromadb", "pinecone"] = Field(
        default="chromadb",
        description="Vector database provider",
    )
    chromadb_host: str = Field(default="localhost", description="ChromaDB host")
    chromadb_port: int = Field(default=8002, description="ChromaDB port")
    chromadb_collection: str = Field(
        default="hermes_knowledge",
        description="ChromaDB collection name",
    )

    # Pinecone settings
    pinecone_api_key: str | None = Field(default=None, description="Pinecone API key")
    pinecone_environment: str | None = Field(
        default=None,
        description="Pinecone environment",
    )
    pinecone_index: str = Field(
        default="hermes-knowledge",
        description="Pinecone index name",
    )

    # ==========================================================================
    # Redis
    # ==========================================================================
    redis_url: RedisDsn = Field(
        default="redis://localhost:6379/0",
        description="Redis connection URL",
    )

    # ==========================================================================
    # Database
    # ==========================================================================
    database_url: PostgresDsn = Field(
        default="postgresql://user:password@localhost:5432/hermes",
        description="PostgreSQL connection URL",
    )

    # ==========================================================================
    # Monitoring
    # ==========================================================================
    metrics_port: int = Field(default=9090, description="Prometheus metrics port")
    enable_prometheus: bool = Field(
        default=True,
        description="Enable Prometheus metrics",
    )

    # ==========================================================================
    # Audio Settings
    # ==========================================================================
    audio_sample_rate: int = Field(
        default=8000,
        description="Audio sample rate in Hz",
    )
    audio_channels: int = Field(
        default=1,
        description="Number of audio channels",
    )
    audio_chunk_duration_ms: int = Field(
        default=20,
        description="Audio chunk duration in milliseconds",
    )

    # ==========================================================================
    # RAG Settings
    # ==========================================================================
    rag_top_k: int = Field(
        default=5,
        ge=1,
        description="Number of documents to retrieve",
    )
    rag_similarity_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum similarity score for RAG results",
    )
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="Model for text embeddings",
    )

    @field_validator("audio_sample_rate")
    @classmethod
    def validate_sample_rate(cls, v: int) -> int:
        """Validate audio sample rate."""
        valid_rates = [8000, 16000, 22050, 44100, 48000]
        if v not in valid_rates:
            raise ValueError(f"Sample rate must be one of {valid_rates}")
        return v

    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.app_env == "production"

    @property
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.app_env == "development"


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance.

    Returns:
        Settings: Application settings instance.
    """
    return Settings()
