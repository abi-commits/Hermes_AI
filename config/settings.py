"""Application settings using Pydantic BaseSettings."""

from functools import lru_cache
from typing import Literal

from pydantic import Field, PostgresDsn, field_validator
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
    twilio_transfer_number: str | None = Field(
        default=None,
        description="E.164 phone number to transfer calls to when the caller presses 0 (e.g. +15550001234)",
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
        default="gemini-2.5-flash",
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
    # Text-to-Speech (Chatterbox Turbo)
    # ==========================================================================
    chatterbox_device: str = Field(
        default="auto",
        description="Device for Chatterbox Turbo model ('auto', 'cuda', 'cpu', 'mps')",
    )
    chatterbox_reference_audio: str | None = Field(
        default=None,
        description="Path to reference audio file for voice cloning",
    )
    chatterbox_watermark_key: str | None = Field(
        default=None,
        description="Hex-encoded secret key for Perth audio watermarking (None to disable)",
    )
    chatterbox_num_workers: int = Field(
        default=1,
        ge=1,
        description="Number of concurrent TTS generation threads",
    )

    # ==========================================================================
    # Text-to-Speech (CosyVoice2) — HTTP client settings
    # ==========================================================================
    cosyvoice2_host: str = Field(
        default="localhost",
        description="Hostname of the CosyVoice2 FastAPI server",
    )
    cosyvoice2_port: int = Field(
        default=50000,
        description="Port of the CosyVoice2 FastAPI server",
    )
    cosyvoice2_mode: str = Field(
        default="zero_shot",
        description="Inference mode: 'sft', 'zero_shot', 'cross_lingual', 'instruct2'",
    )
    cosyvoice2_spk_id: str = Field(
        default="中文女",
        description="Speaker ID for CosyVoice2 SFT mode",
    )
    cosyvoice2_prompt_text: str = Field(
        default="",
        description="Default prompt text for CosyVoice2 zero-shot voice cloning",
    )
    cosyvoice2_prompt_wav: str | None = Field(
        default=None,
        description="Path to reference WAV for CosyVoice2 zero-shot voice cloning",
    )
    cosyvoice2_instruct_text: str = Field(
        default="",
        description="Instruction text for CosyVoice2 instruct2 mode",
    )
    cosyvoice2_speed: float = Field(
        default=1.0,
        ge=0.5,
        le=2.0,
        description="CosyVoice2 speech speed multiplier",
    )
    cosyvoice2_timeout: float = Field(
        default=60.0,
        description="HTTP request timeout (seconds) for CosyVoice2 server",
    )

    # TTS provider selection
    tts_provider: str = Field(
        default="chatterbox",
        description="Active TTS provider: 'chatterbox' or 'cosyvoice2'",
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

    # Chroma Cloud settings
    chroma_cloud_url: str | None = Field(
        default=None,
        description="Chroma Cloud instance URL (enables cloud mode when set)",
    )
    chroma_cloud_api_key: str | None = Field(
        default=None,
        description="Chroma Cloud API key for authentication",
    )
    chroma_tenant: str = Field(
        default="default",
        description="Chroma Cloud tenant",
    )
    chroma_database: str = Field(
        default="default",
        description="Chroma Cloud database",
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
    rag_query_timeout_s: float = Field(
        default=2.0,
        ge=0.1,
        description="Maximum seconds to wait for RAG retrieval before proceeding without context",
    )
    rag_cache_ttl_s: float = Field(
        default=300.0,
        ge=0.0,
        description="TTL in seconds for RAG query cache (0 to disable)",
    )
    rag_cache_max_size: int = Field(
        default=128,
        ge=0,
        description="Maximum number of cached RAG query results (0 to disable)",
    )
    rag_chunk_size: int = Field(
        default=1000,
        ge=100,
        description="Default chunk size in characters (or tokens if token splitting is enabled)",
    )
    rag_chunk_overlap: int = Field(
        default=200,
        ge=0,
        description="Default overlap between consecutive chunks",
    )
    rag_use_token_splitting: bool = Field(
        default=False,
        description="Use token-based splitting (tiktoken) instead of character-based",
    )
    rag_token_encoding: str = Field(
        default="cl100k_base",
        description="Tiktoken encoding name for token-based splitting",
    )
    rag_enable_hybrid_retrieval: bool = Field(
        default=False,
        description="Enable hybrid retrieval combining dense (Chroma) and sparse (BM25) search",
    )
    rag_bm25_weight: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Weight for BM25 results in hybrid retrieval (dense weight = 1 - this)",
    )
    rag_deduplication: bool = Field(
        default=True,
        description="Skip adding documents whose IDs already exist in the collection",
    )
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="Model for text embeddings",
    )

    # Observability / tracing
    rag_enable_tracing: bool = Field(
        default=False,
        description="Enable structured retrieval tracing logs with latency metrics",
    )
    langsmith_api_key: str | None = Field(
        default=None,
        description="LangSmith API key for LangChain tracing (optional)",
    )
    langsmith_project: str = Field(
        default="hermes",
        description="LangSmith project name",
    )

    @field_validator("audio_sample_rate")
    @classmethod
    def validate_sample_rate(cls, v: int) -> int:
        """Validate audio sample rate is one of the supported values."""
        valid_rates = [8000, 16000, 22050, 44100, 48000]
        if v not in valid_rates:
            raise ValueError(f"Sample rate must be one of {valid_rates}")
        return v

    @property
    def is_production(self) -> bool:
        """``True`` when running in production."""
        return self.app_env == "production"

    @property
    def is_development(self) -> bool:
        """``True`` when running in development."""
        return self.app_env == "development"


@lru_cache
def get_settings() -> Settings:
    """Return the cached application settings instance."""
    return Settings()
