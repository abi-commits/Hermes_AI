"""External service integrations for Hermes.

Each service domain is a sub-package with an ABC base and one or more
backend implementations:

* :mod:`hermes.services.stt`  — Speech-to-Text (Deepgram)
* :mod:`hermes.services.llm`  — LLM inference (Gemini)
* :mod:`hermes.services.tts`  — Text-to-Speech (Chatterbox Streaming)
* :mod:`hermes.services.rag`  — Retrieval-Augmented Generation (Chroma Cloud)
"""

# --- STT ---
from hermes.services.stt import (
    AbstractSTTService,
    DeepgramSTTService,
    MockSTTService,
    STTService,
)

# --- LLM ---
from hermes.services.llm import (
    AbstractLLMService,
    GeminiLLMService,
    MockGeminiLLMService,
    create_function_tool,
)

# --- TTS ---
from hermes.services.tts import (
    AbstractTTSService,
    ChatterboxTTSService,
    MockTTSService,
    TTSWorkerPool,
    convert_to_ulaw,
    resample_to_8khz,
)

# --- RAG ---
from hermes.services.rag import (
    AbstractRAGService,
    BM25Retriever,
    ChromaRAGService,
    QueryCache,
    RAGService,
    TextSplitter,
    reciprocal_rank_fusion,
)

__all__ = [
    # STT
    "AbstractSTTService",
    "DeepgramSTTService",
    "MockSTTService",
    "STTService",
    # LLM
    "AbstractLLMService",
    "GeminiLLMService",
    "MockGeminiLLMService",
    "create_function_tool",
    # TTS
    "AbstractTTSService",
    "ChatterboxTTSService",
    "MockTTSService",
    "TTSWorkerPool",
    "convert_to_ulaw",
    "resample_to_8khz",
    # RAG
    "AbstractRAGService",
    "BM25Retriever",
    "ChromaRAGService",
    "QueryCache",
    "RAGService",
    "TextSplitter",
    "reciprocal_rank_fusion",
]
