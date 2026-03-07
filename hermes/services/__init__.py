"""Lazy service exports for Hermes.

Importing a single service package such as ``hermes.services.tts`` should not
eagerly import STT, RAG, and API modules.  Keeping these exports lazy avoids
circular imports during application startup and in tests.
"""

from __future__ import annotations

from importlib import import_module

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
    "ModalRemoteTTSService",
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


_EXPORTS = {
    # STT
    "AbstractSTTService": ("hermes.services.stt", "AbstractSTTService"),
    "DeepgramSTTService": ("hermes.services.stt", "DeepgramSTTService"),
    "MockSTTService": ("hermes.services.stt", "MockSTTService"),
    "STTService": ("hermes.services.stt", "STTService"),
    # LLM
    "AbstractLLMService": ("hermes.services.llm", "AbstractLLMService"),
    "GeminiLLMService": ("hermes.services.llm", "GeminiLLMService"),
    "MockGeminiLLMService": ("hermes.services.llm", "MockGeminiLLMService"),
    "create_function_tool": ("hermes.services.llm", "create_function_tool"),
    # TTS
    "AbstractTTSService": ("hermes.services.tts", "AbstractTTSService"),
    "ChatterboxTTSService": ("hermes.services.tts", "ChatterboxTTSService"),
    "ModalRemoteTTSService": ("hermes.services.tts", "ModalRemoteTTSService"),
    "MockTTSService": ("hermes.services.tts", "MockTTSService"),
    "TTSWorkerPool": ("hermes.services.tts", "TTSWorkerPool"),
    "convert_to_ulaw": ("hermes.services.tts", "convert_to_ulaw"),
    "resample_to_8khz": ("hermes.services.tts", "resample_to_8khz"),
    # RAG
    "AbstractRAGService": ("hermes.services.rag", "AbstractRAGService"),
    "BM25Retriever": ("hermes.services.rag", "BM25Retriever"),
    "ChromaRAGService": ("hermes.services.rag", "ChromaRAGService"),
    "QueryCache": ("hermes.services.rag", "QueryCache"),
    "RAGService": ("hermes.services.rag", "RAGService"),
    "TextSplitter": ("hermes.services.rag", "TextSplitter"),
    "reciprocal_rank_fusion": ("hermes.services.rag", "reciprocal_rank_fusion"),
}


def __getattr__(name: str):
    """Load service exports on first access."""
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = _EXPORTS[name]
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
