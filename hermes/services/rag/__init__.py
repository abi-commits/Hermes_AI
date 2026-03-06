"""RAG sub-package.

Public surface
--------------
:class:`ChromaRAGService`   — production Chroma Cloud implementation
:class:`AbstractRAGService` — ABC for mocking / alternative backends
"""

from hermes.services.rag.base import AbstractRAGService
from hermes.services.rag.bm25 import BM25Retriever, reciprocal_rank_fusion
from hermes.services.rag.cache import QueryCache
from hermes.services.rag.chroma import ChromaRAGService
from hermes.services.rag.splitter import TextSplitter

# Convenience alias
RAGService = ChromaRAGService

__all__ = [
    "AbstractRAGService",
    "BM25Retriever",
    "ChromaRAGService",
    "QueryCache",
    "RAGService",
    "TextSplitter",
    "reciprocal_rank_fusion",
]
