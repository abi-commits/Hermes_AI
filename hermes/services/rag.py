"""
RAG Service Module for Hermes.

This module provides a wrapper around Chroma Cloud for retrieval-augmented
generation.  It uses Chroma's built-in embedding models, so no local embedding
computation is needed.  All synchronous Chroma calls are offloaded to a thread
pool to keep the asyncio event loop responsive.

Improvements over the base implementation:
  - Token-based text splitting via tiktoken
  - Document deduplication on add (skip existing IDs)
  - Hybrid retrieval combining dense (Chroma) and sparse (BM25) search
  - Rich metadata filtering at query time
  - Structured observability / retrieval tracing
"""

from __future__ import annotations

import asyncio
import hashlib
import time
from concurrent.futures import Executor
from typing import TYPE_CHECKING, Any

import structlog

from config import get_settings
from hermes.core.exceptions import RAGError, RAGRetrievalError

if TYPE_CHECKING:
    from chromadb.api import ClientAPI
    from chromadb.api.models.Collection import Collection
    from chromadb.api.types import Metadata

logger = structlog.get_logger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


def _content_hash(text: str) -> str:
    """Return a deterministic 32-char hex hash for *text*."""
    return hashlib.sha256(text.encode()).hexdigest()[:32]


# ──────────────────────────────────────────────────────────────────────────────
# Service
# ──────────────────────────────────────────────────────────────────────────────


class ChromaRAGService:
    """Service for retrieving context from a Chroma Cloud vector database.

    The service connects to a remote Chroma instance and leverages its
    built-in embedding models (no local embedding required).  All Chroma
    calls are synchronous and are offloaded to a thread pool to avoid
    blocking the asyncio event loop.
    """

    def __init__(
        self,
        collection_name: str | None = None,
        chroma_url: str | None = None,
        chroma_api_key: str | None = None,
        tenant: str | None = None,
        database: str | None = None,
        num_results: int | None = None,
        executor: Executor | None = None,
    ) -> None:
        """Initialise the Chroma RAG service.

        When arguments are *None* the corresponding value is read from
        application settings so that callers can simply do
        ``ChromaRAGService()`` during normal operation.
        """
        settings = get_settings()

        self.collection_name = collection_name or settings.chromadb_collection
        self.chroma_url = chroma_url or settings.chroma_cloud_url
        self.chroma_api_key = chroma_api_key or settings.chroma_cloud_api_key
        self.tenant = tenant or settings.chroma_tenant
        self.database = database or settings.chroma_database
        self.num_results = num_results or settings.rag_top_k
        self.similarity_threshold = settings.rag_similarity_threshold
        self._executor = executor

        # Feature flags from settings
        self._use_token_splitting = settings.rag_use_token_splitting
        self._token_encoding = settings.rag_token_encoding
        self._enable_hybrid = settings.rag_enable_hybrid_retrieval
        self._bm25_weight = settings.rag_bm25_weight
        self._deduplication = settings.rag_deduplication
        self._enable_tracing = settings.rag_enable_tracing
        self._chunk_size = settings.rag_chunk_size
        self._chunk_overlap = settings.rag_chunk_overlap

        # Chroma client (created lazily)
        self._client: ClientAPI | None = None
        self._collection: Collection | None = None

        # BM25 index (built lazily when hybrid retrieval is enabled)
        self._bm25_corpus: list[str] = []
        self._bm25_ids: list[str] = []
        self._bm25_index: Any = None  # cached BM25Okapi instance
        self._bm25_dirty = True  # needs rebuild

        self._logger = structlog.get_logger(__name__)

        self._logger.info(
            "chroma_rag_service_initialised",
            collection=self.collection_name,
            url=self.chroma_url,
            hybrid=self._enable_hybrid,
            token_splitting=self._use_token_splitting,
            deduplication=self._deduplication,
            tracing=self._enable_tracing,
        )

    # ------------------------------------------------------------------
    # Connection helpers
    # ------------------------------------------------------------------

    async def _ensure_client(self) -> None:
        """Ensure the Chroma client and collection are connected (lazy init)."""
        if self._collection is not None:
            return

        loop = asyncio.get_running_loop()
        try:
            import chromadb

            if not self.chroma_url:
                raise RAGError(
                    "chroma_cloud_url must be set (via argument or CHROMA_CLOUD_URL env var)"
                )

            cloud_host: str = self.chroma_url

            self._client = await loop.run_in_executor(
                self._executor,
                lambda: chromadb.CloudClient(
                    tenant=self.tenant,
                    database=self.database,
                    api_key=self.chroma_api_key,
                    cloud_host=cloud_host,
                    cloud_port=443,
                    enable_ssl=True,
                ),
            )

            client = self._client
            if client is None:
                raise RAGError("Chroma client initialisation returned None")

            self._collection = await loop.run_in_executor(
                self._executor,
                lambda: client.get_or_create_collection(
                    name=self.collection_name,
                ),
            )
            self._logger.info(
                "chroma_collection_connected",
                collection=self.collection_name,
            )
        except ImportError:
            raise RAGError("chromadb package is not installed")
        except RAGError:
            raise
        except Exception as e:
            self._logger.exception("chroma_connection_failed")
            raise RAGRetrievalError(f"Chroma connection error: {e}")

    # ------------------------------------------------------------------
    # Text splitting
    # ------------------------------------------------------------------

    def _build_splitter(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> Any:
        """Build a LangChain text splitter respecting the token-splitting flag.

        When ``rag_use_token_splitting`` is *True*, uses
        ``RecursiveCharacterTextSplitter.from_tiktoken_encoder`` so that
        chunk boundaries are measured in **tokens** rather than characters.

        Args:
            chunk_size: Override chunk size (default from settings).
            chunk_overlap: Override chunk overlap (default from settings).

        Returns:
            A configured ``RecursiveCharacterTextSplitter`` instance.
        """
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        size = chunk_size or self._chunk_size
        overlap = chunk_overlap or self._chunk_overlap

        if self._use_token_splitting:
            return RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                encoding_name=self._token_encoding,
                chunk_size=size,
                chunk_overlap=overlap,
                separators=["\n\n", "\n", ". ", " ", ""],
            )

        return RecursiveCharacterTextSplitter(
            chunk_size=size,
            chunk_overlap=overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    # ------------------------------------------------------------------
    # Deduplication
    # ------------------------------------------------------------------

    async def _existing_ids(self, ids: list[str]) -> set[str]:
        """Return the subset of *ids* that already exist in the collection.

        Uses Chroma's ``get(ids=...)`` to check existence without downloading
        full embeddings.
        """
        if not self._deduplication:
            return set()

        await self._ensure_client()
        if self._collection is None:
            raise RAGError("Collection not initialised")
        collection = self._collection

        loop = asyncio.get_running_loop()
        try:
            result = await loop.run_in_executor(
                self._executor,
                lambda: collection.get(ids=ids, include=[]),
            )
            return set(result.get("ids", []))
        except Exception as e:
            self._logger.warning("dedup_check_failed", error=str(e))
            return set()  # proceed without dedup on failure

    # ------------------------------------------------------------------
    # Retrieval  — dense (Chroma)
    # ------------------------------------------------------------------

    async def retrieve(
        self,
        query: str,
        k: int | None = None,
        where: dict[str, Any] | None = None,
    ) -> list[str]:
        """Retrieve relevant document chunks for a query.

        If hybrid retrieval is enabled, results from Chroma (dense) and
        BM25 (sparse) are fused using Reciprocal Rank Fusion.

        Args:
            query: The user's query text.
            k: Number of results to return (defaults to ``self.num_results``).
            where: Optional Chroma metadata filter (e.g. ``{"category": "faq"}``).

        Returns:
            List of document chunk texts, ranked by relevance.
        """
        k = k or self.num_results
        t0 = time.perf_counter()

        dense_docs = await self._dense_retrieve(query, k=k, where=where)

        if self._enable_hybrid:
            sparse_docs = await self._bm25_retrieve(query, k=k)
            final_docs = self._reciprocal_rank_fusion(
                dense_docs, sparse_docs, k=k,
            )
        else:
            final_docs = dense_docs

        elapsed_ms = (time.perf_counter() - t0) * 1000

        if self._enable_tracing:
            self._logger.info(
                "rag_retrieve_trace",
                query=query[:80],
                k=k,
                where=where,
                results_count=len(final_docs),
                elapsed_ms=round(elapsed_ms, 2),
                hybrid=self._enable_hybrid,
            )
        else:
            self._logger.debug(
                "rag_retrieved",
                query=query[:50],
                results_count=len(final_docs),
            )

        return final_docs

    async def _dense_retrieve(
        self,
        query: str,
        k: int,
        where: dict[str, Any] | None = None,
    ) -> list[str]:
        """Retrieve from Chroma (dense / embedding-based)."""
        await self._ensure_client()
        if self._collection is None:
            raise RAGError("Collection not initialised")
        collection = self._collection

        loop = asyncio.get_running_loop()
        try:
            results = await loop.run_in_executor(
                self._executor,
                lambda: collection.query(
                    query_texts=[query],
                    n_results=k,
                    where=where,
                ),
            )
        except Exception as e:
            self._logger.error("chroma_query_failed", error=str(e))
            raise RAGRetrievalError(f"Dense retrieval error: {e}")

        docs_outer = results.get("documents") or [[]]
        return docs_outer[0]

    # ------------------------------------------------------------------
    # Retrieval  — sparse (BM25)
    # ------------------------------------------------------------------

    async def _ensure_bm25_index(self) -> None:
        """Build / rebuild the in-memory BM25 index from the collection."""
        if not self._bm25_dirty and self._bm25_corpus:
            return

        await self._ensure_client()
        if self._collection is None:
            raise RAGError("Collection not initialised")
        collection = self._collection

        loop = asyncio.get_running_loop()

        # Fetch all documents from collection (for reasonable-sized corpora).
        all_data = await loop.run_in_executor(
            self._executor,
            lambda: collection.get(include=["documents"]),
        )

        self._bm25_ids = all_data.get("ids", [])
        self._bm25_corpus = all_data.get("documents", []) or []

        # Build and cache the BM25 index so it is not rebuilt per query.
        if self._bm25_corpus:
            from rank_bm25 import BM25Okapi

            tokenized = [doc.lower().split() for doc in self._bm25_corpus]
            self._bm25_index = BM25Okapi(tokenized)
        else:
            self._bm25_index = None

        self._bm25_dirty = False

        self._logger.debug(
            "bm25_index_built",
            corpus_size=len(self._bm25_corpus),
        )

    async def _bm25_retrieve(self, query: str, k: int) -> list[str]:
        """Retrieve using BM25 sparse scoring."""
        try:
            await self._ensure_bm25_index()
        except ImportError:
            self._logger.warning(
                "rank_bm25_not_installed",
                hint="Install rank-bm25 for hybrid retrieval: uv add rank-bm25",
            )
            return []

        if not self._bm25_corpus or self._bm25_index is None:
            return []

        query_tokens = query.lower().split()
        scores = self._bm25_index.get_scores(query_tokens)

        # Get top-k indices
        ranked_indices = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True,
        )[:k]
        return [self._bm25_corpus[i] for i in ranked_indices if scores[i] > 0]

    # ------------------------------------------------------------------
    # Reciprocal Rank Fusion
    # ------------------------------------------------------------------

    @staticmethod
    def _reciprocal_rank_fusion(
        dense: list[str],
        sparse: list[str],
        k: int,
        rrf_k: int = 60,
    ) -> list[str]:
        """Merge two ranked lists using Reciprocal Rank Fusion (RRF).

        RRF score = sum  1 / (rrf_k + rank)  for each list the doc appears in.

        Args:
            dense: Documents from dense retrieval (ordered by relevance).
            sparse: Documents from sparse retrieval (ordered by relevance).
            k: Number of final results to return.
            rrf_k: RRF constant (default 60, per the original paper).

        Returns:
            Merged and re-ranked document list.
        """
        scores: dict[str, float] = {}

        for rank, doc in enumerate(dense, start=1):
            scores[doc] = scores.get(doc, 0.0) + 1.0 / (rrf_k + rank)

        for rank, doc in enumerate(sparse, start=1):
            scores[doc] = scores.get(doc, 0.0) + 1.0 / (rrf_k + rank)

        ranked = sorted(scores, key=scores.__getitem__, reverse=True)
        return ranked[:k]

    # ------------------------------------------------------------------
    # Context formatting helpers
    # ------------------------------------------------------------------

    def format_context(self, documents: list[str]) -> str:
        """Format retrieved document chunks as a context string for the LLM.

        Args:
            documents: List of retrieved document texts.

        Returns:
            Formatted context string.
        """
        if not documents:
            return ""

        context_parts = ["Relevant information:"]
        for i, doc in enumerate(documents, 1):
            context_parts.append(f"\n[{i}] {doc}")

        return "\n".join(context_parts)

    async def query_with_context(
        self,
        query: str,
        k: int | None = None,
        where: dict[str, Any] | None = None,
    ) -> tuple[str, list[str]]:
        """Retrieve documents and return them as a formatted context string.

        Convenience helper that combines :meth:`retrieve` and
        :meth:`format_context` for use with an LLM.

        Args:
            query: The user query.
            k: Number of results to return.
            where: Optional metadata filter.

        Returns:
            Tuple of (formatted context, raw document list).
        """
        documents = await self.retrieve(query, k=k, where=where)
        context = self.format_context(documents)
        return context, documents

    # ------------------------------------------------------------------
    # Document management
    # ------------------------------------------------------------------

    async def add_documents(
        self,
        texts: list[str],
        ids: list[str] | None = None,
        metadatas: list[Metadata] | None = None,
    ) -> list[str]:
        """Add documents to the collection with optional deduplication.

        When ``rag_deduplication`` is enabled, IDs that already exist in the
        collection are silently skipped (upsert-like behaviour without
        overwriting).

        Args:
            texts: List of document texts.
            ids: Optional list of unique IDs.  If *None*, content-hash IDs
                 are generated for deterministic deduplication.
            metadatas: Optional list of metadata dicts.

        Returns:
            List of document IDs that were actually added.
        """
        await self._ensure_client()
        if self._collection is None:
            raise RAGError("Collection not initialised")
        collection = self._collection

        if ids is None:
            ids = [_content_hash(t) for t in texts]

        # ── Deduplication ────────────────────────────────────────────
        if self._deduplication:
            existing = await self._existing_ids(ids)
            if existing:
                keep = [
                    (t, i, m)
                    for t, i, m in zip(
                        texts,
                        ids,
                        metadatas or [None] * len(texts),  # type: ignore[list-item]
                    )
                    if i not in existing
                ]
                skipped = len(texts) - len(keep)
                if skipped:
                    self._logger.info(
                        "dedup_skipped",
                        skipped=skipped,
                        remaining=len(keep),
                    )
                if not keep:
                    return []
                texts = [k[0] for k in keep]
                ids = [k[1] for k in keep]
                metadatas = (
                    [k[2] for k in keep] if metadatas else None  # type: ignore[assignment]
                )

        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(
                self._executor,
                lambda: collection.add(
                    documents=texts,
                    ids=ids,
                    metadatas=metadatas,
                ),
            )
            self._logger.info(
                "documents_added",
                count=len(texts),
                collection=self.collection_name,
            )
        except Exception as e:
            self._logger.error("document_addition_failed", error=str(e))
            raise RAGRetrievalError(f"Document addition error: {e}")

        # Mark BM25 index as stale
        self._bm25_dirty = True

        return ids

    async def delete_documents(self, ids: list[str]) -> None:
        """Delete documents by ID.

        Args:
            ids: List of document IDs to delete.
        """
        await self._ensure_client()
        if self._collection is None:
            raise RAGError("Collection not initialised")
        collection = self._collection

        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(
                self._executor,
                lambda: collection.delete(ids=ids),
            )
            self._logger.info("documents_deleted", count=len(ids))
            self._bm25_dirty = True
        except Exception as e:
            self._logger.error("document_deletion_failed", error=str(e))
            raise RAGRetrievalError(f"Document deletion error: {e}")

    # ------------------------------------------------------------------
    # Lifecycle helpers
    # ------------------------------------------------------------------

    async def split_and_add_documents(
        self,
        texts: list[str],
        metadatas: list[dict[str, Any]] | None = None,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> list[str]:
        """Split texts using LangChain and add resulting chunks to the collection.

        Respects the ``rag_use_token_splitting`` setting:  when enabled, chunks
        are measured in **tokens** (via tiktoken) instead of characters.

        Args:
            texts: Raw document texts (one string per document).
            metadatas: Optional parallel list of metadata dicts (one per text).
            chunk_size: Override chunk size (default from settings).
            chunk_overlap: Override chunk overlap (default from settings).

        Returns:
            List of IDs for all chunks that were added.
        """
        splitter = self._build_splitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap,
        )

        all_chunks: list[str] = []
        all_metas: list[dict[str, Any]] = []

        for i, text in enumerate(texts):
            chunks = splitter.split_text(text)
            base_meta = metadatas[i] if metadatas and i < len(metadatas) else {}
            for ci, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                all_metas.append({
                    **base_meta,
                    "chunk_index": ci,
                    "total_chunks": len(chunks),
                })

        self._logger.info(
            "split_and_add",
            input_documents=len(texts),
            chunks=len(all_chunks),
            token_splitting=self._use_token_splitting,
        )

        return await self.add_documents(
            texts=all_chunks,
            metadatas=all_metas,  # type: ignore[arg-type]
        )

    def set_executor(self, executor: Executor) -> None:
        """Set the thread pool executor (typically called during app startup).

        Args:
            executor: A :class:`concurrent.futures.Executor` instance.
        """
        self._executor = executor

    async def get_collection_stats(self) -> dict[str, Any]:
        """Return basic statistics about the connected collection.

        Returns:
            Dictionary with collection name and document count.
        """
        await self._ensure_client()
        if self._collection is None:
            raise RAGError("Collection not initialised")
        collection = self._collection

        loop = asyncio.get_running_loop()
        count = await loop.run_in_executor(
            self._executor,
            collection.count,
        )
        return {
            "name": self.collection_name,
            "count": count,
        }

    def invalidate_bm25_index(self) -> None:
        """Mark the BM25 index as stale so it is rebuilt on next hybrid query."""
        self._bm25_dirty = True
