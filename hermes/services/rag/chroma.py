"""Chroma Cloud RAG service with query caching, hybrid retrieval, and text splitting."""

from __future__ import annotations

import asyncio
import hashlib
import time
from concurrent.futures import Executor
from typing import TYPE_CHECKING, Any

import structlog

from config import get_settings
from hermes.core.exceptions import RAGError, RAGRetrievalError
from hermes.services.rag.base import AbstractRAGService
from hermes.services.rag.bm25 import BM25Retriever, reciprocal_rank_fusion
from hermes.services.rag.cache import QueryCache
from hermes.services.rag.splitter import TextSplitter

if TYPE_CHECKING:
    from chromadb.api import ClientAPI
    from chromadb.api.models.Collection import Collection
    from chromadb.api.types import Metadata

logger = structlog.get_logger(__name__)


def _content_hash(text: str) -> str:
    """Return a 32-char hex SHA-256 hash of *text* for use as a document ID."""
    return hashlib.sha256(text.encode()).hexdigest()[:32]


class ChromaRAGService(AbstractRAGService):
    """RAG service backed by Chroma Cloud with lazy connection, query caching, and hybrid retrieval."""

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
        """Initialise the service; connection is deferred until first use.

        All parameters fall back to application settings when ``None``.
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

        # Feature flags
        self._enable_hybrid: bool = settings.rag_enable_hybrid_retrieval
        self._deduplication: bool = settings.rag_deduplication
        self._enable_tracing: bool = settings.rag_enable_tracing

        # Composed helpers
        self._cache = QueryCache(
            ttl_s=settings.rag_cache_ttl_s,
            max_size=settings.rag_cache_max_size,
        )
        self._splitter = TextSplitter(
            chunk_size=settings.rag_chunk_size,
            chunk_overlap=settings.rag_chunk_overlap,
            use_token_splitting=settings.rag_use_token_splitting,
            token_encoding=settings.rag_token_encoding,
        )
        self._bm25 = BM25Retriever(weight=settings.rag_bm25_weight)

        # Chroma client (created lazily on first call)
        self._client: ClientAPI | None = None
        self._collection: Collection | None = None

        self._logger = structlog.get_logger(__name__)
        self._logger.info(
            "chroma_rag_service_initialised",
            collection=self.collection_name,
            url=self.chroma_url,
            hybrid=self._enable_hybrid,
        )

    # ------------------------------------------------------------------
    # Connection (lazy)
    # ------------------------------------------------------------------

    async def _ensure_client(self) -> None:
        """Create the Chroma Cloud client and collection on first access."""
        if self._collection is not None:
            return

        loop = asyncio.get_running_loop()
        try:
            import chromadb

            if not self.chroma_url:
                raise RAGError(
                    "chroma_cloud_url must be set (via argument or CHROMA_CLOUD_URL env var)"
                )

            chroma_url: str = self.chroma_url  # narrowed — checked above

            self._client = await loop.run_in_executor(
                self._executor,
                lambda: chromadb.CloudClient(
                    tenant=self.tenant,
                    database=self.database,
                    api_key=self.chroma_api_key,
                    cloud_host=chroma_url,
                    cloud_port=443,
                    enable_ssl=True,
                ),
            )

            client = self._client
            if client is None:
                raise RAGError("Chroma client initialisation returned None")

            self._collection = await loop.run_in_executor(
                self._executor,
                lambda: client.get_or_create_collection(name=self.collection_name),
            )
            self._logger.info(
                "chroma_collection_connected", collection=self.collection_name
            )

        except ImportError:
            raise RAGError("chromadb package is not installed")
        except RAGError:
            raise
        except Exception as exc:
            self._logger.exception("chroma_connection_failed")
            raise RAGRetrievalError(f"Chroma connection error: {exc}") from exc

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def warm_up(self) -> None:
        """Connect to Chroma eagerly to eliminate cold-start latency."""
        t0 = time.perf_counter()
        await self._ensure_client()
        self._logger.info(
            "rag_warm_up_complete",
            elapsed_ms=round((time.perf_counter() - t0) * 1_000, 1),
            collection=self.collection_name,
        )

    def set_executor(self, executor: Executor) -> None:
        """Set the thread-pool executor (call during app startup)."""
        self._executor = executor

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    async def retrieve(
        self,
        query: str,
        k: int | None = None,
        where: dict[str, Any] | None = None,
    ) -> list[str]:
        """Retrieve the top-*k* chunks for *query*, with caching and optional hybrid scoring."""
        k = k or self.num_results

        # --- Cache check ---
        cache_key = self._cache.build_key(query, k, where)
        cached = self._cache.get(cache_key)
        if cached is not None:
            self._logger.debug("rag_cache_hit", query=query[:50], count=len(cached))
            return cached

        t0 = time.perf_counter()

        dense_docs = await self._dense_retrieve(query, k=k, where=where)

        if self._enable_hybrid:
            await self._ensure_bm25_corpus()
            sparse_docs = self._bm25.retrieve(query, k=k)
            final_docs = reciprocal_rank_fusion(dense_docs, sparse_docs, k=k)
        else:
            final_docs = dense_docs

        elapsed_ms = (time.perf_counter() - t0) * 1_000
        self._cache.put(cache_key, final_docs)

        if self._enable_tracing:
            self._logger.info(
                "rag_retrieve_trace",
                query=query[:80],
                k=k,
                where=where,
                count=len(final_docs),
                elapsed_ms=round(elapsed_ms, 2),
                hybrid=self._enable_hybrid,
            )
        else:
            self._logger.debug("rag_retrieved", query=query[:50], count=len(final_docs))

        return final_docs

    async def retrieve_with_timeout(
        self,
        query: str,
        k: int | None = None,
        where: dict[str, Any] | None = None,
        timeout_s: float | None = None,
    ) -> list[str]:
        """Retrieve with a time budget; cache hits return instantly."""
        if timeout_s is None:
            timeout_s = get_settings().rag_query_timeout_s

        try:
            return await asyncio.wait_for(
                self.retrieve(query, k=k, where=where),
                timeout=timeout_s,
            )
        except asyncio.TimeoutError:
            self._logger.warning(
                "rag_retrieve_timeout", query=query[:50], timeout_s=timeout_s
            )
            return []

    # ------------------------------------------------------------------
    # Dense retrieval (Chroma)
    # ------------------------------------------------------------------

    async def _dense_retrieve(
        self,
        query: str,
        k: int,
        where: dict[str, Any] | None = None,
    ) -> list[str]:
        """Query Chroma for the top-*k* dense (embedding) results."""
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
        except Exception as exc:
            self._logger.error("chroma_query_failed", error=str(exc))
            raise RAGRetrievalError(f"Dense retrieval error: {exc}") from exc

        docs_outer: list[list[str]] = results.get("documents") or [[]]
        return docs_outer[0]

    # ------------------------------------------------------------------
    # BM25 corpus loading
    # ------------------------------------------------------------------

    async def _ensure_bm25_corpus(self) -> None:
        """Seed the BM25 index from Chroma on cold start (index empty or dirty).

        Incremental updates via :py:meth:`add_documents` and
        :py:meth:`delete_documents` keep the index current after the first
        build, so this full-collection fetch only runs once per process.
        """
        if not self._bm25.needs_rebuild:
            return

        await self._ensure_client()
        if self._collection is None:
            raise RAGError("Collection not initialised")

        collection = self._collection
        loop = asyncio.get_running_loop()

        self._logger.info(
            "bm25_cold_start_rebuild",
            collection=self.collection_name,
        )
        all_data = await loop.run_in_executor(
            self._executor,
            lambda: collection.get(include=["documents"]),
        )
        corpus: list[str] = all_data.get("documents") or []
        ids: list[str] = all_data.get("ids") or []
        self._bm25.build_index(corpus, ids)

    # ------------------------------------------------------------------
    # Document management
    # ------------------------------------------------------------------

    async def add_documents(
        self,
        texts: list[str],
        ids: list[str] | None = None,
        metadatas: list["Metadata"] | None = None,
    ) -> list[str]:
        """Add *texts* to the collection, skipping duplicates if deduplication is enabled."""
        await self._ensure_client()
        if self._collection is None:
            raise RAGError("Collection not initialised")

        if ids is None:
            ids = [_content_hash(t) for t in texts]

        # --- Deduplication ---
        if self._deduplication:
            existing = await self._existing_ids(ids)
            if existing:
                keep = [
                    (t, i, m)
                    for t, i, m in zip(
                        texts, ids, metadatas or [None] * len(texts)  # type: ignore[list-item]
                    )
                    if i not in existing
                ]
                skipped = len(texts) - len(keep)
                if skipped:
                    self._logger.info("dedup_skipped", skipped=skipped)
                if not keep:
                    return []
                texts = [x[0] for x in keep]
                ids = [x[1] for x in keep]
                metadatas = [x[2] for x in keep] if metadatas else None  # type: ignore[assignment]

        collection = self._collection
        loop = asyncio.get_running_loop()

        try:
            await loop.run_in_executor(
                self._executor,
                lambda: collection.add(documents=texts, ids=ids, metadatas=metadatas),
            )
            self._logger.info("documents_added", count=len(texts))
        except Exception as exc:
            self._logger.error("document_addition_failed", error=str(exc))
            raise RAGRetrievalError(f"Document addition error: {exc}") from exc

        # Keep the BM25 index current without a full Chroma round-trip.
        # If the index has not yet been seeded (cold start), keep it dirty so
        # _ensure_bm25_corpus will do exactly one full fetch on the next query.
        if self._enable_hybrid:
            if self._bm25.needs_rebuild:
                self._bm25.mark_dirty()  # will be rebuilt in full on first query
            else:
                self._bm25.add_to_index(texts, ids)  # incremental — no Chroma fetch
        self._cache.invalidate()
        return ids

    async def delete_documents(self, ids: list[str]) -> None:
        """Remove documents by ID."""
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
        except Exception as exc:
            self._logger.error("document_deletion_failed", error=str(exc))
            raise RAGRetrievalError(f"Document deletion error: {exc}") from exc

        # Mirror the deletion in the in-memory corpus without a Chroma round-trip.
        if self._enable_hybrid:
            if self._bm25.needs_rebuild:
                self._bm25.mark_dirty()
            else:
                self._bm25.remove_from_index(ids)
        self._cache.invalidate()

    async def split_and_add_documents(
        self,
        texts: list[str],
        metadatas: list[dict[str, Any]] | None = None,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> list[str]:
        """Split *texts* into chunks then add them to the collection."""
        all_chunks, all_metas = self._splitter.split_many(
            texts,
            metadatas=metadatas,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        self._logger.info(
            "split_and_add",
            input_documents=len(texts),
            chunks=len(all_chunks),
        )
        return await self.add_documents(texts=all_chunks, metadatas=all_metas)  # type: ignore[arg-type]

    # ------------------------------------------------------------------
    # Context formatting
    # ------------------------------------------------------------------

    def format_context(self, documents: list[str]) -> str:
        """Format retrieved chunks as a numbered context string for LLM injection."""
        if not documents:
            return ""

        lines = ["Relevant information:"]
        for i, doc in enumerate(documents, 1):
            lines.append(f"\n[{i}] {doc}")
        return "\n".join(lines)

    async def query_with_context(
        self,
        query: str,
        k: int | None = None,
        where: dict[str, Any] | None = None,
    ) -> tuple[str, list[str]]:
        """Retrieve and return documents as ``(formatted_context, raw_documents)``."""
        docs = await self.retrieve(query, k=k, where=where)
        return self.format_context(docs), docs

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    async def get_collection_stats(self) -> dict[str, Any]:
        """Return ``{name, count}`` stats for the connected collection."""
        await self._ensure_client()
        if self._collection is None:
            raise RAGError("Collection not initialised")

        collection = self._collection
        loop = asyncio.get_running_loop()
        count: int = await loop.run_in_executor(self._executor, collection.count)
        return {"name": self.collection_name, "count": count}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _existing_ids(self, ids: list[str]) -> set[str]:
        """Return the subset of *ids* already present in the collection."""
        if self._collection is None:
            return set()
        collection = self._collection
        loop = asyncio.get_running_loop()
        try:
            result = await loop.run_in_executor(
                self._executor,
                lambda: collection.get(ids=ids, include=[]),
            )
            return set(result.get("ids", []))
        except Exception as exc:
            self._logger.warning("dedup_check_failed", error=str(exc))
            return set()

    def invalidate_cache(self) -> None:
        """Flush the query cache (e.g. after external document changes)."""
        self._cache.invalidate()
        self._logger.debug("rag_cache_invalidated")

    def invalidate_bm25_index(self) -> None:
        """Mark the BM25 index stale so it rebuilds on the next hybrid query."""
        self._bm25.mark_dirty()
