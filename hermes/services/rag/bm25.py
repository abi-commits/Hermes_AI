"""BM25 sparse retriever and Reciprocal Rank Fusion for hybrid RAG."""

from __future__ import annotations

from typing import Any

import structlog

logger = structlog.get_logger(__name__)


class BM25Retriever:
    """In-memory BM25 sparse retriever; index is rebuilt lazily when dirty."""

    def __init__(self, weight: float = 0.5) -> None:
        """Initialise an empty BM25 retriever."""
        self.weight = weight
        self._corpus: list[str] = []
        self._ids: list[str] = []
        self._index: Any = None  # BM25Okapi instance, built lazily
        self._dirty = True
        self._logger = structlog.get_logger(__name__)

    # ------------------------------------------------------------------
    # Index lifecycle
    # ------------------------------------------------------------------

    def mark_dirty(self) -> None:
        """Mark the index as stale; it will be rebuilt on the next query."""
        self._dirty = True

    def build_index(self, corpus: list[str], ids: list[str]) -> None:
        """(Re)build the BM25 index from *corpus*."""
        self._corpus = corpus
        self._ids = ids
        self._index = None

        if corpus:
            try:
                from rank_bm25 import BM25Okapi

                tokenized = [doc.lower().split() for doc in corpus]
                self._index = BM25Okapi(tokenized)
                self._logger.debug("bm25_index_built", corpus_size=len(corpus))
            except ImportError:
                self._logger.warning(
                    "rank_bm25_not_installed",
                    hint="Install rank-bm25 for hybrid retrieval: uv add rank-bm25",
                )

        self._dirty = False

    def add_to_index(self, texts: list[str], ids: list[str]) -> None:
        """Incrementally add *texts*/*ids* to the in-memory corpus and rebuild the index.

        This avoids a full Chroma round-trip when the index has already been seeded.
        """
        self._corpus.extend(texts)
        self._ids.extend(ids)
        self.build_index(self._corpus, self._ids)
        self._logger.debug(
            "bm25_incremental_add", added=len(texts), total=len(self._corpus)
        )

    def remove_from_index(self, ids: list[str]) -> None:
        """Remove documents by *ids* from the in-memory corpus and rebuild the index.

        This avoids a full Chroma round-trip when the index has already been seeded.
        """
        id_set = set(ids)
        pairs = [
            (text, doc_id)
            for text, doc_id in zip(self._corpus, self._ids)
            if doc_id not in id_set
        ]
        if pairs:
            new_corpus, new_ids = zip(*pairs)
            self.build_index(list(new_corpus), list(new_ids))
        else:
            self.build_index([], [])
        self._logger.debug(
            "bm25_incremental_remove", removed=len(ids), total=len(self._corpus)
        )

    @property
    def needs_rebuild(self) -> bool:
        """``True`` when the index must be rebuilt before the next query."""
        return self._dirty or not self._corpus

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(self, query: str, k: int) -> list[str]:
        """Return the top-*k* documents scored by BM25, or ``[]`` if index is empty."""
        if not self._corpus or self._index is None:
            return []

        scores = self._index.get_scores(query.lower().split())
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [self._corpus[i] for i in ranked if scores[i] > 0]


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion
# ---------------------------------------------------------------------------


def reciprocal_rank_fusion(
    dense: list[str],
    sparse: list[str],
    k: int,
    rrf_k: int = 60,
) -> list[str]:
    """Merge two ranked document lists using Reciprocal Rank Fusion."""
    scores: dict[str, float] = {}

    for rank, doc in enumerate(dense, start=1):
        scores[doc] = scores.get(doc, 0.0) + 1.0 / (rrf_k + rank)

    for rank, doc in enumerate(sparse, start=1):
        scores[doc] = scores.get(doc, 0.0) + 1.0 / (rrf_k + rank)

    return sorted(scores, key=scores.__getitem__, reverse=True)[:k]
