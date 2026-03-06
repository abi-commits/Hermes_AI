"""LRU + TTL query-result cache for the RAG pipeline."""

from __future__ import annotations

import hashlib
import json
import time
from collections import OrderedDict
from typing import Any


class QueryCache:
    """TTL + LRU cache for RAG query results."""

    def __init__(self, ttl_s: float = 300.0, max_size: int = 256) -> None:
        """Initialise an empty cache."""
        self._ttl_s = ttl_s
        self._max_size = max_size
        # key → (insert_time, results)
        self._store: OrderedDict[str, tuple[float, list[str]]] = OrderedDict()

    # ------------------------------------------------------------------
    # Key generation
    # ------------------------------------------------------------------

    @staticmethod
    def build_key(query: str, k: int, where: dict[str, Any] | None) -> str:
        """Return a deterministic 24-char hex cache key for the given parameters."""
        raw = f"{query}|{k}|{json.dumps(where, sort_keys=True) if where else ''}"
        return hashlib.sha256(raw.encode()).hexdigest()[:24]

    # ------------------------------------------------------------------
    # Read / write
    # ------------------------------------------------------------------

    def get(self, key: str) -> list[str] | None:
        """Return cached results for *key*, or ``None`` if missing or expired."""
        if self._ttl_s <= 0 or self._max_size <= 0:
            return None

        entry = self._store.get(key)
        if entry is None:
            return None

        ts, docs = entry
        if (time.monotonic() - ts) > self._ttl_s:
            self._store.pop(key, None)
            return None

        # Promote to MRU position
        self._store.move_to_end(key)
        return docs

    def put(self, key: str, docs: list[str]) -> None:
        """Store *docs* under *key*, evicting the LRU entry if at capacity."""
        if self._ttl_s <= 0 or self._max_size <= 0:
            return

        self._store[key] = (time.monotonic(), docs)
        self._store.move_to_end(key)

        while len(self._store) > self._max_size:
            self._store.popitem(last=False)  # evict LRU

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------

    def invalidate(self) -> None:
        """Clear all cached entries (e.g. after documents are added or deleted)."""
        self._store.clear()

    def __len__(self) -> int:
        """Return the number of live entries (may include unexpired stale entries)."""
        return len(self._store)
