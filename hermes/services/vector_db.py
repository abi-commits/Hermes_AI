"""Vector database service integration."""

import hashlib
from typing import TYPE_CHECKING

import structlog

from config import get_settings
from hermes.core.exceptions import VectorDBError

if TYPE_CHECKING:
    pass

logger = structlog.get_logger(__name__)


class VectorDB:
    """Vector database service using ChromaDB or Pinecone.

    This class provides an abstraction over different vector database
    providers for storing and retrieving document embeddings.
    """

    def __init__(self) -> None:
        """Initialize the vector database service."""
        self.settings = get_settings()
        self._logger = structlog.get_logger(__name__)
        self._client = None
        self._collection = None

    async def connect(self) -> None:
        """Connect to the vector database."""
        if self.settings.vector_db_provider == "chromadb":
            await self._connect_chromadb()
        elif self.settings.vector_db_provider == "pinecone":
            await self._connect_pinecone()
        else:
            raise VectorDBError(f"Unknown vector DB provider: {self.settings.vector_db_provider}")

        self._logger.info("vector_db_connected", provider=self.settings.vector_db_provider)

    async def disconnect(self) -> None:
        """Disconnect from the vector database."""
        if self._client:
            self._client = None
            self._collection = None

    async def _connect_chromadb(self) -> None:
        """Connect to ChromaDB."""
        try:
            import chromadb
            from chromadb.config import Settings as ChromaSettings

            # Connect to ChromaDB
            self._client = chromadb.HttpClient(
                host=self.settings.chromadb_host,
                port=self.settings.chromadb_port,
                settings=ChromaSettings(
                    anonymized_telemetry=False,
                ),
            )

            # Get or create collection
            self._collection = self._client.get_or_create_collection(
                name=self.settings.chromadb_collection,
                metadata={"hnsw:space": "cosine"},
            )

        except ImportError:
            raise VectorDBError("chromadb package not installed")
        except Exception as e:
            raise VectorDBError(f"Failed to connect to ChromaDB: {e}")

    async def _connect_pinecone(self) -> None:
        """Connect to Pinecone."""
        try:
            import pinecone

            if not self.settings.pinecone_api_key:
                raise VectorDBError("Pinecone API key not configured")

            pinecone.init(
                api_key=self.settings.pinecone_api_key,
                environment=self.settings.pinecone_environment or "us-west1-gcp",
            )

            self._client = pinecone

            # Get or create index
            if self.settings.pinecone_index not in pinecone.list_indexes():
                pinecone.create_index(
                    name=self.settings.pinecone_index,
                    dimension=384,  # all-MiniLM-L6-v2 dimension
                    metric="cosine",
                )

            self._collection = pinecone.Index(self.settings.pinecone_index)

        except ImportError:
            raise VectorDBError("pinecone-client package not installed")
        except Exception as e:
            raise VectorDBError(f"Failed to connect to Pinecone: {e}")

    async def add(
        self,
        documents: list[str],
        metadatas: list[dict] | None = None,
        ids: list[str] | None = None,
    ) -> list[str]:
        """Add documents to the vector database.

        Args:
            documents: List of documents to add.
            metadatas: Optional metadata for each document.
            ids: Optional IDs for documents. Generated if not provided.

        Returns:
            List of document IDs.
        """
        if not self._collection:
            raise VectorDBError("Not connected to vector database")

        # Generate IDs if not provided
        if ids is None:
            ids = [
                hashlib.md5(doc.encode()).hexdigest()[:16]
                for doc in documents
            ]

        # Generate embeddings
        embeddings = await self._embed_documents(documents)

        if self.settings.vector_db_provider == "chromadb":
            self._collection.add(
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas or [{}] * len(documents),
                ids=ids,
            )
        elif self.settings.vector_db_provider == "pinecone":
            vectors = [
                (id_, emb, meta)
                for id_, emb, meta in zip(ids, embeddings, metadatas or [{}] * len(documents))
            ]
            self._collection.upsert(vectors=vectors)

        self._logger.debug("documents_added", count=len(documents))
        return ids

    async def search(
        self,
        query: str,
        n_results: int = 5,
        filter_dict: dict | None = None,
    ) -> list[dict]:
        """Search for similar documents.

        Args:
            query: The search query.
            n_results: Number of results to return.
            filter_dict: Optional filter criteria.

        Returns:
            List of matching documents with scores.
        """
        if not self._collection:
            raise VectorDBError("Not connected to vector database")

        # Generate query embedding
        query_embedding = await self._embed_documents([query])

        if self.settings.vector_db_provider == "chromadb":
            results = self._collection.query(
                query_embeddings=query_embedding,
                n_results=n_results,
                where=filter_dict,
                include=["documents", "metadatas", "distances"],
            )

            # Format results
            formatted_results = []
            for i in range(len(results["ids"][0])):
                formatted_results.append({
                    "id": results["ids"][0][i],
                    "content": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "score": 1.0 - results["distances"][0][i],  # Convert distance to similarity
                })

            return formatted_results

        elif self.settings.vector_db_provider == "pinecone":
            results = self._collection.query(
                vector=query_embedding[0],
                top_k=n_results,
                filter=filter_dict,
                include_metadata=True,
            )

            return [
                {
                    "id": match.id,
                    "content": match.metadata.get("text", ""),
                    "metadata": match.metadata,
                    "score": match.score,
                }
                for match in results.matches
            ]

        return []

    async def delete(self, ids: list[str]) -> None:
        """Delete documents by ID.

        Args:
            ids: List of document IDs to delete.
        """
        if not self._collection:
            raise VectorDBError("Not connected to vector database")

        if self.settings.vector_db_provider == "chromadb":
            self._collection.delete(ids=ids)
        elif self.settings.vector_db_provider == "pinecone":
            self._collection.delete(ids=ids)

        self._logger.debug("documents_deleted", count=len(ids))

    async def _embed_documents(self, documents: list[str]) -> list[list[float]]:
        """Generate embeddings for documents.

        This uses sentence-transformers by default.

        Args:
            documents: List of documents to embed.

        Returns:
            List of embedding vectors.
        """
        try:
            from sentence_transformers import SentenceTransformer

            # Use cached model
            if not hasattr(self, "_embedding_model"):
                self._embedding_model = SentenceTransformer(
                    self.settings.embedding_model
                )

            embeddings = self._embedding_model.encode(documents)
            return embeddings.tolist()

        except ImportError:
            raise VectorDBError("sentence-transformers not installed")
        except Exception as e:
            raise VectorDBError(f"Embedding generation failed: {e}")

    async def get_collection_stats(self) -> dict:
        """Get statistics about the collection.

        Returns:
            Dictionary with collection statistics.
        """
        if not self._collection:
            return {"status": "disconnected"}

        if self.settings.vector_db_provider == "chromadb":
            return {
                "name": self._collection.name,
                "count": self._collection.count(),
            }
        elif self.settings.vector_db_provider == "pinecone":
            stats = self._collection.describe_index_stats()
            return {
                "name": self.settings.pinecone_index,
                "total_vectors": stats.total_vector_count,
            }

        return {}
