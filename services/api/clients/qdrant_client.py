"""Qdrant vector database client with local sentence-transformer embeddings."""

from __future__ import annotations

import logging
from typing import Any

from qdrant_client import AsyncQdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384


class QdrantVectorClient:
    """Async Qdrant client with built-in text embedding.

    The sentence-transformer model is loaded once at construction time and
    reused for every embed call.
    """

    def __init__(self, url: str) -> None:
        self._url = url
        self._client: AsyncQdrantClient | None = None
        logger.info("Loading embedding model %s …", EMBEDDING_MODEL)
        self._model = SentenceTransformer(EMBEDDING_MODEL)
        logger.info("QdrantVectorClient initialised (url=%s)", url)

    @property
    def _qdrant(self) -> AsyncQdrantClient:
        if self._client is None:
            raise RuntimeError("QdrantVectorClient is not connected — call connect() first")
        return self._client

    async def connect(self) -> None:
        """Create the async Qdrant client."""
        self._client = AsyncQdrantClient(url=self._url)
        logger.info("Qdrant client connected at %s", self._url)

    async def close(self) -> None:
        """Close the Qdrant client."""
        if self._client is not None:
            await self._client.close()
            self._client = None
            logger.info("Qdrant client closed")

    async def ensure_collections(self) -> None:
        """Create the ``transactions`` and ``merchants`` collections if they do not exist."""
        for name in ("transactions", "merchants"):
            try:
                await self._qdrant.get_collection(name)
                logger.info("Collection '%s' already exists", name)
            except (UnexpectedResponse, Exception):
                await self._qdrant.create_collection(
                    collection_name=name,
                    vectors_config=VectorParams(
                        size=EMBEDDING_DIM,
                        distance=Distance.COSINE,
                    ),
                )
                logger.info("Created collection '%s'", name)

    async def embed_text(self, text: str) -> list[float]:
        """Return the embedding vector for *text*."""
        return self._model.encode(text).tolist()

    async def upsert(
        self,
        collection: str,
        id: str,
        text: str,
        payload: dict[str, Any],
    ) -> None:
        """Embed *text* and upsert the resulting point into *collection*."""
        vector = await self.embed_text(text)
        await self._qdrant.upsert(
            collection_name=collection,
            points=[
                PointStruct(
                    id=id,
                    vector=vector,
                    payload=payload,
                )
            ],
        )
        logger.debug("Upserted point %s into '%s'", id, collection)

    async def search(
        self,
        collection: str,
        query: str,
        limit: int = 10,
        threshold: float = 0.7,
    ) -> list[dict[str, Any]]:
        """Embed *query* and return the closest points above *threshold*.

        Returns a list of ``{"id": ..., "score": ..., "payload": ...}`` dicts.
        """
        vector = await self.embed_text(query)
        results = await self._qdrant.search(
            collection_name=collection,
            query_vector=vector,
            limit=limit,
            score_threshold=threshold,
        )
        return [
            {"id": hit.id, "score": hit.score, "payload": hit.payload}
            for hit in results
        ]

    async def delete(self, collection: str, id: str) -> None:
        """Delete a single point by id from *collection*."""
        await self._qdrant.delete(
            collection_name=collection,
            points_selector=[id],
        )
        logger.debug("Deleted point %s from '%s'", id, collection)
