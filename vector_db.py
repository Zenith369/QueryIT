import logging

from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
import constants as const

class QdrantStorage:
    """
    Configuration for Qdrant database with update, insert and search functionality.
    """
    def __init__(self, url=const.QDRANT_URL, collection=const.COLLECTION_NAME, dim=const.EMBEDDING_DIM,):

        self.client = QdrantClient(url=url, timeout=30)
        self.collection = collection

        if not self.client.collection_exists(self.collection):
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )

    def upsert(self, ids, vectors, payloads) -> None:
        points = [
            PointStruct(id=ids[i], vector=vectors[i], payload=payloads[i])
            for i in range(len(ids))
        ]
        self.client.upsert(self.collection, points=points)

    def search(self, query_vector, top_k: int = 5):

        response = self.client.query_points(
            collection_name=self.collection,
            query=query_vector,
            limit=top_k,
            with_payload=True,
        )

        results = response.points

        contexts = []
        sources = set()

        for r in results:
            payload = getattr(r, "payload", None) or {}
            text = payload.get("text", "")
            source = payload.get("source", "")
            if text:
                contexts.append(text)
                sources.add(source)

        logging.info(f"📊 CONTEXT TEST {contexts}, {sources}")
        return {"contexts": contexts, "sources": list(sources)}