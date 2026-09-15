import uuid

from qdrant_client import QdrantClient, models

from rag.chunker import Chunk
from rag.config import COLLECTION, EMBEDDING_DIM, QDRANT_PATH

# Qdrant point ids must be ints or UUIDs, so chunk ids are hashed into one.
# Deterministically: re-ingesting a file overwrites its points instead of
# piling up duplicates.
_NAMESPACE = uuid.UUID("6f0ad1bc-9d6f-4f8e-8b4a-1d7f3c2e5a90")


def point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(_NAMESPACE, chunk_id))


def get_client() -> QdrantClient:
    """Embedded Qdrant — a local folder, no server.

    It takes an exclusive lock on that folder, so the API and the ingest script
    cannot both hold it. Swap in a server URL here when you dockerise in M4.
    """
    return QdrantClient(path=str(QDRANT_PATH))


def ensure_collection(client: QdrantClient) -> None:
    if not client.collection_exists(COLLECTION):
        client.create_collection(
            COLLECTION,
            vectors_config=models.VectorParams(
                size=EMBEDDING_DIM, distance=models.Distance.COSINE
            ),
        )


def upsert_chunks(
    client: QdrantClient, chunks: list[Chunk], vectors: list[list[float]]
) -> None:
    client.upsert(
        COLLECTION,
        points=[
            models.PointStruct(
                id=point_id(chunk.id),
                vector=vector,
                payload={
                    "chunk_id": chunk.id,
                    "text": chunk.text,
                    "source": chunk.source,
                    "page": chunk.page,
                },
            )
            for chunk, vector in zip(chunks, vectors)
        ],
    )


def search(client: QdrantClient, vector: list[float], top_k: int):
    return client.query_points(
        COLLECTION, query=vector, limit=top_k, with_payload=True
    ).points


def count(client: QdrantClient) -> int:
    return client.count(COLLECTION).count
