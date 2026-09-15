import uuid

from qdrant_client import QdrantClient, models

from rag.chunker import Chunk
from rag.config import COLLECTION, EMBEDDING_DIM, QDRANT_PATH

# Qdrant point ids must be ints or UUIDs, so chunk ids are hashed into one.
# Deterministically: re-ingesting a file overwrites its points instead of
# piling up duplicates.
_NAMESPACE = uuid.UUID("6f0ad1bc-9d6f-4f8e-8b4a-1d7f3c2e5a90")

# Both vectors live on the same point, so a chunk is one record either way.
DENSE = "dense"
SPARSE = "sparse"


def point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(_NAMESPACE, chunk_id))


def get_client() -> QdrantClient:
    """Embedded Qdrant — a local folder, no server.

    It takes an exclusive lock on that folder, so the API and the ingest script
    cannot both hold it. Swap in a server URL here when you dockerise in M4.
    """
    return QdrantClient(path=str(QDRANT_PATH))


def create_collection(client: QdrantClient) -> None:
    client.create_collection(
        COLLECTION,
        vectors_config={
            DENSE: models.VectorParams(
                size=EMBEDDING_DIM, distance=models.Distance.COSINE
            )
        },
        # IDF is a corpus-wide statistic, so Qdrant computes it over the indexed
        # documents rather than fastembed guessing it per batch.
        sparse_vectors_config={
            SPARSE: models.SparseVectorParams(modifier=models.Modifier.IDF)
        },
    )


def ensure_collection(client: QdrantClient) -> None:
    if not client.collection_exists(COLLECTION):
        create_collection(client)


def is_hybrid_schema(client: QdrantClient) -> bool:
    """False for a collection built before sparse vectors existed."""
    if not client.collection_exists(COLLECTION):
        return False
    params = client.get_collection(COLLECTION).config.params
    dense_ok = isinstance(params.vectors, dict) and DENSE in params.vectors
    sparse_ok = bool(params.sparse_vectors) and SPARSE in params.sparse_vectors
    return dense_ok and sparse_ok


def to_sparse_vector(sparse) -> models.SparseVector:
    return models.SparseVector(
        indices=sparse.indices.tolist(), values=sparse.values.tolist()
    )


def upsert_chunks(
    client: QdrantClient,
    chunks: list[Chunk],
    dense_vectors: list[list[float]],
    sparse_vectors: list,
) -> None:
    client.upsert(
        COLLECTION,
        points=[
            models.PointStruct(
                id=point_id(chunk.id),
                vector={DENSE: dense, SPARSE: to_sparse_vector(sparse)},
                payload={
                    "chunk_id": chunk.id,
                    "text": chunk.text,
                    "source": chunk.source,
                    "page": chunk.page,
                },
            )
            for chunk, dense, sparse in zip(chunks, dense_vectors, sparse_vectors)
        ],
    )


def dense_search(client: QdrantClient, vector: list[float], limit: int):
    return client.query_points(
        COLLECTION, query=vector, using=DENSE, limit=limit, with_payload=True
    ).points


def sparse_search(client: QdrantClient, sparse, limit: int):
    return client.query_points(
        COLLECTION,
        query=to_sparse_vector(sparse),
        using=SPARSE,
        limit=limit,
        with_payload=True,
    ).points


def count(client: QdrantClient) -> int:
    return client.count(COLLECTION).count
