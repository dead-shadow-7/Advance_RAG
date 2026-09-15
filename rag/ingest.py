"""Index everything in data/ into Qdrant.

Run with:  python -m rag.ingest
"""

import sys

from qdrant_client import QdrantClient

from rag.chunker import chunk_documents
from rag.config import CHUNK_OVERLAP, CHUNK_SIZE, COLLECTION, DATA_DIR
from rag.embedder import embed_passages, embed_passages_sparse
from rag.loader import load_directory
from rag.store import (
    count,
    create_collection,
    ensure_collection,
    get_client,
    is_hybrid_schema,
    prune,
    upsert_chunks,
)


def ingest(client: QdrantClient | None = None) -> dict:
    """Index data/ into Qdrant.

    Pass a client to reuse an already-open store — that is how the API reindexes
    itself without releasing the folder lock it is holding.
    """
    docs = load_directory(DATA_DIR)
    if not docs:
        print(f"No .pdf, .md or .txt files found in {DATA_DIR}")
        return {"documents": 0, "chunks": 0, "indexed": 0}
    print(f"Loaded {len(docs)} document(s) from {DATA_DIR}")

    chunks = chunk_documents(docs, CHUNK_SIZE, CHUNK_OVERLAP)
    print(f"Split into {len(chunks)} chunks")

    print("Embedding dense + sparse (first run downloads the models)...")
    texts = [chunk.text for chunk in chunks]
    dense = embed_passages(texts)
    sparse = embed_passages_sparse(texts)

    borrowed = client is not None
    client = client or get_client()
    try:
        if client.collection_exists(COLLECTION) and not is_hybrid_schema(client):
            # Built before sparse vectors: the schema cannot be altered in place.
            # Plain ASCII: the Windows console default codepage mangles em-dashes.
            print("Existing collection predates hybrid search - recreating it.")
            client.delete_collection(COLLECTION)
            create_collection(client)
        else:
            ensure_collection(client)

        upsert_chunks(client, chunks, dense, sparse)
        removed = prune(client, [chunk.id for chunk in chunks])
        total = count(client)
    finally:
        if not borrowed:
            client.close()

    if removed:
        print(f"Removed {removed} chunks whose source is gone from {DATA_DIR.name}/")
    print(f"Indexed {len(chunks)} chunks. Collection now holds {total}.")
    return {
        "documents": len(docs),
        "chunks": len(chunks),
        "removed": removed,
        "indexed": total,
    }


def main() -> None:
    try:
        ingest()
    except RuntimeError as exc:
        # Embedded Qdrant allows a single process at a time.
        if "already accessed by another instance" in str(exc):
            sys.exit("Qdrant storage is locked — stop the API server, then re-run.")
        raise


if __name__ == "__main__":
    main()
