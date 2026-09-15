"""Index everything in data/ into Qdrant.

Run with:  python -m rag.ingest
"""

import sys

from rag.chunker import chunk_documents
from rag.config import CHUNK_OVERLAP, CHUNK_SIZE, DATA_DIR
from rag.embedder import embed_passages
from rag.loader import load_directory
from rag.store import count, ensure_collection, get_client, upsert_chunks


def ingest() -> int:
    docs = load_directory(DATA_DIR)
    if not docs:
        print(f"No .pdf, .md or .txt files found in {DATA_DIR}")
        return 0
    print(f"Loaded {len(docs)} document(s) from {DATA_DIR}")

    chunks = chunk_documents(docs, CHUNK_SIZE, CHUNK_OVERLAP)
    print(f"Split into {len(chunks)} chunks")

    print("Embedding (first run downloads the model)...")
    vectors = embed_passages([chunk.text for chunk in chunks])

    client = get_client()
    ensure_collection(client)
    upsert_chunks(client, chunks, vectors)
    total = count(client)
    client.close()

    print(f"Indexed {len(chunks)} chunks. Collection now holds {total}.")
    return len(chunks)


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
