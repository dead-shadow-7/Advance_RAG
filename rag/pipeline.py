from qdrant_client import QdrantClient

from rag.config import CANDIDATES, RERANK, RERANK_CANDIDATES, RETRIEVAL_MODE
from rag.embedder import embed_query, embed_query_sparse
from rag.fusion import reciprocal_rank_fusion
from rag.llm import complete
from rag.reranker import rerank_passages
from rag.schemas import AskResponse, Citation
from rag.store import dense_search, ensure_collection, get_client, sparse_search

_client: QdrantClient | None = None


def get_store() -> QdrantClient:
    """Opened once and held — embedded Qdrant locks its folder per process."""
    global _client
    if _client is None:
        _client = get_client()
        ensure_collection(_client)
    return _client


def search_points(
    client: QdrantClient,
    question: str,
    top_k: int,
    mode: str = RETRIEVAL_MODE,
    candidates: int = CANDIDATES,
    rerank: bool = RERANK,
) -> list[tuple]:
    """Return (point, score) pairs, best first.

    Takes an explicit client so the evaluation harness can run against its own
    index without fighting the API for the storage lock.
    """
    # Reranking can only reorder what retrieval handed it, so fetch a shortlist
    # rather than just the top_k that will ultimately be returned.
    depth = max(top_k, RERANK_CANDIDATES) if rerank else top_k

    if mode == "dense":
        hits = dense_search(client, embed_query(question), depth)
        scored = [(hit, hit.score) for hit in hits]
    elif mode == "sparse":
        hits = sparse_search(client, embed_query_sparse(question), depth)
        scored = [(hit, hit.score) for hit in hits]
    elif mode == "hybrid":
        # Pull deep from each retriever so fusion has room to promote a chunk
        # that only one of them found.
        scored = reciprocal_rank_fusion(
            [
                dense_search(client, embed_query(question), candidates),
                sparse_search(client, embed_query_sparse(question), candidates),
            ],
            limit=depth,
        )
    else:
        raise ValueError(f"unknown retrieval mode: {mode}")

    if not rerank or not scored:
        return scored[:top_k]

    shortlist = scored[:RERANK_CANDIDATES]
    scores = rerank_passages(question, [point.payload["text"] for point, _ in shortlist])
    reordered = sorted(
        ((point, score) for (point, _), score in zip(shortlist, scores)),
        key=lambda pair: pair[1],
        reverse=True,
    )
    return reordered[:top_k]


def to_citations(scored: list[tuple]) -> list[Citation]:
    return [
        Citation(
            source=point.payload["source"],
            chunk_id=point.payload["chunk_id"],
            text=point.payload["text"],
            score=score,
            page=point.payload.get("page"),
        )
        for point, score in scored
    ]


def retrieve(question: str, top_k: int) -> list[Citation]:
    return to_citations(search_points(get_store(), question, top_k))


SYSTEM_PROMPT = """You answer questions using only the numbered context passages you are given.

Rules:
- Use only those passages. Never add outside knowledge.
- End every claim with the number of the passage it came from, like [2].
- If the passages do not answer the question, say so plainly instead of guessing.
- Be concise."""


def format_context(chunks: list[Citation]) -> str:
    """Number the passages so the model can cite them positionally."""
    blocks = []
    for number, chunk in enumerate(chunks, start=1):
        where = chunk.source + (f", page {chunk.page}" if chunk.page else "")
        blocks.append(f"[{number}] ({where})\n{chunk.text}")
    return "\n\n".join(blocks)


def generate(question: str, chunks: list[Citation]) -> str:
    if not chunks:
        return "Nothing indexed matches that question."
    return complete(
        SYSTEM_PROMPT,
        f"Context:\n{format_context(chunks)}\n\nQuestion: {question}",
    )


def answer_question(question: str, top_k: int = 5) -> AskResponse:
    chunks = retrieve(question, top_k)
    return AskResponse(answer=generate(question, chunks), citations=chunks)
