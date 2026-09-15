from qdrant_client import QdrantClient

from rag.embedder import embed_query
from rag.llm import complete
from rag.schemas import AskResponse, Citation
from rag.store import ensure_collection, get_client, search

_client: QdrantClient | None = None


def _store() -> QdrantClient:
    """Opened once and held — embedded Qdrant locks its folder per process."""
    global _client
    if _client is None:
        _client = get_client()
        ensure_collection(_client)
    return _client


def retrieve(question: str, top_k: int) -> list[Citation]:
    hits = search(_store(), embed_query(question), top_k)
    return [
        Citation(
            source=hit.payload["source"],
            chunk_id=hit.payload["chunk_id"],
            text=hit.payload["text"],
            score=hit.score,
            page=hit.payload.get("page"),
        )
        for hit in hits
    ]


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
