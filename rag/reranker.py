"""Cross-encoder reranking.

Retrieval compares a query vector against passage vectors that were embedded
without ever seeing the query. A cross-encoder reads both together in one pass,
so it can judge whether a passage actually answers the question rather than
merely sitting near it in vector space. That costs a model call per candidate,
which is why it runs on a shortlist instead of the whole index.
"""

import tiktoken
from fastembed.rerank.cross_encoder import TextCrossEncoder

from rag.config import MODEL_CACHE, RERANK_MAX_TOKENS, RERANK_MODEL

_reranker: TextCrossEncoder | None = None
_enc = tiktoken.get_encoding("cl100k_base")


def get_reranker() -> TextCrossEncoder:
    global _reranker
    if _reranker is None:
        _reranker = TextCrossEncoder(RERANK_MODEL, cache_dir=str(MODEL_CACHE))
    return _reranker


def truncate(text: str, max_tokens: int = RERANK_MAX_TOKENS) -> str:
    tokens = _enc.encode(text)
    return text if len(tokens) <= max_tokens else _enc.decode(tokens[:max_tokens])


def rerank_passages(query: str, passages: list[str]) -> list[float]:
    """Relevance scores, one per passage, in the order given.

    Scores are raw logits: unbounded, and comparable only within one query.
    Passages are truncated first — scoring cost is linear in their length.
    """
    return list(get_reranker().rerank(query, [truncate(p) for p in passages]))
