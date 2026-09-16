"""Reciprocal Rank Fusion.

Dense scores (cosine, ~0-1) and BM25 scores (unbounded, corpus-dependent) are
not comparable, and their ranges shift per query — so averaging them is
meaningless. RRF throws the scores away and keeps only the positions.
"""

from rag.config import RRF_K
from rag.schemas import Hit


def reciprocal_rank_fusion(
    ranked_lists: list[list[Hit]], k: int = RRF_K, limit: int = 10
) -> list[tuple[Hit, float]]:
    """Merge ranked result lists into one.

    Each list contributes 1 / (k + rank) to every hit it contains, so a hit
    found by both retrievers beats one that a single retriever loved. Returns
    (hit, fused_score) pairs, best first.

    Only positions are read, never the incoming scores, so the lists need not
    come from comparable retrievers — or even from the same kind of store.
    """
    scores: dict[str, float] = {}
    hits: dict[str, Hit] = {}

    for ranked in ranked_lists:
        for rank, hit in enumerate(ranked, start=1):
            scores[hit.chunk_id] = scores.get(hit.chunk_id, 0.0) + 1.0 / (k + rank)
            hits[hit.chunk_id] = hit

    ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return [(hits[chunk_id], score) for chunk_id, score in ordered[:limit]]
