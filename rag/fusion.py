"""Reciprocal Rank Fusion.

Dense scores (cosine, ~0-1) and BM25 scores (unbounded, corpus-dependent) are
not comparable, and their ranges shift per query — so averaging them is
meaningless. RRF throws the scores away and keeps only the positions.
"""

from rag.config import RRF_K


def reciprocal_rank_fusion(ranked_lists: list[list], k: int = RRF_K, limit: int = 10):
    """Merge ranked result lists into one.

    Each list contributes 1 / (k + rank) to every point it contains, so a point
    found by both retrievers beats one that a single retriever loved. Returns
    (point, fused_score) pairs, best first.
    """
    scores: dict[str, float] = {}
    points: dict[str, object] = {}

    for ranked in ranked_lists:
        for rank, point in enumerate(ranked, start=1):
            scores[point.id] = scores.get(point.id, 0.0) + 1.0 / (k + rank)
            points[point.id] = point

    ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return [(points[pid], score) for pid, score in ordered[:limit]]
