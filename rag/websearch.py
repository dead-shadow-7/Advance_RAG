"""Web search as a third retriever.

Tavily returns its own ranked list, which never touches Qdrant. The passages
are not embedded, not indexed, and do not outlive the request — they are turned
straight into Hits, so fusion and reranking handle them exactly like corpus
passages. RRF reads positions and ignores scores, which is what lets Tavily's
relevance number sit next to cosine and BM25 without being made comparable.

Indexing them instead would mean paying embedding latency on every query, and
prune() would delete them at the next /ingest anyway, since their chunk ids are
not produced by anything in data/.
"""

import os
from functools import lru_cache
from urllib.parse import urlparse

import requests
from dotenv import load_dotenv
from tavily import TavilyClient, errors

from rag.chunker import split_text
from rag.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    WEB_CHUNKS_PER_RESULT,
    WEB_RAW_CONTENT,
    WEB_RESULTS,
    WEB_SEARCH_DEPTH,
    WEB_SELECT_CANDIDATES,
    WEB_SELECT_DENSE,
    WEB_TIMEOUT,
)
from rag.embedder import (
    embed_passages,
    embed_passages_sparse,
    embed_query,
    embed_query_sparse,
)
from rag.schemas import Hit

load_dotenv()

# Tavily's exceptions all derive from Exception directly with no shared base,
# so catching "a web search failed" means naming them. requests covers the
# network itself: DNS, connection resets, timeouts.
SearchError = (
    errors.BadRequestError,
    errors.ForbiddenError,
    errors.InvalidAPIKeyError,
    errors.MissingAPIKeyError,
    errors.TimeoutError,
    errors.UsageLimitExceededError,
    requests.exceptions.RequestException,
)


@lru_cache(maxsize=1)
def get_client() -> TavilyClient:
    key = os.environ.get("TAVILY_API_KEY")
    if not key:
        raise RuntimeError("TAVILY_API_KEY is not set — add it to .env")
    return TavilyClient(api_key=key)


def domain(url: str) -> str:
    return urlparse(url).netloc or url


def lexical_scores(query: str, chunks: list[str]) -> list[float]:
    """How strongly each chunk shares vocabulary with the query.

    The BM25 term weights come from fastembed rather than Qdrant, so there are
    no corpus-wide document frequencies behind them — this is term overlap, not
    true BM25. Good enough to sort one page's chunks, and it costs ~1ms each.
    """
    query_vector = embed_query_sparse(query)
    weights = dict(zip(query_vector.indices.tolist(), query_vector.values.tolist()))
    return [
        sum(
            weights.get(index, 0.0) * value
            for index, value in zip(vector.indices.tolist(), vector.values.tolist())
        )
        for vector in embed_passages_sparse(chunks)
    ]


def top_indexes(scores: list[float], keep: int) -> list[int]:
    """Indexes of the `keep` best scores, back in their original order."""
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    return sorted(ranked[:keep])


def best_chunks(
    query: str, chunks: list[str], keep: int, dense: bool = WEB_SELECT_DENSE
) -> list[str]:
    """The `keep` chunks of one page that best answer the query.

    Taking a page's *first* chunks instead would mostly collect nav bars,
    subscribe pitches and cookie notices: a real page opens with chrome and
    reaches its point somewhere in the middle.

    Keyword overlap narrows the page for almost nothing. The embedder then
    chooses between the survivors on meaning — which is what would catch a
    passage that answers the question without reusing its words — but it costs
    seconds per query, so it stays off until measured.
    """
    if len(chunks) <= keep:
        return chunks

    narrowed = top_indexes(
        lexical_scores(query, chunks), WEB_SELECT_CANDIDATES if dense else keep
    )
    chunks = [chunks[index] for index in narrowed]
    if not dense or len(chunks) <= keep:
        return chunks

    query_vector = embed_query(query)
    # BGE returns L2-normalised vectors, so a dot product is already the cosine.
    similarity = [
        sum(q * v for q, v in zip(query_vector, vector))
        for vector in embed_passages(chunks)
    ]
    return [chunks[index] for index in top_indexes(similarity, keep)]


def to_hits(
    query: str, results: list[dict], max_chunks: int = WEB_CHUNKS_PER_RESULT
) -> list[Hit]:
    """Flatten Tavily results into Hits, best result first.

    Results arrive ranked, and the flattening preserves that order, so a hit's
    position in this list is still Tavily's opinion of it — which is all RRF
    reads.
    """
    hits: list[Hit] = []
    seen: set[str] = set()

    for result in results:
        url = result.get("url")
        if not url or url in seen:  # the same page can be returned twice
            continue
        seen.add(url)

        # raw_content is the whole page, and is sometimes null even when asked
        # for. content is not a query-relevant extract despite reading like one
        # — it is the page's opening, truncated — so it is a fallback, not a
        # better-targeted alternative.
        text = (result.get("raw_content") or result.get("content") or "").strip()
        if not text:
            continue

        title = (result.get("title") or "").strip() or domain(url)
        score = float(result.get("score") or 0.0)
        chunks = split_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
        for index, piece in enumerate(best_chunks(query, chunks, max_chunks)):
            hits.append(
                Hit(
                    chunk_id=f"{url}#{index}",
                    text=piece,
                    source=title,
                    score=score,
                    url=url,
                    origin="web",
                )
            )
    return hits


def search(query: str, limit: int = WEB_RESULTS) -> list[Hit]:
    """Search the web and return passages, best first.

    Raises on failure — whether a failed search should sink the request or just
    narrow it is the caller's decision, not this module's.
    """
    response = get_client().search(
        query,
        max_results=limit,
        search_depth=WEB_SEARCH_DEPTH,
        include_raw_content=WEB_RAW_CONTENT,
        timeout=WEB_TIMEOUT,
    )
    return to_hits(query, response.get("results", []))
