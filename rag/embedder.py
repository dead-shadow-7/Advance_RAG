from fastembed import TextEmbedding

from rag.config import EMBEDDING_MODEL, MODEL_CACHE

# BGE was trained with retrieval queries carrying this prefix while passages are
# embedded bare. fastembed's query_embed() does not add it for this model, so we
# do it here. Skipping it costs real recall.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

_model: TextEmbedding | None = None


def get_model() -> TextEmbedding:
    """Loaded once, lazily — the first call downloads ~130MB."""
    global _model
    if _model is None:
        _model = TextEmbedding(EMBEDDING_MODEL, cache_dir=str(MODEL_CACHE))
    return _model


def embed_passages(texts: list[str]) -> list[list[float]]:
    return [vector.tolist() for vector in get_model().embed(texts)]


def embed_query(text: str) -> list[float]:
    return list(get_model().embed([QUERY_PREFIX + text]))[0].tolist()
