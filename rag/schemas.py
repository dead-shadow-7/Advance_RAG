from dataclasses import dataclass

from pydantic import BaseModel, Field


@dataclass
class Hit:
    """One retrieved passage, carrying no trace of which store produced it.

    Retrieval, fusion and reranking all work on these, so a passage can travel
    the pipeline without every stage knowing it came out of Qdrant.
    """

    chunk_id: str  # unique per passage, and the key fusion dedups on
    text: str
    source: str  # filename for a corpus passage, page title for a web one
    # From whichever retriever produced this hit. Cosine, BM25, RRF and rerank
    # logits are all on different scales, so it only means anything next to the
    # other scores in the same list.
    score: float
    page: int | None = None
    url: str | None = None  # web passages only
    origin: str = "local"  # "local" | "web"


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)


class Citation(BaseModel):
    source: str  # file the chunk came from
    chunk_id: str
    text: str  # the passage the answer leaned on
    score: float
    page: int | None = None


class AskResponse(BaseModel):
    answer: str
    citations: list[Citation]
