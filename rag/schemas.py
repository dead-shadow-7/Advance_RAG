from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)


class Citation(BaseModel):
    source: str  # file the chunk came from
    chunk_id: str
    text: str  # the passage the answer leaned on
    score: float


class AskResponse(BaseModel):
    answer: str
    citations: list[Citation]
