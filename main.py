from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from openai import OpenAIError

from rag.ingest import ingest
from rag.pipeline import answer_question, get_store
from rag.schemas import AskRequest, AskResponse

app = FastAPI(title="ProductionRAG")


@app.exception_handler(OpenAIError)
def llm_unavailable(request: Request, exc: OpenAIError) -> JSONResponse:
    """The upstream LLM failing is not our bug — say so with a 502."""
    return JSONResponse(
        status_code=502, content={"detail": f"LLM request failed: {type(exc).__name__}"}
    )


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    return answer_question(request.question, request.top_k)


@app.post("/ingest")
def reindex():
    """Re-index data/ using the store this process already has open.

    Embedded Qdrant locks its folder to one process, so indexing from here is
    what lets you add a document without stopping the server.
    """
    return ingest(get_store())
