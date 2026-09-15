from fastapi import FastAPI

from rag.pipeline import answer_question
from rag.schemas import AskRequest, AskResponse

app = FastAPI(title="ProductionRAG")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    return answer_question(request.question, request.top_k)
