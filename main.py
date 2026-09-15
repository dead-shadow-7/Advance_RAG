from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from openai import OpenAIError

from rag.config import DATA_DIR
from rag.ingest import ingest
from rag.loader import SUPPORTED
from rag.pipeline import answer_question, get_store
from rag.schemas import AskRequest, AskResponse

app = FastAPI(title="ProductionRAG")

MAX_UPLOAD_BYTES = 50 * 1024 * 1024


def resolve_in_data(filename: str) -> Path:
    """Map a client-supplied filename to a path inside data/, or refuse.

    Filenames arrive from the network, so anything resembling a path is
    stripped and the result is checked to still sit inside data/.
    """
    name = Path(filename or "").name
    if Path(name).suffix.lower() not in SUPPORTED:
        raise HTTPException(400, f"Only {sorted(SUPPORTED)} files are supported")
    path = (DATA_DIR / name).resolve()
    if not path.is_relative_to(DATA_DIR.resolve()):
        raise HTTPException(400, "Invalid filename")
    return path


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


@app.get("/documents")
def list_documents():
    if not DATA_DIR.is_dir():
        return []
    return [
        {"name": path.name, "bytes": path.stat().st_size}
        for path in sorted(DATA_DIR.iterdir())
        if path.is_file() and path.suffix.lower() in SUPPORTED
    ]


@app.post("/documents")
async def upload_document(file: UploadFile = File(...)):
    """Save an uploaded file into data/. Indexing is a separate call, so a
    batch of uploads costs one reindex rather than one each."""
    path = resolve_in_data(file.filename)
    contents = await file.read()
    if not contents:
        raise HTTPException(400, "File is empty")
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"File exceeds {MAX_UPLOAD_BYTES // 1024 // 1024}MB")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path.write_bytes(contents)
    return {"name": path.name, "bytes": len(contents)}


@app.delete("/documents/{filename}")
def delete_document(filename: str):
    path = resolve_in_data(filename)
    if not path.is_file():
        raise HTTPException(404, f"No such document: {path.name}")
    path.unlink()
    # The chunks survive until the next reindex, where prune() removes them.
    return {"deleted": path.name}


@app.post("/ingest")
def reindex():
    """Re-index data/ using the store this process already has open.

    Embedded Qdrant locks its folder to one process, so indexing from here is
    what lets you add a document without stopping the server.
    """
    return ingest(get_store())
