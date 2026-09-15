from rag.schemas import AskResponse, Citation


def retrieve(question: str, top_k: int) -> list[Citation]:
    """Find the chunks most likely to answer the question.

    TODO: embed the question and search Qdrant. Returns nothing until
    documents are indexed.
    """
    return []


def generate(question: str, chunks: list[Citation]) -> str:
    """Write an answer grounded in the retrieved chunks.

    TODO: send question + chunks to the LLM.
    """
    if not chunks:
        return "No documents indexed yet, so I have nothing to answer from."
    return "Retrieval is wired up, but the LLM call is not."


def answer_question(question: str, top_k: int = 5) -> AskResponse:
    chunks = retrieve(question, top_k)
    return AskResponse(answer=generate(question, chunks), citations=chunks)
