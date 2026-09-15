"""Debug UI for the RAG pipeline.

Run with:  streamlit run ui/app.py

Talks to the API over HTTP rather than importing rag.pipeline, because embedded
Qdrant locks its folder to one process — importing it here would fight uvicorn
for that lock.
"""

import time

import requests
import streamlit as st

DEFAULT_API = "http://127.0.0.1:8000"

st.set_page_config(page_title="RAG debug", layout="wide")

# --- sidebar -----------------------------------------------------------------
api_url = st.sidebar.text_input("API URL", DEFAULT_API).rstrip("/")
top_k = st.sidebar.slider("top_k", min_value=1, max_value=20, value=5)

try:
    healthy = requests.get(f"{api_url}/health", timeout=2).status_code == 200
except requests.RequestException:
    healthy = False
st.sidebar.write("API: " + ("🟢 up" if healthy else "🔴 unreachable"))
if not healthy:
    st.sidebar.code("uvicorn main:app --reload")

def reindex(label: str = "Loading, chunking and embedding data/ ...") -> None:
    """The API owns the Qdrant lock, so it reindexes on our behalf."""
    with st.spinner(label):
        try:
            done = requests.post(f"{api_url}/ingest", timeout=1800)
            done.raise_for_status()
        except requests.RequestException as exc:
            st.sidebar.error(f"Indexing failed: {exc}")
            return
    counts = done.json()
    message = f"{counts['chunks']} chunks from {counts['documents']} pages/files"
    if counts.get("removed"):
        message += f", {counts['removed']} stale removed"
    st.sidebar.success(message)


with st.sidebar.expander("Documents", expanded=not healthy):
    uploads = st.file_uploader(
        "Add to the corpus",
        type=["pdf", "md", "txt"],
        accept_multiple_files=True,
        disabled=not healthy,
    )
    if uploads and st.button("Upload and index", type="primary", width="stretch"):
        failed = []
        for upload in uploads:
            try:
                sent = requests.post(
                    f"{api_url}/documents",
                    files={"file": (upload.name, upload.getvalue())},
                    timeout=600,
                )
                sent.raise_for_status()
            except requests.RequestException as exc:
                failed.append(f"{upload.name}: {exc}")
        if failed:
            st.error("\n".join(failed))
        # One reindex for the whole batch rather than one per file.
        reindex(f"Indexing {len(uploads)} new file(s) ...")

    try:
        documents = requests.get(f"{api_url}/documents", timeout=5).json()
    except requests.RequestException:
        documents = []
    for document in documents:
        name_col, delete_col = st.columns([4, 1])
        name_col.caption(f"{document['name']}  ·  {document['bytes'] // 1024}KB")
        if delete_col.button("✕", key=f"del{document['name']}", help="Delete"):
            requests.delete(f"{api_url}/documents/{document['name']}", timeout=30)
            # Chunks outlive the file until the next reindex, so do it now.
            reindex(f"Removing {document['name']} from the index ...")
            st.rerun()

if st.sidebar.button("Re-index data/", disabled=not healthy, width="stretch"):
    reindex()

# Re-asking the same question after each retrieval change is the M2 loop.
history = st.session_state.setdefault("history", [])
if history:
    st.sidebar.subheader("Recent")
    for past in reversed(history[-8:]):
        if st.sidebar.button(past, key=f"h{past}", width="stretch"):
            st.session_state["question"] = past

# --- ask ---------------------------------------------------------------------
st.title("🔎 RAG debug")

question = st.text_area("Question", key="question", height=80)
if st.button("Ask", type="primary", disabled=not question.strip()):
    started = time.time()
    try:
        response = requests.post(
            f"{api_url}/ask",
            json={"question": question, "top_k": top_k},
            timeout=120,
        )
    except requests.RequestException as exc:
        st.session_state["result"] = {"error": f"Could not reach {api_url} — {exc}"}
    else:
        if response.status_code == 200:
            st.session_state["result"] = {
                "data": response.json(),
                "elapsed": time.time() - started,
            }
            if question not in history:
                history.append(question)
        else:
            detail = response.json().get("detail", response.text)
            st.session_state["result"] = {
                "error": f"HTTP {response.status_code} — {detail}"
            }

# --- results -----------------------------------------------------------------
def render(data: dict, elapsed: float, requested_k: int) -> None:
    citations = data["citations"]
    answer_col, chunks_col = st.columns([2, 3], gap="large")

    with answer_col:
        st.subheader("Answer")
        st.markdown(data["answer"])
        left, right = st.columns(2)
        left.metric("Latency", f"{elapsed:.1f}s")
        right.metric("Chunks retrieved", f"{len(citations)}/{requested_k}")
        if len(citations) < requested_k:
            st.caption("Fewer chunks than requested — the index is small.")

    with chunks_col:
        st.subheader("Retrieved chunks")
        if not citations:
            st.warning("Nothing retrieved. Ingested yet? `python -m rag.ingest`")
        # Cosine sits in 0-1 but a fused RRF score is ~0.03, so bars are drawn
        # relative to the top hit and mean "how close to the best result".
        top_score = max((c["score"] for c in citations), default=1.0) or 1.0
        for rank, citation in enumerate(citations, start=1):
            where = citation["source"] + (
                f" · page {citation['page']}" if citation["page"] else ""
            )
            header = f"**[{rank}]** {where} — `{citation['score']:.4f}`"
            with st.expander(header, expanded=rank == 1):
                st.progress(max(0.0, min(1.0, citation["score"] / top_score)))
                st.text(citation["text"])
                st.caption(f"chunk_id: {citation['chunk_id']}")


result = st.session_state.get("result")

if not result:
    st.info("Ask something. Retrieved chunks and their scores show up on the right.")
elif "error" in result:
    st.error(result["error"])
else:
    render(result["data"], result["elapsed"], top_k)
