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
        for rank, citation in enumerate(citations, start=1):
            where = citation["source"] + (
                f" · page {citation['page']}" if citation["page"] else ""
            )
            header = f"**[{rank}]** {where} — `{citation['score']:.4f}`"
            with st.expander(header, expanded=rank == 1):
                # Cosine similarity is 0..1 here; the bar makes rank gaps visible.
                st.progress(max(0.0, min(1.0, citation["score"])))
                st.text(citation["text"])
                st.caption(f"chunk_id: {citation['chunk_id']}")


result = st.session_state.get("result")

if not result:
    st.info("Ask something. Retrieved chunks and their scores show up on the right.")
elif "error" in result:
    st.error(result["error"])
else:
    render(result["data"], result["elapsed"], top_k)
