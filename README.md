# 🔎 Research Assistant RAG

An AI research assistant that answers questions from your documents and the web, and writes back a short report with citations.

## 🎯 What it should do

1. Answer a question using uploaded documents.
2. Pull in the web when the documents aren't enough.
3. Cite every claim it makes.
4. Tell you when it isn't sure.

## 🛠️ Tech Stack

Start with these:

- **Backend:** Python + FastAPI
- **Embeddings:** Sentence Transformers (BGE)
- **Vector DB:** Qdrant
- **LLM:** OpenRouter

Add later, only when a milestone needs it: LangGraph (M3), Tavily web search (M2), PostgreSQL, React frontend, Ragas, Docker.

## 🏗️ How it works

```text
Question
   │
   ▼
Retrieve  ──► documents (+ web, from M2)
   │
   ▼
Rank the best chunks
   │
   ▼
LLM writes the answer
   │
   ▼
Answer + citations
```

Everything in later milestones is a better version of one of those four boxes.

## 📚 Roadmap

### M1 — Make it work

A question goes in, a cited answer comes out.

- [x] FastAPI app with an `/ask` endpoint.
- [x] Load PDFs and Markdown, split into chunks.
- [x] Embed the chunks and store them in Qdrant.
- [x] Retrieve the top chunks and have the LLM answer from them.
- [x] Return which chunks the answer came from.

Run it: `python -m rag.ingest` to index `data/`, then `uvicorn main:app --reload`.

Debug UI: `streamlit run ui/app.py` — shows the ranked chunks and scores behind
each answer, and manages the corpus: upload documents, delete them, re-index.
It calls the API over HTTP, so the API must be running too.

| endpoint | does |
|---|---|
| `POST /ask` | question in, cited answer out |
| `GET /documents` | list the corpus |
| `POST /documents` | upload a .pdf/.md/.txt into `data/` |
| `DELETE /documents/{name}` | remove one |
| `POST /ingest` | re-index `data/` |

Embedded Qdrant locks its folder to one process, which is why indexing goes
through the API's own `POST /ingest` rather than a second process. Chunks
outlive a deleted file until the next index, where `prune()` removes them.

### M2 — Make it find the right things

Retrieval is where RAG quality lives, so this is the milestone worth the most time.

- [x] Add BM25 keyword search alongside vector search, merged with RRF.
- [x] Add a reranker on top (built and measurable; `RERANK = False` until a real
      corpus justifies its cost).
- [ ] Add web search and fold the results into the same pipeline.
- [x] Write ~20 test questions with known answers, and measure Recall@5 before and after each change above.

Measure with `python -m eval.evaluate` (builds its own in-memory index, so it
runs alongside the API). Current corpus is 9 chunks, 20 questions:

| run | recall@5 | MRR | MRR exact | MRR paraphrase | ms/query |
|---|---|---|---|---|---|
| dense | 0.90 | 0.695 | 0.674 | 0.708 | 6 |
| sparse (BM25) | 1.00 | **0.875** | 0.875 | **0.875** | 1 |
| hybrid (RRF) | 1.00 | 0.838 | 0.875 | 0.812 | 7 |
| dense + rerank | 1.00 | 0.823 | 0.917 | 0.760 | 703 |
| hybrid + rerank | 1.00 | 0.823 | **0.917** | 0.760 | 694 |

Reranking rescues a weak ranking (dense 0.695 → 0.823) but slightly *hurts* an
already-good one, at ~100x the latency. At 9 chunks the shortlist is the entire
corpus, so there is nothing for it to rescue. That is a fact about this corpus,
not about rerankers.

Earlier, with a 126-page PDF also indexed (161 chunks), the same questions gave
dense 0.675 / sparse 0.733 / **hybrid 0.833** — each retriever winning its own
half and hybrid winning overall. On 9 chunks hybrid looks worse than BM25 alone.
Small evaluation sets do not return "no signal"; they return a confident wrong
answer.

### M3 — Make it handle hard questions

- [ ] Split a complex question into subquestions, research each, combine the results.
- [ ] Introduce LangGraph once the control flow stops fitting in plain functions.
- [ ] Check each claim in the answer against the retrieved evidence; flag unsupported ones.

### M4 — Make it shippable

- [ ] Stream responses.
- [ ] Add caching, logging, and a simple frontend.
- [ ] Dockerize and deploy.

Add auth, rate limiting, and background jobs when something actually needs them.

## 📁 Project Structure

Start flat — one file until it hurts:

```text
ProductionRAG/
├── main.py            # FastAPI app
├── rag/               # chunking, embedding, retrieval, generation
├── data/              # source documents
├── eval/              # test questions + scoring script
└── requirements.txt
```

Split into `api/`, `services/`, `core/` when a file gets hard to navigate, not before.

## 📌 Approach

Build one milestone end to end before starting the next. Measure retrieval before tuning it — otherwise "improvements" are guesses.

## 📖 Resources

- [FastAPI](https://fastapi.tiangolo.com/)
- [Qdrant](https://qdrant.tech/documentation/)
- [LangGraph](https://langchain-ai.github.io/langgraph/)
- [Ragas](https://docs.ragas.io/)

## 📄 License

To be added.
