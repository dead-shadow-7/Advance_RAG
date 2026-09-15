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
each answer, and has a "Re-index data/" button. It calls the API over HTTP, so
the API must be running too. Embedded Qdrant locks its folder to one process,
which is why reindexing goes through the API's own `POST /ingest` rather than a
second process.

### M2 — Make it find the right things

Retrieval is where RAG quality lives, so this is the milestone worth the most time.

- [x] Add BM25 keyword search alongside vector search, merged with RRF.
- [ ] Add a reranker on top.
- [ ] Add web search and fold the results into the same pipeline.
- [x] Write ~20 test questions with known answers, and measure Recall@5 before and after each change above.

Measure with `python -m eval.evaluate` (builds its own in-memory index, so it
runs alongside the API). On 161 chunks, 20 questions:

| mode | recall@5 | MRR | MRR exact | MRR paraphrase |
|---|---|---|---|---|
| dense | 0.85 | 0.675 | 0.625 | 0.708 |
| sparse (BM25) | 0.95 | 0.733 | **0.875** | 0.639 |
| hybrid (RRF) | **1.00** | **0.833** | 0.812 | **0.847** |

Each retriever wins its own half — BM25 on exact identifiers, dense on
paraphrase — and hybrid is the only one strong at both. Dense misses questions
like "what does IDF stand for" entirely; BM25 ranks them first.

On the earlier 9-chunk corpus hybrid looked *worse* than BM25 alone. That
corpus was too small to measure anything: 10 results requested from 9 chunks
makes recall@10 meaningless. Small evaluation sets mislead confidently.

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
