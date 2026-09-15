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

- [ ] FastAPI app with an `/ask` endpoint.
- [ ] Load PDFs and Markdown, split into chunks.
- [ ] Embed the chunks and store them in Qdrant.
- [ ] Retrieve the top chunks and have the LLM answer from them.
- [ ] Return which chunks the answer came from.

### M2 — Make it find the right things

Retrieval is where RAG quality lives, so this is the milestone worth the most time.

- [ ] Add BM25 keyword search alongside vector search, merged with RRF.
- [ ] Add a reranker on top.
- [ ] Add web search and fold the results into the same pipeline.
- [ ] Write ~20 test questions with known answers, and measure Recall@5 before and after each change above.

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
