# Running RAG in Production

## Where the latency goes

A single question typically spends a few milliseconds embedding the query, ten
to fifty milliseconds searching the index, and one to several seconds waiting
for the language model. The generation step dominates so completely that
optimising retrieval for speed is almost always the wrong target. Optimise
retrieval for quality and generation for latency.

Adding a reranker inverts part of this. A cross-encoder scores every candidate
against the query jointly, which is far more accurate than comparing two
independent vectors and far slower. Reranking twenty candidates typically adds
fifty to two hundred milliseconds on CPU. That is cheap against a two second
generation, and expensive against nothing at all, so it is worth measuring
rather than assuming.

## Reasoning models

A hybrid reasoning model emits a chain of thought before its answer, and those
thinking tokens are billed as output. For grounded question answering over
retrieved passages, the reasoning rarely changes the answer, because the work of
finding the evidence has already been done by the retriever.

Turning reasoning off for the answering step is usually a large latency win for
no quality loss. Leave it on for planning steps, where the model genuinely has
to decide what to look for.

Be careful combining a token limit with reasoning. If the reasoning trace
exhausts the limit, the response comes back with no content at all, and it is
still billed.

## Caching

Three caches are worth having, in increasing order of effort. Prompt caching at
the provider discounts the repeated instruction block that prefixes every
request, often by five times or more. An embedding cache keyed on the text hash
avoids re-embedding documents that have not changed. A full response cache keyed
on the normalised question is the largest win and the riskiest, since it serves
stale answers after the corpus changes and must be invalidated on reindex.

## Cost

Cost is dominated by output tokens, which are typically three to five times the
price of input tokens. Retrieval itself is nearly free once the index exists;
local embedding costs only CPU time.

The usual surprise is that a verbose system prompt multiplied by every request
outweighs the documents being retrieved. Measure tokens per request before
optimising anything, and track the number alongside latency so a regression in
either is visible.

## Deployment

Embedded vector stores that keep their data in a local folder take an exclusive
lock on that folder, so exactly one process may open them. That is fine for
development and impossible for a deployment with more than one worker. Moving to
a server removes the constraint and usually changes one line of client setup.

Run ingestion as a separate job from serving. Indexing competes for the same CPU
the embedding model needs, and a long ingest will visibly slow every query if it
shares a process with the API.
