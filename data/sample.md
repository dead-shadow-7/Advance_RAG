# Retrieval-Augmented Generation

Retrieval-Augmented Generation (RAG) combines a retrieval system with a
language model. Instead of relying only on what the model memorised during
training, the system first searches a corpus for relevant passages and then
asks the model to answer using those passages as context.

## Why retrieval helps

Language models hallucinate when asked about facts they never saw, or saw
rarely. Retrieval sidesteps this by putting the relevant text directly in the
prompt. It also makes answers auditable: every claim can point back to the
passage it came from, which is the whole basis for citations.

A second benefit is freshness. Retraining a model to teach it new facts is
expensive. Adding a document to a vector store costs nothing by comparison, and
the change takes effect on the next query.

## Chunking

Documents are split into chunks before embedding, because embedding models have
a bounded input window and because a smaller passage produces a more focused
vector. Chunks that are too large dilute the embedding with unrelated content.
Chunks that are too small lose the context needed to interpret them.

Overlapping chunks mitigate the boundary problem, where a sentence answering the
question is split across two chunks and neither one scores well on its own.

## Retrieval quality

Recall@k measures whether the correct passage appears in the top k results. It
is the ceiling on answer quality: a passage that is never retrieved can never be
cited. Mean Reciprocal Rank additionally rewards putting the correct passage
near the top of the list rather than merely somewhere in it.
