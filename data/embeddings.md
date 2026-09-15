# Embedding Models

An embedding model maps text to a fixed-length vector so that semantic
similarity becomes geometric proximity. Everything downstream depends on that
mapping being good for your domain.

## Dimensions and size

BAAI/bge-small-en-v1.5 produces 384 dimensions and is roughly 130 megabytes on
disk. The base variant produces 768 dimensions at about 440 megabytes, and the
large variant 1024 dimensions at around 1.3 gigabytes.

Larger is not automatically better. Going from small to base typically buys a
few points of recall on general English text and costs proportionally more
memory in the index and more time per query. On a corpus of a few thousand
chunks the small model is usually indistinguishable in practice.

The dimension is fixed at collection creation. Changing the embedding model
means recreating the collection and re-embedding every chunk; there is no
migration path, because the old and new vectors live in unrelated spaces.

## Normalisation

BGE returns L2-normalised vectors, meaning every vector has length one. When
vectors are normalised, cosine similarity and dot product give identical
rankings, so either distance metric works and the choice is cosmetic.

Normalisation also bounds the similarity to the range minus one to one, which
makes scores comparable across queries and lets you set an absolute threshold
below which a result is treated as no match.

## The query prefix

BGE was trained asymmetrically. Passages are embedded as plain text, while
retrieval queries are prefixed with an instruction, conventionally "Represent
this sentence for searching relevant passages:". The two sides of the training
objective were shaped differently, and the prefix tells the model which side it
is embedding.

Omitting the prefix costs recall, usually a little rather than catastrophically,
and the loss is invisible without an evaluation set because the system still
returns plausible-looking results. Applying the prefix to passages as well as
queries is the worse mistake, since it destroys the asymmetry entirely.

Not every library applies the prefix automatically, and some apply it only for
models they recognise. It is worth verifying rather than assuming: embed the
same string as a query and as a passage, and check the vectors differ.

## Local versus hosted

A local model costs nothing per call, keeps documents on your machine, and adds
no network latency. A hosted embedding API removes the install and is usually
faster per batch on large jobs.

Running locally through ONNX rather than PyTorch avoids a multi-gigabyte
dependency and is generally faster on CPU for single queries, at the cost of
being limited to models that have been converted.

## Context limits

Embedding models truncate silently. BGE accepts 512 tokens and discards
everything past that without warning, so a chunk that overruns the limit is
indexed only by its opening. This is the practical reason chunk size is measured
in tokens rather than characters, and why chunk size should sit below the model
limit rather than exactly at it.
