# Hybrid Search

Dense vector search and keyword search fail in different ways, which is the
whole argument for running both.

## Where dense retrieval breaks

An embedding compresses a passage into a few hundred floats. That is excellent
for paraphrase: a query asking "how do I stop the model inventing facts" will
match a passage about hallucination without sharing a single content word.

It is poor at exact tokens. Product codes, error identifiers, function names,
version strings and rare proper nouns get smeared into the same region of space
as their neighbours. A query for `cl100k_base` may rank a passage about
tokenizers in general above the one passage that names it, because the model has
no strong representation for a string it effectively never saw in training.

## Where keyword retrieval breaks

BM25 scores a document by how often the query terms appear in it, damped by how
common those terms are across the corpus. The damping is inverse document
frequency, or IDF: a term appearing in every document carries almost no signal,
while a rare term is highly discriminative.

BM25 is exact. It cannot match "car" to "automobile", and it cannot tell that a
question about cost is answered by a passage discussing pricing. Any query
phrased differently from the source text scores zero on the terms that matter.

## Reciprocal Rank Fusion

The two systems return different lists with incomparable scores. A cosine
similarity of 0.62 and a BM25 score of 14.3 cannot be averaged in any principled
way, because they are not on the same scale and their distributions shift with
every query.

Reciprocal Rank Fusion sidesteps the problem by throwing the scores away and
keeping only the ranks. Each document gets a score of one divided by the sum of
a constant and its rank in a list, summed across every list it appears in:

    score(d) = sum over lists of  1 / (k + rank(d))

The constant k is conventionally 60. It damps the influence of the very top
ranks, so a document that placed second in both lists can outrank a document
that placed first in one and nowhere in the other. Raising k flattens the
weighting further and makes the fusion more forgiving of a single list's
mistakes; lowering it trusts the top of each list more.

RRF needs no tuning, no score normalisation, and no training data. That is why
it is usually the first fusion method worth trying, and often the last.

## Practical notes

Retrieve more candidates from each retriever than you intend to keep. Fusing the
top 20 from each list and then taking the best 5 gives the fusion room to
promote a document that only one retriever found. Fusing two lists of 5 mostly
reproduces whichever list was already correct.

Sparse vectors are the usual way to store BM25 in a vector database. Each
dimension corresponds to a token, and only the tokens actually present carry a
value, so the vector is almost entirely zeros and can be indexed efficiently
alongside the dense vector on the same point.
