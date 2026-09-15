# Evaluating Retrieval

Retrieval quality sets a ceiling on answer quality. A passage that is never
retrieved can never be cited, and no amount of prompt engineering recovers it.
So retrieval is measured first and separately.

## Recall@k

Recall@k asks a single question: does the correct passage appear anywhere in the
top k results? It is scored one or zero per query and averaged over the question
set. Recall@5 and Recall@10 are the usual pair.

Recall@5 is the number to watch when the top five chunks are what gets stuffed
into the prompt, because it exactly describes how often the model was given what
it needed. Recall@10 is diagnostic: a large gap between the two means the right
passage is being found but ranked badly, which is the specific problem a
reranker fixes.

Recall saturates on small corpora. With forty chunks indexed and k set to five,
a random retriever scores about twelve percent, and any working retriever scores
near one hundred. The metric stops discriminating long before the system is
good.

## Mean Reciprocal Rank

MRR keeps the rank rather than discarding it. Each query contributes one divided
by the position of the first correct result: a hit at rank one scores 1.0, at
rank two 0.5, at rank three 0.33. The mean across queries is the MRR.

Because it is sensitive to position, MRR keeps moving after Recall has pinned to
one. On a small evaluation set it is the more informative of the two, and it is
the metric that will show a reranker earning its latency.

## nDCG

Normalised Discounted Cumulative Gain handles the case where relevance is graded
rather than binary, and where several passages are relevant to different
degrees. It discounts each hit logarithmically by its position and divides by
the best achievable ordering. It is the right metric when you have graded
judgements, and overkill when you do not.

## Building the question set

Thirty to fifty questions is enough to see a real change and small enough to
write honestly in an afternoon. Aim for a spread:

- Questions whose answer uses the same words as the source text. These favour
  keyword search.
- Questions phrased entirely differently from the source. These favour dense
  retrieval.
- Questions naming an exact identifier, version or acronym.
- Questions the corpus genuinely cannot answer, to check the system declines
  rather than inventing an answer.

Record the expected source document and a distinctive phrase from the passage
that answers it. Matching on a phrase rather than a chunk identifier keeps the
question set valid when the chunk size changes, which it will.

## The discipline

Measure before the change and after it, on the same questions, and write both
numbers down. Retrieval changes are unusually prone to feeling like improvements
while making things worse, because the failure cases are invisible unless
counted. A change that improves MRR on twenty questions and costs eighty
milliseconds is a real decision, not an obvious one.
