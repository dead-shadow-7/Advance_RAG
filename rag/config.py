from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
QDRANT_PATH = PROJECT_ROOT / "qdrant_data"
MODEL_CACHE = PROJECT_ROOT / ".model_cache"

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIM = 384  # must match the model; changing models means re-indexing
SPARSE_MODEL = "Qdrant/bm25"
COLLECTION = "chunks"

# "dense" | "sparse" | "hybrid" — hybrid fuses the other two with RRF.
RETRIEVAL_MODE = "hybrid"
# Each retriever returns this many candidates before fusion. Fusing two short
# lists mostly reproduces whichever was already right, so keep it well above
# the top_k that is finally returned.
CANDIDATES = 20
RRF_K = 60  # damps the top ranks; higher forgives one list's mistakes more

# A cross-encoder reads query and passage together instead of comparing two
# independent vectors. Far more accurate, far slower — so it only ever sees the
# shortlist the retrievers already produced.
# Off by default: on the current 9-chunk corpus reranking scores slightly worse
# than hybrid alone (MRR 0.823 vs 0.838) for ~100x the latency, because the
# shortlist is the whole corpus and there is nothing left to rescue. It clearly
# helps a weak ranking (dense alone goes 0.695 -> 0.823), so re-measure on a
# real corpus and flip this.
RERANK = False
RERANK_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"
RERANK_CANDIDATES = 20
# Cost scales with passage length: ~80ms per 512-token passage on CPU. Cutting
# passages to 256 tokens halved that but cost 0.12 MRR, because the answering
# sentence often sits past the halfway point of a 512-token chunk. Not worth it
# at this chunk size; revisit if chunks get smaller.
RERANK_MAX_TOKENS = 512

CHUNK_SIZE = 512
CHUNK_OVERLAP = 64

# Web search runs as a third retriever whose results never enter Qdrant.
# "never" | "always". A "fallback" mode — search only when the corpus looks too
# thin to answer — needs a sufficiency signal, and the honest one is the
# reranker's top score. That number is not measured yet, so the mode that would
# depend on guessing it does not exist.
WEB_SEARCH = "never"
WEB_RESULTS = 5
WEB_TIMEOUT = 10.0  # seconds; a slow search should not hold up a local answer
WEB_SEARCH_DEPTH = "basic"  # "advanced" digs deeper and costs 2 credits a call
# Tavily hands back a short query-relevant snippet per result, and the full page
# text only when asked. RRF ranks by position alone, so the length gap does not
# bias fusion — but it does bias the reranker and the LLM's context, and a
# snippet that stops just short of the answer is a miss no ranking can undo.
# "text" over "markdown": the markdown form keeps every nav link as
# [label](href), which burns tokens and buries the prose. Plain text still
# carries a few stray nav words at the top of a page, which is cheap by
# comparison. False falls back to the snippet, which is only the page's head.
WEB_RAW_CONTENT = "text"
# A single web page can chunk into dozens of passages, which would swamp a
# 9-chunk corpus in the fused list purely by arriving in bulk. Only the few
# chunks of each page that best match the query compete.
WEB_CHUNKS_PER_RESULT = 3
# Selection is a recall bottleneck, not a ranking: a chunk dropped here is one
# the reranker never gets to see. So keyword overlap keeps a margin above
# WEB_CHUNKS_PER_RESULT rather than deciding outright, and the embedder — when
# it runs at all — chooses between the survivors on meaning.
WEB_SELECT_CANDIDATES = 8
# Measured on a 5-result search (54 chunks): BM25 scores every chunk in 76ms,
# dense embedding costs ~180-226ms *each*, and passing threads=12 to fastembed
# changes nothing — that is simply what BGE-small costs at 512 tokens on this
# CPU. Refining 8 candidates a page across 5 pages therefore adds ~7s to every
# web query, which is why it is off until the eval says it buys recall worth
# paying for. Off, lexical overlap alone picks the chunks.
WEB_SELECT_DENSE = False

LLM_BASE_URL = "https://api.aicredits.in/v1"
LLM_MODEL = "qwen/qwen3.7-flash"
LLM_TEMPERATURE = 0.2  # grounded answers, not creative ones
LLM_MAX_TOKENS = 800

# Qwen3.7 is a hybrid reasoner that thinks by default: ~1850 reasoning tokens
# and 25s for a one-line answer, with no quality gain on grounded extraction.
# Note the gateway ignores Alibaba's native enable_thinking flag — only the
# reasoning field below actually turns it off.
LLM_REASONING = False
