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

LLM_BASE_URL = "https://api.aicredits.in/v1"
LLM_MODEL = "qwen/qwen3.7-flash"
LLM_TEMPERATURE = 0.2  # grounded answers, not creative ones
LLM_MAX_TOKENS = 800

# Qwen3.7 is a hybrid reasoner that thinks by default: ~1850 reasoning tokens
# and 25s for a one-line answer, with no quality gain on grounded extraction.
# Note the gateway ignores Alibaba's native enable_thinking flag — only the
# reasoning field below actually turns it off.
LLM_REASONING = False
