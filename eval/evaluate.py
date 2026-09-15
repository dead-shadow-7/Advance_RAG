"""Measure retrieval quality across modes.

Run with:  python -m eval.evaluate

Builds its own in-memory index from data/, so it never competes with the API
for the on-disk Qdrant lock and always measures the current code and config.
"""

import json
import re
import time
from collections import defaultdict

from qdrant_client import QdrantClient

from rag.chunker import chunk_documents
from rag.config import CHUNK_OVERLAP, CHUNK_SIZE, DATA_DIR, PROJECT_ROOT
from rag.embedder import embed_passages, embed_passages_sparse
from rag.loader import load_directory
from rag.pipeline import search_points
from rag.store import create_collection, upsert_chunks

QUESTIONS_PATH = PROJECT_ROOT / "eval" / "questions.json"
MODES = ("dense", "sparse", "hybrid")
DEPTH = 10


def build_index() -> tuple[QdrantClient, int]:
    docs = load_directory(DATA_DIR)
    chunks = chunk_documents(docs, CHUNK_SIZE, CHUNK_OVERLAP)
    texts = [chunk.text for chunk in chunks]

    client = QdrantClient(":memory:")
    create_collection(client)
    upsert_chunks(client, chunks, embed_passages(texts), embed_passages_sparse(texts))
    return client, len(chunks)


def normalise(text: str) -> str:
    """Source prose is hard-wrapped, so a phrase can straddle a newline."""
    return re.sub(r"\s+", " ", text).strip().lower()


def validate(questions: list[dict]) -> list[str]:
    """A question whose phrase is not in its source can never be scored."""
    corpus = {doc.source: "" for doc in load_directory(DATA_DIR)}
    for doc in load_directory(DATA_DIR):
        corpus[doc.source] += doc.text
    problems = []
    for question in questions:
        text = corpus.get(question["source"])
        if text is None:
            problems.append(f"{question['question'][:45]!r} -> no such source")
        elif normalise(question["contains"]) not in normalise(text):
            problems.append(f"{question['question'][:45]!r} -> phrase not in source")
    return problems


def first_hit_rank(scored: list[tuple], source: str, phrase: str) -> int | None:
    needle = normalise(phrase)
    for rank, (point, _) in enumerate(scored, start=1):
        payload = point.payload
        if payload["source"] == source and needle in normalise(payload["text"]):
            return rank
    return None


def score(ranks: list[int | None]) -> dict[str, float]:
    total = len(ranks)
    return {
        "recall@5": sum(r is not None and r <= 5 for r in ranks) / total,
        "recall@10": sum(r is not None and r <= 10 for r in ranks) / total,
        "mrr": sum(1 / r for r in ranks if r) / total,
    }


def main() -> None:
    questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))

    problems = validate(questions)
    if problems:
        print("Broken questions — these can never be scored:")
        for problem in problems:
            print("  " + problem)
        print()

    client, chunk_count = build_index()
    print(f"{chunk_count} chunks indexed, {len(questions)} questions\n")

    results, timings = {}, {}
    for mode in MODES:
        started = time.time()
        results[mode] = [
            first_hit_rank(
                search_points(client, q["question"], DEPTH, mode=mode),
                q["source"],
                q["contains"],
            )
            for q in questions
        ]
        timings[mode] = (time.time() - started) / len(questions) * 1000

    print(f"{'mode':8} {'recall@5':>9} {'recall@10':>10} {'MRR':>7} {'ms/query':>9}")
    print("-" * 47)
    for mode in MODES:
        metrics = score(results[mode])
        print(
            f"{mode:8} {metrics['recall@5']:>9.2f} {metrics['recall@10']:>10.2f} "
            f"{metrics['mrr']:>7.3f} {timings[mode]:>9.0f}"
        )

    # The split that matters: keyword search should win on exact identifiers,
    # dense on questions worded nothing like the source.
    by_kind = defaultdict(list)
    for index, question in enumerate(questions):
        by_kind[question["kind"]].append(index)

    print(f"\n{'kind':12} {'n':>3}" + "".join(f"{mode:>10}" for mode in MODES) + "   (MRR)")
    print("-" * 47)
    for kind, indexes in sorted(by_kind.items()):
        row = "".join(
            f"{score([results[mode][i] for i in indexes])['mrr']:>10.3f}"
            for mode in MODES
        )
        print(f"{kind:12} {len(indexes):>3}{row}")

    print(f"\n{'question':52}" + "".join(f"{mode:>8}" for mode in MODES) + "   (rank)")
    print("-" * 78)
    for index, question in enumerate(questions):
        row = "".join(
            f"{results[mode][index] or '-':>8}" for mode in MODES
        )
        print(f"{question['question'][:50]:52}{row}")

    misses = [
        questions[i]["question"]
        for i, rank in enumerate(results["hybrid"])
        if rank is None
    ]
    if misses:
        print(f"\nhybrid missed {len(misses)} of {len(questions)}:")
        for miss in misses:
            print("  " + miss)


if __name__ == "__main__":
    main()
