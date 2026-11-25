"""Compare retrieval settings on the 40 questions (and 6 off-topic ones) in eval/questions.jsonl.

The grid is three BGE models x four chunkings x the query instruction off and on, plus one row
that reproduces the first version of this repository: the Project Gutenberg header and licence
indexed, Chroma's default L2 space and LangChain's relevance score.

Usage:
    python scripts/evaluate_retrieval.py                        # the whole grid
    python scripts/evaluate_retrieval.py --models bge-small     # one model only
    python scripts/evaluate_retrieval.py --threshold 0.6

Chunk and question embeddings are cached in runs/cache/ and each setting's result in
runs/retrieval/, so an interrupted run carries on where it stopped. Writes
results/retrieval_grid.csv, results/retrieval_questions.csv, results/retrieval_best.json and,
for the best setting, results/book_map.csv (a 2-D PCA projection of the chunk embeddings).
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from langchain_core.embeddings import Embeddings

from rag_local import embeddings as emb
from rag_local.config import BGE_QUERY_INSTRUCTION, DATA_DIR, EMBEDDING_MODELS, QUESTIONS, Settings
from rag_local.documents import load_books, split_books
from rag_local.evaluation import evaluate_retrieval, load_questions
from rag_local.store import memory_store

CHUNKINGS = [(500, 100), (1000, 200), (1000, 500), (2000, 400)]
ORIGINAL = Settings("bge-large", 1000, 500, query_instruction=False, strip_boilerplate=False)
GRID_COLUMNS = [
    "setting", "model", "chunk_size", "chunk_overlap", "instruction", "boilerplate", "score",
    "chunks", "over_512_tokens", "hit@1", "hit@3", "hit@5", "mrr", "context_chars",
    "on_topic_pass", "off_topic_pass", "threshold", "top5_exact", "embed_seconds", "best",
]  # fmt: skip


class Precomputed(Embeddings):
    """Hands back vectors that were computed (or cached) earlier, keyed by text."""

    def __init__(self, texts: list[str], vectors: np.ndarray):
        self.vectors = dict(zip(texts, vectors.tolist()))

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.vectors[text] for text in texts]

    def embed_query(self, text: str) -> list[float]:
        raise NotImplementedError("queries are embedded in advance")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cached_vectors(path: Path, texts: list[str], encode) -> tuple[np.ndarray, float]:
    """Vectors for ``texts`` from ``path`` if it holds exactly these texts, else encode and save."""
    digest = hashlib.sha256("\x00".join(texts).encode()).hexdigest()
    if path.exists():
        data = np.load(path)
        if str(data["digest"]) == digest:
            return data["vectors"], float(data["seconds"])
    started = time.perf_counter()
    vectors = np.asarray(encode(texts), dtype=np.float32)
    seconds = time.perf_counter() - started
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, vectors=vectors, digest=digest, seconds=seconds)
    return vectors, seconds


def count_long_chunks(texts: list[str], model_id: str, limit: int = 512) -> int:
    """Chunks longer than the model's 512-token window (they are truncated when embedded)."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    encoded = tokenizer(texts, add_special_tokens=True, verbose=False)["input_ids"]
    lengths = [len(ids) for ids in encoded]
    return sum(length > limit for length in lengths)


def free_accelerator_memory() -> None:
    import torch

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def pca_2d(vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean, the first two principal axes and their shares of the variance."""
    mean = vectors.mean(axis=0)
    _, singular, axes = np.linalg.svd(vectors - mean, full_matrices=False)
    share = singular**2 / np.sum(singular**2)
    return mean, axes[:2], share[:2]


def nearest_same_chapter(chunks, vectors: np.ndarray) -> tuple[float, float]:
    """Share of chunks whose most similar non-overlapping chunk is in the same chapter, and the
    share expected if that neighbour were picked at random. Front matter is left out."""
    starts = np.array([c.start for c in chunks])
    ends = np.array([c.end for c in chunks])
    chapters = np.array([c.chapter for c in chunks])
    similarity = vectors @ vectors.T
    same, chance = [], []
    for i in np.flatnonzero(chapters > 0):
        eligible = (chapters > 0) & ((ends <= starts[i]) | (starts >= ends[i]))
        if not eligible.any():
            continue
        j = np.flatnonzero(eligible)[np.argmax(similarity[i, eligible])]
        same.append(chapters[j] == chapters[i])
        chance.append(np.mean(chapters[eligible] == chapters[i]))
    return float(np.mean(same)), float(np.mean(chance))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--models", nargs="+", default=list(EMBEDDING_MODELS))
    parser.add_argument("--no-original", action="store_true", help="skip the first-version row")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("-k", type=int, default=3, help="chunks handed to the model")
    parser.add_argument("--data", default=DATA_DIR)
    parser.add_argument("--questions", default=QUESTIONS)
    parser.add_argument("--runs", default="runs", help="cache folder")
    parser.add_argument("--out", default="results", help="output folder")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    runs, out = Path(args.runs), Path(args.out)
    questions = load_questions(args.questions)
    question_texts = [q.question for q in questions]
    base_fingerprint = {
        "questions": sha256(Path(args.questions)),
        "books": [sha256(p) for p in sorted(Path(args.data).glob("*.md"))],
        "threshold": args.threshold,
        "k": args.k,
    }
    plan: list[tuple[Settings, str]] = [
        (Settings(model, size, overlap, instruction), "cosine")
        for model in args.models
        for size, overlap in CHUNKINGS
        for instruction in (False, True)
    ]
    if "bge-large" in args.models and not args.no_original:
        plan.append((ORIGINAL, "l2"))

    rows, per_question, long_counts = [], [], {}
    loaded: tuple[str, Embeddings] | None = None
    for settings, space in plan:
        fingerprint = dict(base_fingerprint, settings=settings.to_dict(), space=space)
        result_path = runs / "retrieval" / f"{settings.key()}.json"
        if result_path.exists():
            cached = json.loads(result_path.read_text())
            if cached["fingerprint"] == fingerprint:
                print(f"cached   {settings.label()}")
                rows.append(cached["row"])
                per_question.extend(cached["questions"])
                continue

        if loaded is None or loaded[0] != settings.embedding_model:
            loaded = None
            gc.collect()
            free_accelerator_memory()
            print(f"loading  {settings.model_id}")
            loaded = (settings.embedding_model, emb.load_embeddings(settings.model_id, args.device))
        base = loaded[1]

        books = load_books(args.data, strip_boilerplate=settings.strip_boilerplate)
        chunks = split_books(books, settings.chunk_size, settings.chunk_overlap)
        texts = [chunk.text for chunk in chunks]
        tag = f"{settings.chunk_size}-{settings.chunk_overlap}"
        tag += "" if settings.strip_boilerplate else "_raw"
        cache = runs / "cache" / settings.embedding_model
        chunk_vecs, seconds = cached_vectors(
            cache / f"chunks_{tag}.npz", texts, base.embed_documents
        )
        prefix = BGE_QUERY_INSTRUCTION if settings.query_instruction else ""
        query_vecs, _ = cached_vectors(
            cache / f"questions_{'instr' if prefix else 'plain'}.npz",
            [prefix + text for text in question_texts],
            base.embed_documents,
        )
        if tag not in long_counts:
            long_counts[tag] = count_long_chunks(texts, settings.model_id)

        store = memory_store(
            chunks, Precomputed(texts, chunk_vecs), settings, books, space, f"grid_{settings.key()}"
        )
        summary, results = evaluate_retrieval(
            store, questions, books, args.k, args.threshold, query_vectors=query_vecs.tolist()
        )
        store.vectorstore.delete_collection()
        exact = np.argsort(-(query_vecs @ chunk_vecs.T), axis=1)[:, :5]
        agree = [set(r.top_ids) == {chunks[j].id for j in row} for r, row in zip(results, exact)]

        row = {
            "setting": settings.key(),
            "model": settings.embedding_model,
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap,
            "instruction": settings.query_instruction,
            "boilerplate": not settings.strip_boilerplate,
            "score": "cosine" if space == "cosine" else "l2 relevance",
            "chunks": len(chunks),
            "over_512_tokens": long_counts[tag],
            **{name: summary[name] for name in ("hit@1", "hit@3", "hit@5", "mrr")},
            "context_chars": summary["context_chars"],
            "on_topic_pass": summary["on_topic_pass"],
            "off_topic_pass": summary["off_topic_pass"],
            "threshold": args.threshold,
            "top5_exact": float(np.mean(agree)),
            "embed_seconds": round(seconds, 1),
        }
        question_rows = [
            {
                "setting": settings.key(),
                "id": r.id,
                "off_topic": r.off_topic,
                "rank": r.rank,
                "best_score": r.best_score,
                "gold": " ".join(r.gold),
                "top_ids": " ".join(r.top_ids),
            }
            for r in results
        ]
        result_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"fingerprint": fingerprint, "row": row, "questions": question_rows}
        result_path.write_text(json.dumps(payload, indent=1))
        rows.append(row)
        per_question.extend(question_rows)
        print(
            f"done     {settings.label():<36} hit@1 {row['hit@1']:.3f}  hit@3 {row['hit@3']:.3f}"
            f"  MRR {row['mrr']:.3f}  ({len(chunks)} chunks, {seconds:.0f} s)"
        )

    candidates = [row for row in rows if row["score"] == "cosine"]
    best = max(candidates, key=lambda r: (r["hit@3"], r["mrr"], -r["context_chars"]))
    for row in rows:
        row["best"] = row is best

    out.mkdir(parents=True, exist_ok=True)
    with open(out / "retrieval_grid.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=GRID_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {key: round(v, 4) if isinstance(v, float) else v for key, v in row.items()}
            )
    with open(out / "retrieval_questions.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_question[0]), lineterminator="\n")
        writer.writeheader()
        for row in per_question:
            score = row["best_score"]
            writer.writerow(dict(row, best_score=None if score is None else round(score, 4)))

    best_settings = next(s for s, space in plan if s.key() == best["setting"] and space == "cosine")
    tag = f"{best_settings.chunk_size}-{best_settings.chunk_overlap}"
    cache = runs / "cache" / best_settings.embedding_model
    chunk_vecs = np.load(cache / f"chunks_{tag}.npz")["vectors"]
    kind = "instr" if best_settings.query_instruction else "plain"
    query_vecs = np.load(cache / f"questions_{kind}.npz")["vectors"]
    books = load_books(args.data)
    chunks = split_books(books, best_settings.chunk_size, best_settings.chunk_overlap)
    same_chapter, chance = nearest_same_chapter(chunks, chunk_vecs)
    mean, axes, share = pca_2d(chunk_vecs)
    chunk_xy = (chunk_vecs - mean) @ axes.T
    query_xy = (query_vecs - mean) @ axes.T
    with open(out / "book_map.csv", "w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["kind", "id", "chapter", "x", "y"])
        for chunk, (x, y) in zip(chunks, chunk_xy):
            writer.writerow(["chunk", chunk.id, chunk.chapter, f"{x:.5f}", f"{y:.5f}"])
        for question, (x, y) in zip(questions, query_xy):
            chapter = "" if question.chapter is None else question.chapter
            writer.writerow(["question", question.id, chapter, f"{x:.5f}", f"{y:.5f}"])
    summary = {
        "best": best,
        "settings": best_settings.to_dict(),
        "label": best_settings.label(),
        "rule": "highest hit@3, then highest MRR, then the least context",
        "pca_variance_share": [round(float(v), 4) for v in share],
        "nearest_same_chapter": round(same_chapter, 4),
        "nearest_same_chapter_by_chance": round(chance, 4),
    }
    (out / "retrieval_best.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(
        f"\nBest setting: {best_settings.label()} "
        f"(hit@3 {best['hit@3']:.3f}, MRR {best['mrr']:.3f})"
    )
    print(f"Wrote {out.as_posix()}/retrieval_grid.csv, retrieval_questions.csv, book_map.csv")


if __name__ == "__main__":
    main()
