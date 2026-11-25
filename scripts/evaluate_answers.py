"""Closed-book versus RAG answers from a local model, and refusals on off-topic questions.

Each of the 40 book questions is asked twice: once with no context (closed book) and once
through the RAG pipeline (top-k chunks in the prompt). An answer counts as correct when the gold
answer or an accepted alias appears in it (see rag_local.evaluation.contains_answer). The six
off-topic questions go through the RAG pipeline only, to see whether it declines them.

Usage:
    python scripts/evaluate_answers.py                    # best grid setting, Qwen2.5-1.5B
    python scripts/evaluate_answers.py --preset original  # the first version's retrieval
    python scripts/evaluate_answers.py --model Qwen/Qwen2.5-0.5B-Instruct --out runs/try

Every generation is cached in runs/answers/generations.jsonl, so a rerun only does what is
missing. Writes results/answers.jsonl (every output), results/answers_summary.json and
results/hand_check.csv (a fixed sample to check by hand).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import time
from pathlib import Path

from rag_local import __version__
from rag_local.config import DATA_DIR, LOCAL_MODEL, QUESTIONS, Settings, preset
from rag_local.documents import load_books
from rag_local.evaluation import (
    contains_answer,
    gold_chunk_ids,
    is_refusal,
    load_questions,
    locate_quote,
)
from rag_local.generators import LocalGenerator
from rag_local.ingest import ingest
from rag_local.query import answer_question
from rag_local.store import open_store, read_manifest

CLOSED_BOOK_TEMPLATE = (
    "Answer this question about Lewis Carroll's Alice's Adventures in Wonderland: {question}"
)
HAND_CHECK_EVERY = 4  # q04, q08, ... q40 in both modes, plus every off-topic answer


class CachedGenerator:
    """Wraps a generator and stores each output by model and prompt in a JSONL file."""

    def __init__(self, generator: LocalGenerator, path: Path):
        self.generator = generator
        self.path = path
        self.name = generator.name
        self.cache: dict[str, dict] = {}
        self.new = 0
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                entry = json.loads(line)
                self.cache[entry["key"]] = entry

    def key(self, prompt: str) -> str:
        text = f"{self.generator.model_id}\x00{self.generator.max_new_tokens}\x00{prompt}"
        return hashlib.sha256(text.encode()).hexdigest()

    def generate(self, prompt: str) -> str:
        key = self.key(prompt)
        if key not in self.cache:
            started = time.perf_counter()
            output = self.generator.generate(prompt)
            entry = {"key": key, "output": output, "seconds": time.perf_counter() - started}
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry) + "\n")
            self.cache[key] = entry
            self.new += 1
        return self.cache[key]["output"]

    def seconds(self, prompt: str) -> float:
        return self.cache[self.key(prompt)]["seconds"]


def best_settings(results: Path) -> Settings:
    path = results / "retrieval_best.json"
    if not path.exists():
        raise SystemExit(f"{path.as_posix()} not found: run evaluate_retrieval.py or pass --preset")
    return Settings(**json.loads(path.read_text())["settings"])


def model_label(model: str) -> str:
    """The model id as given; for a local folder, the id from a Hugging Face cache path or the
    folder name, so no local path is written to the results."""
    path = Path(model)
    if not path.exists():
        return model
    for part in path.parts:
        if part.startswith("models--"):
            return part[len("models--") :].replace("--", "/")
    return path.name


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--preset", help="retrieval preset (default: best row of the grid)")
    parser.add_argument("--model", default=LOCAL_MODEL)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--data", default=DATA_DIR)
    parser.add_argument("--questions", default=QUESTIONS)
    parser.add_argument("--results", default="results", help="where retrieval_best.json is")
    parser.add_argument("--runs", default="runs/answers", help="store and generation cache")
    parser.add_argument("--out", default="results")
    args = parser.parse_args()

    settings = preset(args.preset) if args.preset else best_settings(Path(args.results))
    runs, out = Path(args.runs), Path(args.out)
    store_dir = runs / f"chroma_{settings.key()}"
    try:
        built = read_manifest(store_dir)["settings"] == settings.to_dict()
    except FileNotFoundError:
        built = False
    if not built:
        ingest(settings, args.data, store_dir, args.device)
    store = open_store(store_dir, device=args.device)
    books = load_books(args.data)
    chunks = store.chunks()
    questions = load_questions(args.questions)

    local = LocalGenerator(args.model, args.device, args.max_new_tokens)
    generator = CachedGenerator(local, runs / "generations.jsonl")
    label = model_label(args.model)
    records = []
    started = time.perf_counter()
    for number, question in enumerate(questions, start=1):
        sample = question.off_topic or number % HAND_CHECK_EVERY == 0
        base = {
            "id": question.id,
            "question": question.question,
            "answer": question.answer,
            "aliases": list(question.aliases),
            "hand_check": sample,
        }
        if not question.off_topic:
            prompt = CLOSED_BOOK_TEMPLATE.format(question=question.question)
            output = generator.generate(prompt)
            correct = contains_answer(output, question.answer, question.aliases)
            seconds = round(generator.seconds(prompt), 2)
            records.append(
                dict(
                    base,
                    mode="closed_book",
                    output=output,
                    correct=correct,
                    declined=is_refusal(output),
                    seconds=seconds,
                )
            )

        result = answer_question(
            store, question.question, generator, settings.k, settings.threshold
        )
        gold: list[str] = []
        if not question.off_topic:
            book, span = locate_quote(books, question.quote)
            gold = gold_chunk_ids(chunks, book.path, span)
        retrieved = [
            {"id": hit.id, "chapter": hit.chapter, "score": round(hit.score, 4)}
            for hit in result.hits
        ]
        record = dict(
            base,
            mode="rag",
            output=result.text,
            gated=not result.passed,
            declined=result.passed and is_refusal(result.text),
            best_score=None if result.best_score is None else round(result.best_score, 4),
            retrieved=retrieved,
            gold=gold,
            gold_retrieved=any(hit["id"] in gold for hit in retrieved),
            seconds=round(generator.seconds(result.prompt), 2) if result.passed else 0.0,
        )
        if not question.off_topic:
            record["correct"] = result.passed and contains_answer(
                result.text, question.answer, question.aliases
            )
        records.append(record)
        print(f"{question.id} {'off-topic' if question.off_topic else ''}".rstrip())

    on = [q for q in questions if not q.off_topic]
    closed = [r for r in records if r["mode"] == "closed_book"]
    rag_on = [r for r in records if r["mode"] == "rag" and r["answer"] is not None]
    rag_off = [r for r in records if r["mode"] == "rag" and r["answer"] is None]
    timed = [r["seconds"] for r in records if r["seconds"]]
    with_gold = [r for r in rag_on if r["gold_retrieved"]]
    summary = {
        "model": label,
        "max_new_tokens": args.max_new_tokens,
        "decoding": "greedy",
        "device": local.device,
        "retrieval": settings.to_dict(),
        "retrieval_label": settings.label(),
        "chunks": len(chunks),
        "questions": len(on),
        "closed_book_correct": sum(r["correct"] for r in closed),
        "rag_correct": sum(r["correct"] for r in rag_on),
        "rag_gold_retrieved": sum(r["gold_retrieved"] for r in rag_on),
        "rag_correct_when_gold_retrieved": sum(r["correct"] for r in with_gold),
        "rag_correct_without_gold": sum(r["correct"] for r in rag_on)
        - sum(r["correct"] for r in with_gold),
        "rag_on_topic_gated": sum(r["gated"] for r in rag_on),
        "off_topic": len(rag_off),
        "off_topic_gated": sum(r["gated"] for r in rag_off),
        "off_topic_declined_by_model": sum(r["declined"] for r in rag_off),
        "off_topic_answered": sum(not r["gated"] and not r["declined"] for r in rag_off),
        "mean_seconds_per_answer": round(sum(timed) / len(timed), 2) if timed else None,
        "new_generations": generator.new,
        "wall_seconds": round(time.perf_counter() - started, 1),
        "rag_local": __version__,
        "machine": f"{platform.system()} {platform.machine()}",
    }

    out.mkdir(parents=True, exist_ok=True)
    with open(out / "answers.jsonl", "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    (out / "answers_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with open(out / "hand_check.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            ["id", "mode", "question", "gold", "output", "auto_correct", "hand_correct"]
        )
        for r in records:
            if r["hand_check"]:
                row = [r["id"], r["mode"], r["question"], r["answer"] or "", r["output"]]
                writer.writerow([*row, r.get("correct", ""), ""])

    n = summary["questions"]
    print(f"\n{label}, retrieval {settings.label()}, k={settings.k}")
    print(f"closed book {summary['closed_book_correct']}/{n}, RAG {summary['rag_correct']}/{n}")
    print(
        f"off-topic: {summary['off_topic_gated']} refused by the threshold, "
        f"{summary['off_topic_declined_by_model']} declined by the model, "
        f"{summary['off_topic_answered']} answered"
    )


if __name__ == "__main__":
    main()
