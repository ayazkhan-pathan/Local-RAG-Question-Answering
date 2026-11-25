"""Command line interface.

rag-local ingest [--preset small] [--model bge-base] [--chunk-size 1000] [--chunk-overlap 200]
rag-local query "What is the name of Alice's cat?" [--retrieve-only] [--generator local|hf]
rag-local compare apple iphone [--model bge-large]
rag-local evaluate [--questions eval/questions.jsonl]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import textwrap
from dataclasses import replace
from pathlib import Path

from . import __version__
from . import embeddings as emb
from .compare import compare
from .config import (
    BGE_QUERY_INSTRUCTION,
    DATA_DIR,
    DEFAULT_PRESET,
    EMBEDDING_MODELS,
    LOCAL_MODEL,
    PRESETS,
    QUESTIONS,
    STORE_DIR,
    Settings,
    preset,
)
from .documents import load_books
from .evaluation import evaluate_retrieval, load_questions
from .generators import MissingTokenError, make_generator
from .ingest import ingest
from .query import answer_question
from .store import Hit, open_store


def settings_from_args(args: argparse.Namespace) -> Settings:
    settings = preset(args.preset)
    overrides = {
        "embedding_model": args.model,
        "chunk_size": args.chunk_size,
        "chunk_overlap": args.chunk_overlap,
        "query_instruction": args.instruction,
        "k": getattr(args, "k", None),
        "threshold": getattr(args, "threshold", None),
    }
    given = {key: value for key, value in overrides.items() if value is not None}
    settings = replace(settings, **given)
    if args.keep_boilerplate:
        settings = replace(settings, strip_boilerplate=False)
    if settings.chunk_overlap >= settings.chunk_size:
        raise SystemExit("--chunk-overlap must be smaller than --chunk-size")
    return settings


def excerpt(text: str, width: int = 96) -> str:
    return textwrap.shorten(" ".join(text.split()), width=width, placeholder=" ...")


def print_hits(hits: list[Hit]) -> None:
    for number, hit in enumerate(hits, start=1):
        print(
            f"{number}. similarity {hit.score:.3f}  chapter {hit.chapter}  "
            f"characters {hit.start:,}-{hit.end:,} of {hit.source}"
        )
        print(f"   {excerpt(hit.text)}")


def cmd_ingest(args: argparse.Namespace) -> int:
    settings = settings_from_args(args)
    print(f"Settings: {settings.label()}")
    ingest(settings, args.data, args.store, args.device)
    return 0


def cmd_query(args: argparse.Namespace) -> int:
    store = open_store(args.store, device=args.device)
    k = args.k or store.settings.k
    threshold = store.settings.threshold if args.threshold is None else args.threshold
    if args.retrieve_only:
        hits = store.search(args.question, k)
        print_hits(hits)
        best = hits[0].score if hits else 0.0
        verdict = "would be answered" if best >= threshold else "would be refused"
        print(f"Best similarity {best:.3f} against a threshold of {threshold:.2f}: {verdict}.")
        return 0
    try:
        generator = make_generator(args.generator, args.llm, args.device, args.max_new_tokens)
    except MissingTokenError as error:
        print(error, file=sys.stderr)
        return 2
    answer = answer_question(store, args.question, generator, k, threshold)
    if not answer.passed:
        print(answer.text)
        return 0
    print(f"Answer: {answer.text}\n\nSources:")
    print_hits(answer.hits)
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    instruction = BGE_QUERY_INSTRUCTION if args.instruction else None
    embeddings = emb.make_embeddings(args.model, instruction, args.device)
    similarity, dimension = compare(args.text_a, args.text_b, embeddings)
    model_id = EMBEDDING_MODELS.get(args.model, args.model)
    print(f"{model_id}, {dimension} dimensions")
    print(f"Cosine similarity of {args.text_a!r} and {args.text_b!r}: {similarity:.4f}")
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    store = open_store(args.store, device=args.device)
    settings = store.settings
    books = load_books(args.data, strip_boilerplate=settings.strip_boilerplate)
    indexed = {book["path"]: book["sha256"] for book in store.manifest["books"]}
    changed = [book.path for book in books if indexed.get(book.path) != book.sha256]
    if changed:
        print(
            f"Warning: re-run ingest; these books changed since: {', '.join(changed)}",
            file=sys.stderr,
        )
    questions = load_questions(args.questions)
    k = args.k or settings.k
    threshold = settings.threshold if args.threshold is None else args.threshold
    summary, rows = evaluate_retrieval(store, questions, books, k, threshold)
    print(f"Store: {settings.label()}, {store.count()} chunks")
    print(f"On-topic questions: {summary['on_topic']}, off-topic: {summary['off_topic']}")
    for name in ("hit@1", "hit@3", "hit@5", "mrr"):
        print(f"  {name:<6} {summary[name]:.3f}")
    print(f"  mean context for k={k}: {summary['context_chars']:,.0f} characters")
    print(
        f"  best score >= {threshold:.2f}: on-topic {summary['on_topic_pass']:.0%}, "
        f"off-topic {summary['off_topic_pass']:.0%}"
    )
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "settings": settings.to_dict(),
            "summary": summary,
            "questions": [row.to_dict() for row in rows],
        }
        out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {out.as_posix()}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rag-local",
        description="Ask questions about the books in data/books, answered by a local LLM.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    def common(sub: argparse.ArgumentParser) -> None:
        sub.add_argument("--store", default=STORE_DIR, help="Chroma directory (default: chroma)")
        sub.add_argument("--device", default="auto", help="auto, cuda, mps or cpu")

    sub = commands.add_parser("ingest", help="split, embed and store the books")
    sub.add_argument("--preset", default=DEFAULT_PRESET, choices=sorted(PRESETS))
    sub.add_argument("--model", help="bge-small, bge-base, bge-large or a Hugging Face id")
    sub.add_argument("--chunk-size", type=int)
    sub.add_argument("--chunk-overlap", type=int)
    sub.add_argument(
        "--instruction",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="prefix queries with the BGE retrieval instruction",
    )
    sub.add_argument(
        "--keep-boilerplate",
        action="store_true",
        help="index the Project Gutenberg header and licence too",
    )
    sub.add_argument("--data", default=DATA_DIR, help="folder of .md books (default: data/books)")
    common(sub)
    sub.set_defaults(func=cmd_ingest)

    sub = commands.add_parser("query", help="answer a question from the store")
    sub.add_argument("question")
    sub.add_argument("-k", type=int, help="chunks to retrieve (default: from the store)")
    sub.add_argument("--threshold", type=float, help="minimum cosine similarity of the best chunk")
    sub.add_argument("--retrieve-only", action="store_true", help="show the chunks, no LLM")
    sub.add_argument(
        "--generator",
        default="local",
        choices=["local", "hf"],
        help="local transformers model (default) or Llama 3.1 8B via the HF API",
    )
    sub.add_argument("--llm", help=f"model id (default: {LOCAL_MODEL} for local)")
    sub.add_argument("--max-new-tokens", type=int, default=128)
    common(sub)
    sub.set_defaults(func=cmd_query)

    sub = commands.add_parser("compare", help="cosine similarity of two texts")
    sub.add_argument("text_a")
    sub.add_argument("text_b")
    sub.add_argument("--model", default="bge-small")
    sub.add_argument(
        "--instruction",
        action="store_true",
        help="embed both texts as queries with the BGE instruction",
    )
    sub.add_argument("--device", default="auto")
    sub.set_defaults(func=cmd_compare)

    sub = commands.add_parser("evaluate", help="retrieval metrics for the store")
    sub.add_argument("--questions", default=QUESTIONS)
    sub.add_argument("--data", default=DATA_DIR)
    sub.add_argument("-k", type=int)
    sub.add_argument("--threshold", type=float)
    sub.add_argument("--out", help="write per-question results to this JSON file")
    common(sub)
    sub.set_defaults(func=cmd_evaluate)
    return parser


def main(argv: list[str] | None = None) -> int:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except FileNotFoundError as error:
        print(error, file=sys.stderr)
        return 1
