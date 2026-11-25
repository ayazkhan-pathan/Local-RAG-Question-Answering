"""Questions with gold quotes, retrieval metrics (hit@k, MRR) and answer matching."""

from __future__ import annotations

import json
import re
import unicodedata
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from .documents import Book, Chunk
from .embeddings import embed_queries
from .store import Hit, Store


@dataclass(frozen=True)
class Question:
    id: str
    question: str
    answer: str | None = None
    aliases: tuple[str, ...] = ()
    quote: str | None = None
    chapter: int | None = None

    @property
    def off_topic(self) -> bool:
        return self.answer is None


def load_questions(path: str | Path) -> list[Question]:
    questions = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("off_topic"):
                questions.append(Question(row["id"], row["question"]))
            else:
                questions.append(
                    Question(
                        row["id"],
                        row["question"],
                        row["answer"],
                        tuple(row.get("aliases", ())),
                        row["quote"],
                        row.get("chapter"),
                    )
                )
    ids = [q.id for q in questions]
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate question ids in {Path(path).name}")
    return questions


def find_quote(text: str, quote: str) -> list[tuple[int, int]]:
    """Every place ``quote`` occurs in ``text``; any run of whitespace matches any other."""
    pattern = r"\s+".join(re.escape(token) for token in quote.split())
    return [match.span() for match in re.finditer(pattern, text)]


def locate_quote(books: Sequence[Book], quote: str) -> tuple[Book, tuple[int, int]]:
    """The book and span of a quote that must occur exactly once inside the books' bodies."""
    found = [
        (book, span)
        for book in books
        for span in find_quote(book.text, quote)
        if book.start <= span[0] and span[1] <= book.end
    ]
    if len(found) != 1:
        raise ValueError(f"quote occurs {len(found)} times, expected once: {quote!r}")
    return found[0]


def gold_chunk_ids(chunks: Iterable[Chunk], source: str, span: tuple[int, int]) -> list[str]:
    """Chunks that contain the whole quote; failing that, those that overlap it most."""
    start, end = span
    candidates = [chunk for chunk in chunks if chunk.source == source]
    whole = [chunk.id for chunk in candidates if chunk.start <= start and end <= chunk.end]
    if whole:
        return whole
    overlap = {chunk.id: min(end, chunk.end) - max(start, chunk.start) for chunk in candidates}
    best = max(overlap.values(), default=0)
    return [cid for cid, value in overlap.items() if value == best and value > 0]


def first_relevant_rank(ranked_ids: Sequence[str], relevant: Iterable[str]) -> int | None:
    """1-based rank of the first relevant id, or None if none was retrieved."""
    relevant = set(relevant)
    for rank, cid in enumerate(ranked_ids, start=1):
        if cid in relevant:
            return rank
    return None


def hit_rate(ranks: Sequence[int | None], k: int) -> float:
    return sum(rank is not None and rank <= k for rank in ranks) / len(ranks) if ranks else 0.0


def mean_reciprocal_rank(ranks: Sequence[int | None]) -> float:
    return sum(1.0 / rank for rank in ranks if rank) / len(ranks) if ranks else 0.0


_NON_ALNUM = re.compile(r"[^0-9a-z]+")
_ARTICLE = re.compile(r"^(?:a|an|the) ")


def normalise(text: str) -> str:
    """Lower case, accents and punctuation removed, single spaces: "O’Clock!" -> "o clock"."""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = text.lower().replace("&", " and ")
    return _NON_ALNUM.sub(" ", text).strip()


def contains_answer(output: str, answer: str, aliases: Sequence[str] = ()) -> bool:
    """True if the gold answer or an alias appears in ``output`` as whole words.

    A leading article on the gold answer is optional, so "a watch" also matches "his watch".
    """
    haystack = f" {normalise(output)} "
    for candidate in (answer, *aliases):
        needle = _ARTICLE.sub("", normalise(candidate))
        if needle and f" {needle} " in haystack:
            return True
    return False


_REFUSALS = [
    normalise(phrase)
    for phrase in (
        "don't know",
        "do not know",
        "not mentioned",
        "no mention",
        "does not mention",
        "doesn't mention",
        "no information",
        "not provided",
        "does not provide",
        "doesn't provide",
        "cannot answer",
        "can't answer",
        "unable to answer",
        "not in the context",
        "does not contain",
        "doesn't contain",
        "not specified",
        "not enough information",
        "cannot be determined",
        "does not say",
        "doesn't say",
        "is not stated",
        "not stated",
    )
]


def is_refusal(output: str) -> bool:
    """A rough check that a model declined to answer; flagged answers are checked by hand."""
    text = f" {normalise(output)} "
    return any(f" {phrase} " in text for phrase in _REFUSALS)


@dataclass
class QuestionResult:
    id: str
    off_topic: bool
    gold: list[str]
    rank: int | None
    best_score: float | None
    top_ids: list[str]
    context_chars: int

    def to_dict(self) -> dict:
        return dict(self.__dict__)


def rank_all(store: Store, vectors: Sequence[Sequence[float]]) -> list[list[Hit]]:
    """Every chunk for each query vector, best first."""
    count = store.count()
    return [store.search_vector(list(vector), count) for vector in vectors]


def evaluate_retrieval(
    store: Store,
    questions: Sequence[Question],
    books: Sequence[Book],
    k: int = 3,
    threshold: float = 0.5,
    query_vectors: Sequence[Sequence[float]] | None = None,
    ks: Sequence[int] = (1, 3, 5),
) -> tuple[dict, list[QuestionResult]]:
    """Retrieval metrics for a store.

    hit@k and MRR use on-topic questions, with the full ranking for MRR. ``context_chars`` is the
    length of the ``k`` chunks a question would hand to the model. The pass rates are the shares
    of on-topic and off-topic questions whose best score reaches ``threshold``.
    """
    chunks = store.chunks()
    if query_vectors is None:
        query_vectors = embed_queries(store.embeddings, [q.question for q in questions])
    results = []
    for question, hits in zip(questions, rank_all(store, query_vectors)):
        gold: list[str] = []
        if not question.off_topic:
            book, span = locate_quote(books, question.quote)
            gold = gold_chunk_ids(chunks, book.path, span)
        ranked = [hit.id for hit in hits]
        results.append(
            QuestionResult(
                id=question.id,
                off_topic=question.off_topic,
                gold=gold,
                rank=first_relevant_rank(ranked, gold) if gold else None,
                best_score=hits[0].score if hits else None,
                top_ids=ranked[:5],
                context_chars=sum(len(hit.text) for hit in hits[:k]),
            )
        )
    return summarise(results, k, threshold, ks), results


def summarise(
    results: Sequence[QuestionResult], k: int, threshold: float, ks: Sequence[int] = (1, 3, 5)
) -> dict:
    on = [r for r in results if not r.off_topic]
    off = [r for r in results if r.off_topic]
    ranks = [r.rank for r in on]

    def passing(rows: Sequence[QuestionResult]) -> float:
        scores = [r.best_score for r in rows]
        return sum(s is not None and s >= threshold for s in scores) / len(rows) if rows else 0.0

    summary = {f"hit@{n}": hit_rate(ranks, n) for n in ks}
    summary.update(
        {
            "mrr": mean_reciprocal_rank(ranks),
            "context_chars": sum(r.context_chars for r in on) / len(on) if on else 0.0,
            "on_topic_pass": passing(on),
            "off_topic_pass": passing(off),
            "on_topic": len(on),
            "off_topic": len(off),
            "k": k,
            "threshold": threshold,
        }
    )
    return summary
