"""The evaluation set: every gold quote is unique in the book and lands in a chunk."""

from __future__ import annotations

from collections import Counter

import pytest

from rag_local.documents import load_book, read_text, split_book
from rag_local.evaluation import contains_answer, find_quote, gold_chunk_ids, load_questions

from .conftest import BOOK, QUESTIONS

GRID_CHUNKING = [(500, 100), (1000, 200), (1000, 500), (2000, 400)]


@pytest.fixture(scope="module")
def book():
    return load_book(BOOK)


@pytest.fixture(scope="module")
def questions():
    return load_questions(QUESTIONS)


def test_counts_and_chapter_coverage(questions):
    on = [q for q in questions if not q.off_topic]
    assert len(on) == 40
    assert sum(q.off_topic for q in questions) == 6
    per_chapter = Counter(q.chapter for q in on)
    assert sorted(per_chapter) == list(range(1, 13))
    assert min(per_chapter.values()) >= 3


def test_each_quote_occurs_exactly_once(book, questions):
    raw = read_text(BOOK)
    for q in questions:
        if q.off_topic:
            continue
        spans = find_quote(raw, q.quote)
        assert len(spans) == 1, q.id
        start, end = spans[0]
        assert book.start <= start and end <= book.end, q.id
        assert book.chapter_at(start) == q.chapter, q.id
        assert contains_answer(raw[start:end], q.answer, q.aliases), q.id


@pytest.mark.parametrize("size,overlap", GRID_CHUNKING)
def test_every_quote_sits_whole_in_some_chunk(book, questions, size, overlap):
    chunks = split_book(book, size, overlap)
    for q in questions:
        if q.off_topic:
            continue
        span = find_quote(book.text, q.quote)[0]
        gold = gold_chunk_ids(chunks, book.path, span)
        assert gold, q.id
        by_id = {c.id: c for c in chunks}
        assert all(by_id[g].start <= span[0] and span[1] <= by_id[g].end for g in gold), q.id
