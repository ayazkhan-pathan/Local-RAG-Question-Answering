from __future__ import annotations

import pytest

from rag_local.documents import Chunk
from rag_local.evaluation import (
    QuestionResult,
    contains_answer,
    first_relevant_rank,
    gold_chunk_ids,
    hit_rate,
    is_refusal,
    mean_reciprocal_rank,
    normalise,
    summarise,
)


def test_ranks_and_rates():
    assert first_relevant_rank(["a", "b", "c"], {"c"}) == 3
    assert first_relevant_rank(["a", "b"], {"z"}) is None
    ranks = [1, 2, None, 5]
    assert hit_rate(ranks, 1) == 0.25
    assert hit_rate(ranks, 3) == 0.5
    assert hit_rate(ranks, 5) == 0.75
    assert mean_reciprocal_rank(ranks) == pytest.approx((1 + 0.5 + 0 + 0.2) / 4)
    assert hit_rate([], 3) == 0.0


def test_normalise():
    assert normalise("Où est ma Chatte?") == "ou est ma chatte"
    assert normalise("It’s six o’clock!") == normalise("It's six o'clock")
    assert normalise("writing-desk") == "writing desk"


@pytest.mark.parametrize(
    "output,answer,aliases,expected",
    [
        ("The jar was labelled ORANGE MARMALADE.", "ORANGE MARMALADE", [], True),
        ("He takes out his pocket watch.", "a watch", [], True),
        ("It is always six o’clock there.", "six o'clock", [], True),
        ("She says “Ou est ma chatte?”", "Où est ma chatte?", [], True),
        ("They often meet.", "ten", [], False),
        ("It was the Lizard.", "Bill", ["the Lizard"], True),
        ("I don't know.", "Dinah", [], False),
    ],
)
def test_contains_answer(output, answer, aliases, expected):
    assert contains_answer(output, answer, aliases) is expected


def test_refusals():
    assert is_refusal("I don’t know.")
    assert is_refusal("The context does not mention the boiling point of water.")
    assert not is_refusal("The Hatter is the first witness.")


def chunk(cid, start, end):
    return Chunk(cid, "x" * (end - start), "book.md", start, end, 1)


def test_gold_chunks_prefer_whole_quote():
    chunks = [chunk("a", 0, 100), chunk("b", 80, 180), chunk("c", 160, 260)]
    assert gold_chunk_ids(chunks, "book.md", (85, 95)) == ["a", "b"]
    assert gold_chunk_ids(chunks, "other.md", (85, 95)) == []


def test_gold_chunks_fall_back_to_largest_overlap():
    chunks = [chunk("a", 0, 100), chunk("b", 100, 200)]
    assert gold_chunk_ids(chunks, "book.md", (90, 130)) == ["b"]


def test_summary_pass_rates():
    rows = [
        QuestionResult("q1", False, ["a"], 1, 0.8, ["a"], 300),
        QuestionResult("q2", False, ["b"], 4, 0.4, ["c"], 500),
        QuestionResult("x1", True, [], None, 0.6, ["c"], 400),
        QuestionResult("x2", True, [], None, 0.3, ["a"], 400),
    ]
    summary = summarise(rows, k=3, threshold=0.5)
    assert summary["hit@1"] == 0.5
    assert summary["hit@5"] == 1.0
    assert summary["mrr"] == pytest.approx((1 + 0.25) / 2)
    assert summary["context_chars"] == 400
    assert summary["on_topic_pass"] == 0.5
    assert summary["off_topic_pass"] == 0.5
    assert (summary["on_topic"], summary["off_topic"]) == (2, 2)
