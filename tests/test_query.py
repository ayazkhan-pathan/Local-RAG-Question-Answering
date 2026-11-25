from __future__ import annotations

import pytest

from rag_local.config import Settings
from rag_local.documents import load_books, split_books
from rag_local.query import CONTEXT_SEPARATOR, answer_question, build_prompt
from rag_local.store import Hit, memory_store

from .conftest import FakeGenerator


@pytest.fixture
def store(books_dir, fake_embeddings):
    books = load_books(books_dir)
    chunks = split_books(books, 200, 40)
    return memory_store(
        chunks, fake_embeddings, Settings("fake", 200, 40), books, collection="test_query"
    )


def test_prompt_wording():
    hits = [
        Hit("a", "First passage.", "b.md", 0, 14, 1, 0.9),
        Hit("b", "Second passage.", "b.md", 20, 35, 1, 0.8),
    ]
    prompt = build_prompt("Who?", hits)
    assert prompt.startswith("Answer the question based only on the following context:\n\n")
    assert "First passage." + CONTEXT_SEPARATOR + "Second passage." in prompt
    assert prompt.endswith("Answer the question based on the above context: Who?")


def test_answers_when_a_chunk_is_close(store):
    generator = FakeGenerator()
    answer = answer_question(store, "Who was painting the roses red?", generator, 3, 0.1)
    assert answer.passed
    assert answer.text == "They painted the roses red."
    assert len(generator.prompts) == 1
    assert "painting" in generator.prompts[0]
    assert answer.best_score == answer.hits[0].score


def test_refuses_below_the_threshold(store):
    generator = FakeGenerator()
    answer = answer_question(store, "Who was painting the roses red?", generator, 3, 0.999)
    assert not answer.passed
    assert generator.prompts == []
    assert "below the threshold of 1.00" in answer.text
    assert answer.prompt is None


def test_retrieval_without_a_generator(store):
    answer = answer_question(store, "a whiting and a snail", None, 2, 0.0)
    assert answer.passed and answer.text == ""
    assert len(answer.hits) == 2
    assert "whiting" in answer.prompt
