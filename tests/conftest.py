"""Shared fixtures: a small Gutenberg-style book and a deterministic bag-of-words embedding."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import numpy as np
import pytest
from langchain_core.embeddings import Embeddings

REPO = Path(__file__).resolve().parents[1]
BOOK = REPO / "data" / "books" / "alice_in_wonderland.md"
QUESTIONS = REPO / "eval" / "questions.jsonl"

GARDEN = (
    "The three gardeners were busily painting the white roses red with their brushes. "
    "They had planted a white rose-tree by mistake, and the Queen would be angry. "
    "Alice watched the gardeners paint every rose on the tree a bright scarlet. "
)
SEA = (
    "Under the sea the lobsters dance a quadrille with the turtles and the seals. "
    "A whiting asked a snail to walk a little faster, for a porpoise was close behind. "
    "The lobsters were thrown far out to sea, and the dancers turned a somersault. "
)
SAMPLE_BOOK = (
    "The Project Gutenberg eBook of A Test Book\n\n"
    "This ebook is for the use of anyone anywhere under the terms of the licence.\n\n"
    "*** START OF THE PROJECT GUTENBERG EBOOK A TEST BOOK ***\n"
    "Contents\n\nCHAPTER I. The Garden\nCHAPTER II. The Sea\n\n"
    "CHAPTER I.\nThe Garden\n\n" + "\n\n".join([GARDEN] * 3) + "\n\n"
    "CHAPTER II.\nThe Sea\n\n" + "\n\n".join([SEA] * 3) + "\n\n"
    "*** END OF THE PROJECT GUTENBERG EBOOK A TEST BOOK ***\n\n"
    "Section 1. General terms of use and redistribution of Project Gutenberg works. "
    "Donations to the Project Gutenberg Literary Archive Foundation are gratefully accepted.\n"
)


class FakeEmbeddings(Embeddings):
    """Hashed bag of words: texts that share words get a high cosine similarity."""

    def __init__(self, dim: int = 256):
        self.dim = dim

    def _vector(self, text: str) -> list[float]:
        vector = np.zeros(self.dim)
        for word in re.findall(r"[a-z]+", text.lower()):
            vector[int(hashlib.md5(word.encode()).hexdigest(), 16) % self.dim] += 1.0
        norm = np.linalg.norm(vector)
        if norm == 0:
            vector[0], norm = 1.0, 1.0
        return (vector / norm).tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)


class FakeGenerator:
    name = "fake"

    def __init__(self, reply: str = "They painted the roses red."):
        self.reply = reply
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.reply


@pytest.fixture
def fake_embeddings() -> FakeEmbeddings:
    return FakeEmbeddings()


@pytest.fixture
def books_dir(tmp_path: Path) -> Path:
    folder = tmp_path / "books"
    folder.mkdir()
    # Written with CRLF line endings, like the real Gutenberg file.
    (folder / "test_book.md").write_bytes(SAMPLE_BOOK.replace("\n", "\r\n").encode("utf-8"))
    return folder


@pytest.fixture
def patch_embeddings(monkeypatch: pytest.MonkeyPatch) -> FakeEmbeddings:
    """Make every model name load the fake embedding, so no model is downloaded."""
    fake = FakeEmbeddings()
    monkeypatch.setattr(
        "rag_local.embeddings.load_embeddings", lambda model_id, device="auto", batch_size=32: fake
    )
    return fake
