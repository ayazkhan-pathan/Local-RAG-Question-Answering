from __future__ import annotations

import json

import numpy as np
import pytest

from rag_local.config import Settings
from rag_local.documents import load_books, split_books
from rag_local.store import (
    MANIFEST,
    build_store,
    client_settings,
    l2_relevance_from_cosine,
    memory_store,
    open_store,
)

SETTINGS = Settings("fake", 200, 40, query_instruction=False)


@pytest.fixture
def corpus(books_dir):
    books = load_books(books_dir)
    return books, split_books(books, SETTINGS.chunk_size, SETTINGS.chunk_overlap)


def test_build_reopen_and_search(tmp_path, corpus, fake_embeddings):
    books, chunks = corpus
    store_dir = tmp_path / "chroma"
    store = build_store(chunks, fake_embeddings, store_dir, SETTINGS, books)
    assert store.count() == len(chunks)

    reopened = open_store(store_dir, embeddings=fake_embeddings)
    assert reopened.count() == len(chunks)
    assert reopened.settings == SETTINGS
    assert [c.id for c in reopened.chunks()] == [c.id for c in chunks]

    query = "Who was painting the white roses red?"
    hits = reopened.search(query, 3)
    assert len(hits) == 3
    assert "painting" in hits[0].text
    assert hits[0].chapter == 1
    q = np.array(fake_embeddings.embed_query(query))
    by_id = {c.id: c for c in chunks}
    for hit in hits:
        expected = float(q @ np.array(fake_embeddings.embed_query(by_id[hit.id].text)))
        assert hit.score == pytest.approx(expected, abs=1e-5)
    assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)


def test_manifest_is_portable(tmp_path, corpus, fake_embeddings):
    books, chunks = corpus
    store_dir = tmp_path / "chroma"
    build_store(chunks, fake_embeddings, store_dir, SETTINGS, books)
    text = (store_dir / MANIFEST).read_text()
    manifest = json.loads(text)
    assert manifest["space"] == "cosine"
    assert manifest["chunks"] == len(chunks)
    assert manifest["books"][0]["path"] == "test_book.md"
    assert str(tmp_path) not in text
    assert '": "/' not in text  # no value is an absolute path


def test_rebuild_in_place(tmp_path, corpus, fake_embeddings):
    books, chunks = corpus
    store_dir = tmp_path / "chroma"
    build_store(chunks, fake_embeddings, store_dir, SETTINGS, books)
    store = build_store(chunks[:3], fake_embeddings, store_dir, SETTINGS, books)
    assert store.count() == 3


def test_refuses_to_replace_other_folders(tmp_path, corpus, fake_embeddings):
    books, chunks = corpus
    (tmp_path / "notes").mkdir()
    (tmp_path / "notes" / "keep.txt").write_text("mine")
    with pytest.raises(FileExistsError):
        build_store(chunks, fake_embeddings, tmp_path / "notes", SETTINGS, books)
    assert (tmp_path / "notes" / "keep.txt").exists()


def test_open_missing_store(tmp_path):
    with pytest.raises(FileNotFoundError):
        open_store(tmp_path / "nothing")


def test_l2_space_reproduces_langchain_relevance(corpus, fake_embeddings):
    """In Chroma's default L2 space a relevance of 0.5 means a cosine of about 0.65."""
    books, chunks = corpus
    cosine = memory_store(chunks, fake_embeddings, SETTINGS, books, "cosine", "test_cos")
    l2 = memory_store(chunks, fake_embeddings, SETTINGS, books, "l2", "test_l2")
    query = "lobsters dance a quadrille"
    cosine_scores = {hit.id: hit.score for hit in cosine.search(query, len(chunks))}
    for hit in l2.search(query, len(chunks)):
        assert hit.score == pytest.approx(l2_relevance_from_cosine(cosine_scores[hit.id]), abs=1e-5)
    assert l2_relevance_from_cosine(1 - 0.5 / np.sqrt(2)) == pytest.approx(0.5)


def test_telemetry_is_off():
    assert client_settings().anonymized_telemetry is False
