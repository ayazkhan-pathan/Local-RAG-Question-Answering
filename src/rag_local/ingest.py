"""Build the vector store: load the books, split them, embed the chunks and save them in Chroma."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

from langchain_core.embeddings import Embeddings

from . import embeddings as emb
from .config import DATA_DIR, STORE_DIR, Settings
from .documents import load_books, split_books
from .store import Store, build_store


def ingest(
    settings: Settings,
    data_dir: str | Path = DATA_DIR,
    store_dir: str | Path = STORE_DIR,
    device: str = "auto",
    embeddings: Embeddings | None = None,
    log: Callable[[str], None] = print,
) -> Store:
    started = time.perf_counter()
    books = load_books(data_dir, strip_boilerplate=settings.strip_boilerplate)
    chunks = split_books(books, settings.chunk_size, settings.chunk_overlap)
    for book in books:
        kept = book.end - book.start
        log(
            f"{book.path}: kept {kept:,} of {len(book.text):,} characters, "
            f"{len(book.chapter_starts)} chapters"
        )
    log(
        f"Split into {len(chunks)} chunks of up to {settings.chunk_size} characters "
        f"({settings.chunk_overlap} overlap)."
    )
    if embeddings is None:
        log(f"Loading {settings.model_id} ...")
        embeddings = emb.make_embeddings(settings.model_id, settings.instruction, device)
    store = build_store(chunks, embeddings, store_dir, settings, books)
    log(
        f"Saved {store.count()} chunks to {Path(store_dir).as_posix()}/ with {settings.model_id} "
        f"in {time.perf_counter() - started:.1f} s."
    )
    return store
