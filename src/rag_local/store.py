"""A Chroma vector store in cosine space, with a manifest that records how it was built.

Chroma's default space is squared L2. LangChain turns that distance into a "relevance" of
1 - d / sqrt(2), which for unit vectors is 1 - sqrt(2) * (1 - cosine), so a relevance cut-off of
0.5 is really a cosine cut-off of about 0.65. Building the collection in cosine space makes the
score the cosine similarity itself.
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

from langchain_core.embeddings import Embeddings

from . import embeddings as emb
from .config import Settings
from .documents import Book, Chunk

os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

COLLECTION = "books"
MANIFEST = "manifest.json"
FORMAT = 1


def client_settings():
    from chromadb.config import Settings as ChromaSettings

    return ChromaSettings(anonymized_telemetry=False)


def distance_to_score(distance: float, space: str) -> float:
    """Chroma distance to a similarity score: cosine similarity, or LangChain's L2 relevance."""
    if space == "cosine":
        return 1.0 - distance
    if space == "l2":
        return 1.0 - distance / math.sqrt(2)
    raise ValueError(f"unsupported space {space!r}")


def l2_relevance_from_cosine(cosine: float) -> float:
    """LangChain's relevance score for Chroma's default L2 space, for unit vectors."""
    return 1.0 - math.sqrt(2) * (1.0 - cosine)


@dataclass(frozen=True)
class Hit:
    id: str
    text: str
    source: str
    start: int
    end: int
    chapter: int
    score: float


def _clear_chroma_cache() -> None:
    # Chroma caches one client per path; a stale entry breaks a store rebuilt at the same path.
    with contextlib.suppress(ImportError, AttributeError):
        from chromadb.api.client import SharedSystemClient

        SharedSystemClient.clear_system_cache()


def new_vectorstore(
    embeddings: Embeddings,
    store_dir: str | Path | None = None,
    space: str = "cosine",
    collection: str = COLLECTION,
):
    """A LangChain Chroma store on disk, or in memory when ``store_dir`` is None."""
    import chromadb
    from langchain_chroma import Chroma

    if store_dir is None:
        client = chromadb.EphemeralClient(settings=client_settings())
    else:
        client = chromadb.PersistentClient(path=str(store_dir), settings=client_settings())
    return Chroma(
        collection_name=collection,
        embedding_function=embeddings,
        client=client,
        collection_metadata={"hnsw:space": space},
    )


class Store:
    def __init__(self, vectorstore, embeddings: Embeddings, manifest: dict):
        self.vectorstore = vectorstore
        self.embeddings = embeddings
        self.manifest = manifest
        self.space = manifest.get("space", "cosine")

    @property
    def settings(self) -> Settings:
        return Settings(**self.manifest["settings"])

    def add(self, chunks: list[Chunk]) -> None:
        self.vectorstore.add_texts(
            [chunk.text for chunk in chunks],
            metadatas=[chunk.metadata() for chunk in chunks],
            ids=[chunk.id for chunk in chunks],
        )

    def count(self) -> int:
        return len(self.vectorstore.get(include=[])["ids"])

    def chunks(self) -> list[Chunk]:
        data = self.vectorstore.get(include=["documents", "metadatas"])
        found = [
            Chunk(id=cid, text=text, **meta)
            for cid, text, meta in zip(data["ids"], data["documents"], data["metadatas"])
        ]
        return sorted(found, key=lambda chunk: (chunk.source, chunk.start))

    def search_vector(self, vector: list[float], k: int) -> list[Hit]:
        """The ``k`` closest chunks to an embedded query, best first."""
        results = self.vectorstore.similarity_search_by_vector_with_relevance_scores(vector, k=k)
        hits = [
            Hit(
                id=doc.id,
                text=doc.page_content,
                score=distance_to_score(dist, self.space),
                **doc.metadata,
            )
            for doc, dist in results
        ]
        return sorted(hits, key=lambda hit: -hit.score)

    def search(self, query: str, k: int) -> list[Hit]:
        return self.search_vector(self.embeddings.embed_query(query), k)


def make_manifest(settings: Settings, books: list[Book], chunks: list[Chunk], space: str) -> dict:
    per_book = {book.path: 0 for book in books}
    for chunk in chunks:
        per_book[chunk.source] += 1
    return {
        "format": FORMAT,
        "collection": COLLECTION,
        "space": space,
        "settings": settings.to_dict(),
        "embedding_model_id": settings.model_id,
        "query_instruction": settings.instruction,
        "chunks": len(chunks),
        "books": [
            {
                "path": book.path,
                "sha256": book.sha256,
                "characters": book.end - book.start,
                "chunks": per_book[book.path],
            }
            for book in books
        ],
    }


def _looks_like_store(path: Path) -> bool:
    return (path / MANIFEST).exists() or (path / "chroma.sqlite3").exists()


def build_store(
    chunks: list[Chunk],
    embeddings: Embeddings,
    store_dir: str | Path,
    settings: Settings,
    books: list[Book],
    space: str = "cosine",
) -> Store:
    """Replace whatever store is at ``store_dir`` with one holding ``chunks``."""
    store_dir = Path(store_dir)
    if store_dir.exists() and any(store_dir.iterdir()):
        if not _looks_like_store(store_dir):
            raise FileExistsError(f"{store_dir} is not empty and is not a Chroma store")
        _clear_chroma_cache()
        shutil.rmtree(store_dir)
    store_dir.mkdir(parents=True, exist_ok=True)
    manifest = make_manifest(settings, books, chunks, space)
    store = Store(new_vectorstore(embeddings, store_dir, space), embeddings, manifest)
    store.add(chunks)
    (store_dir / MANIFEST).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return store


def memory_store(
    chunks: list[Chunk],
    embeddings: Embeddings,
    settings: Settings,
    books: list[Book],
    space: str = "cosine",
    collection: str = COLLECTION,
) -> Store:
    """An in-memory store, for evaluation; give each live store its own ``collection`` name."""
    vectorstore = new_vectorstore(embeddings, None, space, collection)
    with contextlib.suppress(Exception):
        vectorstore.delete_collection()
    vectorstore = new_vectorstore(embeddings, None, space, collection)
    store = Store(vectorstore, embeddings, make_manifest(settings, books, chunks, space))
    store.add(chunks)
    return store


def read_manifest(store_dir: str | Path) -> dict:
    path = Path(store_dir) / MANIFEST
    if not path.exists():
        raise FileNotFoundError(f"no store at {store_dir}; run 'rag-local ingest' first")
    return json.loads(path.read_text(encoding="utf-8"))


def open_store(
    store_dir: str | Path, embeddings: Embeddings | None = None, device: str = "auto"
) -> Store:
    """Reopen a store, loading the embedding model and instruction it was built with."""
    manifest = read_manifest(store_dir)
    if embeddings is None:
        embeddings = emb.make_embeddings(
            manifest["embedding_model_id"], manifest["query_instruction"], device
        )
    vectorstore = new_vectorstore(embeddings, store_dir, manifest["space"], manifest["collection"])
    return Store(vectorstore, embeddings, manifest)
