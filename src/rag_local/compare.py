"""Cosine similarity of two texts under an embedding model."""

from __future__ import annotations

from langchain_core.embeddings import Embeddings

from .embeddings import cosine_similarity


def compare(text_a: str, text_b: str, embeddings: Embeddings) -> tuple[float, int]:
    """Return the cosine similarity of the two texts and the embedding dimension."""
    vec_a = embeddings.embed_query(text_a)
    vec_b = embeddings.embed_query(text_b)
    return cosine_similarity(vec_a, vec_b), len(vec_a)
