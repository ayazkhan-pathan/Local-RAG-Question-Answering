"""Embedding models: device choice, the optional BGE query instruction and cosine similarity."""

from __future__ import annotations

import math
from collections.abc import Sequence

from langchain_core.embeddings import Embeddings

from .config import EMBEDDING_MODELS


def pick_device(preferred: str = "auto") -> str:
    """'cuda' if available, then Apple's 'mps', then 'cpu'; any other value is passed through."""
    if preferred != "auto":
        return preferred
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class InstructedEmbeddings(Embeddings):
    """Wrap an embedding model so that queries, but not passages, carry an instruction prefix."""

    def __init__(self, base: Embeddings, instruction: str | None = None):
        self.base = base
        self.instruction = instruction or ""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.base.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        return self.base.embed_query(self.instruction + text)

    def embed_queries(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed many queries in one batch."""
        return self.base.embed_documents([self.instruction + text for text in texts])


def load_embeddings(model_id: str, device: str = "auto", batch_size: int = 32) -> Embeddings:
    """A sentence-transformers model through LangChain, returning unit-length vectors."""
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name=model_id,
        model_kwargs={"device": pick_device(device)},
        encode_kwargs={"normalize_embeddings": True, "batch_size": batch_size},
    )


def make_embeddings(
    model: str, instruction: str | None = None, device: str = "auto"
) -> InstructedEmbeddings:
    """Load ``model`` (a short name such as 'bge-small' or a Hugging Face id)."""
    model_id = EMBEDDING_MODELS.get(model, model)
    return InstructedEmbeddings(load_embeddings(model_id, device), instruction)


def embed_queries(embeddings: Embeddings, texts: Sequence[str]) -> list[list[float]]:
    if isinstance(embeddings, InstructedEmbeddings):
        return embeddings.embed_queries(texts)
    return [embeddings.embed_query(text) for text in texts]


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0
