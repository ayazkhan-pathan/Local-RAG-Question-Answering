"""Pure helpers in scripts/, loaded from their files."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from rag_local.documents import Chunk

from .conftest import REPO


def load(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


retrieval = load("evaluate_retrieval")
answers = load("evaluate_answers")


def test_cached_vectors_reuse_and_invalidate(tmp_path):
    calls = []

    def encode(texts):
        calls.append(list(texts))
        return [[float(len(t)), 1.0] for t in texts]

    path = tmp_path / "v.npz"
    first, _ = retrieval.cached_vectors(path, ["a", "bb"], encode)
    again, _ = retrieval.cached_vectors(path, ["a", "bb"], encode)
    assert np.array_equal(first, again) and len(calls) == 1
    retrieval.cached_vectors(path, ["a", "ccc"], encode)
    assert len(calls) == 2


def test_nearest_same_chapter_skips_overlapping_chunks():
    chunks = [
        Chunk("a", "", "b.md", 0, 100, 1),
        Chunk("b", "", "b.md", 50, 150, 1),
        Chunk("c", "", "b.md", 200, 300, 1),
        Chunk("d", "", "b.md", 400, 500, 2),
    ]
    vectors = np.array([[1.0, 0.0], [1.0, 0.0], [0.6, 0.8], [0.8, 0.6]])
    same, chance = retrieval.nearest_same_chapter(chunks, vectors)
    # a and b overlap, so a, b and c all pick d (chapter 2), and d picks c (chapter 1).
    assert same == pytest.approx(0.0)
    assert 0 < chance < 1


def test_pca_2d_finds_the_main_axis():
    points = np.array([[x, 0.1 * (x % 2)] for x in range(10)], dtype=float)
    _, axes, share = retrieval.pca_2d(points)
    assert abs(axes[0][0]) == pytest.approx(1.0, abs=1e-3)
    assert share[0] > 0.99


def test_model_label_never_writes_a_local_path(tmp_path):
    cache = tmp_path / "hub" / "models--Org--Tiny-Model" / "snapshots" / "abc123"
    cache.mkdir(parents=True)
    assert answers.model_label(str(cache)) == "Org/Tiny-Model"
    assert answers.model_label(str(tmp_path)) == Path(tmp_path).name
    assert answers.model_label("Qwen/Qwen2.5-1.5B-Instruct") == "Qwen/Qwen2.5-1.5B-Instruct"
