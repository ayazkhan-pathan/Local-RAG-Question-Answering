from __future__ import annotations

import json

import pytest

from rag_local.cli import main

from .conftest import FakeGenerator


@pytest.fixture
def store_dir(tmp_path, books_dir, patch_embeddings):
    store = tmp_path / "chroma"
    args = [
        "ingest",
        "--data",
        str(books_dir),
        "--store",
        str(store),
        "--chunk-size",
        "200",
        "--chunk-overlap",
        "40",
        "--no-instruction",
    ]
    assert main(args) == 0
    return store


def test_ingest_writes_a_manifest(store_dir, capsys):
    manifest = json.loads((store_dir / "manifest.json").read_text())
    assert manifest["settings"]["chunk_size"] == 200
    assert manifest["settings"]["embedding_model"] == "bge-small"
    assert manifest["query_instruction"] is None


def test_retrieve_only(store_dir, capsys):
    code = main(
        ["query", "Who was painting the roses red?", "--store", str(store_dir), "--retrieve-only"]
    )
    out = capsys.readouterr().out
    assert code == 0
    assert out.startswith("1. similarity ")
    assert "chapter 1" in out
    assert "would be answered" in out


def test_query_with_a_generator(store_dir, capsys, monkeypatch):
    monkeypatch.setattr("rag_local.cli.make_generator", lambda *args, **kwargs: FakeGenerator())
    assert main(["query", "Who was painting the roses red?", "--store", str(store_dir)]) == 0
    out = capsys.readouterr().out
    assert out.startswith("Answer: They painted the roses red.")
    assert "Sources:" in out


def test_query_refusal(store_dir, capsys, monkeypatch):
    monkeypatch.setattr("rag_local.cli.make_generator", lambda *args, **kwargs: FakeGenerator())
    main(
        [
            "query",
            "Who was painting the roses red?",
            "--store",
            str(store_dir),
            "--threshold",
            "0.999",
        ]
    )
    assert "No passage in the indexed books is close enough" in capsys.readouterr().out


def test_hf_without_token(store_dir, capsys, monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    code = main(["query", "Who?", "--store", str(store_dir), "--generator", "hf"])
    assert code == 2
    assert "HF_TOKEN" in capsys.readouterr().err


def test_compare(patch_embeddings, capsys):
    assert main(["compare", "apple", "apple"]) == 0
    out = capsys.readouterr().out
    assert "BAAI/bge-small-en-v1.5, 256 dimensions" in out
    assert "Cosine similarity of 'apple' and 'apple': 1.0000" in out


def test_evaluate(store_dir, books_dir, tmp_path, capsys):
    questions = tmp_path / "questions.jsonl"
    rows = [
        {
            "id": "q1",
            "chapter": 1,
            "question": "Who was painting the white roses red?",
            "answer": "the gardeners",
            "aliases": [],
            "quote": "planted a white rose-tree by mistake, and the Queen would be angry. Alice",
        },
        {"id": "x1", "off_topic": True, "question": "What is the capital of France?"},
    ]
    questions.write_text("".join(json.dumps(row) + "\n" for row in rows))
    out_file = tmp_path / "eval.json"
    # The sample book repeats its paragraphs, so that quote is not unique.
    with pytest.raises(ValueError, match="3 times"):
        main(
            [
                "evaluate",
                "--store",
                str(store_dir),
                "--data",
                str(books_dir),
                "--questions",
                str(questions),
            ]
        )
    rows[0]["quote"] = "Contents"
    questions.write_text("".join(json.dumps(row) + "\n" for row in rows))
    assert (
        main(
            [
                "evaluate",
                "--store",
                str(store_dir),
                "--data",
                str(books_dir),
                "--questions",
                str(questions),
                "--out",
                str(out_file),
            ]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert "On-topic questions: 1, off-topic: 1" in out
    assert "hit@1" in out
    payload = json.loads(out_file.read_text())
    assert payload["questions"][0]["gold"]


@pytest.mark.parametrize("command", ["ingest", "query", "compare", "evaluate"])
def test_help(command, capsys):
    with pytest.raises(SystemExit) as exit_info:
        main([command, "--help"])
    assert exit_info.value.code == 0
