from __future__ import annotations

from types import SimpleNamespace

import pytest

from rag_local.embeddings import InstructedEmbeddings, pick_device
from rag_local.generators import HFGenerator, LocalGenerator, MissingTokenError, make_generator

from .conftest import FakeEmbeddings


class FakeClient:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        self.calls.append(kwargs)
        message = SimpleNamespace(content="  Dinah.  ")
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def test_hf_needs_a_token(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    with pytest.raises(MissingTokenError, match="HF_TOKEN"):
        HFGenerator(client_factory=FakeClient)


def test_hf_calls_llama_through_sambanova(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "test-token")
    generator = HFGenerator(client_factory=FakeClient)
    assert generator.client.kwargs == {"provider": "sambanova", "token": "test-token"}
    assert generator.generate("What is the cat called?") == "Dinah."
    call = generator.client.calls[0]
    assert call["model"] == "meta-llama/Llama-3.1-8B-Instruct"
    assert call["messages"] == [{"role": "user", "content": "What is the cat called?"}]


class FakeBatch(dict):
    def to(self, device):
        self["device"] = device
        return self


class FakeTokenizer:
    pad_token_id = None
    eos_token_id = 0

    def apply_chat_template(self, messages, **kwargs):
        self.messages = messages
        self.template_kwargs = kwargs
        return FakeBatch(input_ids=[[5, 6, 7]], attention_mask=[[1, 1, 1]])

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(f"t{i}" for i in ids) + "  "


class FakeModel:
    def generate(self, **kwargs):
        self.kwargs = kwargs
        return [[5, 6, 7, 8, 9]]


def test_local_generator_is_greedy_and_returns_new_tokens_only():
    loads = []
    tokenizer, model = FakeTokenizer(), FakeModel()

    def loader(model_id, device):
        loads.append((model_id, device))
        return tokenizer, model

    generator = LocalGenerator("some/model", device="cpu", loader=loader)
    assert loads == []
    assert generator.generate("Hello") == "t8 t9"
    assert generator.generate("Again") == "t8 t9"
    assert loads == [("some/model", "cpu")]
    assert tokenizer.messages == [{"role": "user", "content": "Again"}]
    assert tokenizer.template_kwargs["add_generation_prompt"] is True
    assert model.kwargs["do_sample"] is False
    assert model.kwargs["max_new_tokens"] == 128
    assert model.kwargs["pad_token_id"] == 0


def test_default_local_model():
    assert LocalGenerator(device="cpu").model_id == "Qwen/Qwen2.5-1.5B-Instruct"


def test_unknown_generator():
    with pytest.raises(ValueError):
        make_generator("other")


def test_pick_device(monkeypatch):
    import torch

    assert pick_device("cpu") == "cpu"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert pick_device() == "mps"
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    assert pick_device() == "cpu"


def test_query_instruction_applies_to_queries_only():
    base = FakeEmbeddings()
    wrapped = InstructedEmbeddings(base, "Represent this: ")
    assert wrapped.embed_documents(["cat"]) == base.embed_documents(["cat"])
    assert wrapped.embed_query("cat") == base.embed_query("Represent this: cat")
    assert wrapped.embed_queries(["cat"]) == [base.embed_query("Represent this: cat")]
