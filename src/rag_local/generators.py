"""Text generators: a local Hugging Face model (the default) or Llama 3.1 8B through the HF API."""

from __future__ import annotations

import os
from typing import Callable, Protocol

from .config import HF_MODEL, HF_PROVIDER, LOCAL_MODEL
from .embeddings import pick_device


class Generator(Protocol):
    name: str

    def generate(self, prompt: str) -> str: ...


class MissingTokenError(RuntimeError):
    pass


def _load_transformers(model_id: str, device: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = torch.float16 if device in ("cuda", "mps") else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype).to(device).eval()
    return tokenizer, model


class LocalGenerator:
    """A chat model run with transformers: half precision on a GPU, greedy decoding.

    The model is loaded on the first call, so building a generator is cheap.
    """

    name = "local"

    def __init__(
        self,
        model: str = LOCAL_MODEL,
        device: str = "auto",
        max_new_tokens: int = 128,
        loader: Callable | None = None,
    ):
        self.model_id = model
        self.device = pick_device(device)
        self.max_new_tokens = max_new_tokens
        self._loader = loader or _load_transformers
        self._tokenizer = None
        self._model = None

    def generate(self, prompt: str) -> str:
        if self._model is None:
            self._tokenizer, self._model = self._loader(self.model_id, self.device)
        tokenizer, model = self._tokenizer, self._model
        inputs = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        ).to(self.device)
        pad = tokenizer.pad_token_id
        output = model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
            pad_token_id=tokenizer.eos_token_id if pad is None else pad,
        )
        prompt_length = len(inputs["input_ids"][0])
        return tokenizer.decode(output[0][prompt_length:], skip_special_tokens=True).strip()


class HFGenerator:
    """Llama 3.1 8B Instruct through the Hugging Face Inference API; needs HF_TOKEN."""

    name = "hf"

    def __init__(
        self,
        model: str = HF_MODEL,
        provider: str = HF_PROVIDER,
        token: str | None = None,
        max_tokens: int = 512,
        client_factory: Callable | None = None,
    ):
        token = token or os.environ.get("HF_TOKEN")
        if not token:
            raise MissingTokenError(
                "The hf generator calls the Hugging Face Inference API and needs an access "
                "token in the HF_TOKEN environment variable. Create one at "
                "https://huggingface.co/settings/tokens, or use the default local generator, "
                "which needs no token."
            )
        if client_factory is None:
            from huggingface_hub import InferenceClient

            client_factory = InferenceClient
        self.model_id = model
        self.max_tokens = max_tokens
        self.client = client_factory(provider=provider, token=token)

    def generate(self, prompt: str) -> str:
        completion = self.client.chat.completions.create(
            model=self.model_id,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self.max_tokens,
        )
        return (completion.choices[0].message.content or "").strip()


def make_generator(
    name: str, model: str | None = None, device: str = "auto", max_new_tokens: int = 128
) -> Generator:
    if name == "local":
        return LocalGenerator(model or LOCAL_MODEL, device, max_new_tokens)
    if name == "hf":
        return HFGenerator(model or HF_MODEL, max_tokens=max_new_tokens)
    raise ValueError(f"unknown generator {name!r}; choose 'local' or 'hf'")
