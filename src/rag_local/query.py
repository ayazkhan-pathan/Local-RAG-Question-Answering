"""Answer a question: retrieve the closest chunks, refuse if none is close enough, else prompt."""

from __future__ import annotations

from dataclasses import dataclass

from langchain_core.prompts import PromptTemplate

from .generators import Generator
from .store import Hit, Store

# The wording of the prompt in the first version of this repository.
PROMPT_TEMPLATE = """Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}"""
CONTEXT_SEPARATOR = "\n\n---\n\n"


def build_prompt(question: str, hits: list[Hit]) -> str:
    context = CONTEXT_SEPARATOR.join(hit.text for hit in hits)
    return PromptTemplate.from_template(PROMPT_TEMPLATE).format(context=context, question=question)


def refusal_message(best: float | None, threshold: float) -> str:
    if best is None:
        return "The store is empty, so there is nothing to answer from."
    return (
        "No passage in the indexed books is close enough to answer this: the best match has a "
        f"cosine similarity of {best:.2f}, below the threshold of {threshold:.2f}."
    )


@dataclass
class Answer:
    question: str
    hits: list[Hit]
    passed: bool
    text: str = ""
    prompt: str | None = None
    threshold: float = 0.5

    @property
    def best_score(self) -> float | None:
        return self.hits[0].score if self.hits else None


def answer_question(
    store: Store,
    question: str,
    generator: Generator | None = None,
    k: int = 3,
    threshold: float = 0.5,
) -> Answer:
    """Retrieve ``k`` chunks; below ``threshold`` refuse, otherwise prompt ``generator``.

    With no generator the answer holds the retrieved chunks and the prompt but no text.
    """
    hits = store.search(question, k)
    best = hits[0].score if hits else None
    if best is None or best < threshold:
        return Answer(question, hits, False, refusal_message(best, threshold), None, threshold)
    prompt = build_prompt(question, hits)
    text = generator.generate(prompt) if generator is not None else ""
    return Answer(question, hits, True, text, prompt, threshold)
