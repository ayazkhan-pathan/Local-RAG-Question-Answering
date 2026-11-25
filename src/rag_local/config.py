"""Model names, retrieval presets and default paths."""

from __future__ import annotations

from dataclasses import asdict, dataclass

EMBEDDING_MODELS = {
    "bge-small": "BAAI/bge-small-en-v1.5",
    "bge-base": "BAAI/bge-base-en-v1.5",
    "bge-large": "BAAI/bge-large-en-v1.5",
}
# The BGE v1.5 model cards recommend this prefix for short queries that retrieve passages.
BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

LOCAL_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
HF_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
HF_PROVIDER = "sambanova"

DATA_DIR = "data/books"
STORE_DIR = "chroma"
QUESTIONS = "eval/questions.jsonl"


@dataclass(frozen=True)
class Settings:
    """Everything that decides which chunks a question retrieves."""

    embedding_model: str = "bge-small"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    query_instruction: bool = True
    strip_boilerplate: bool = True
    k: int = 3
    threshold: float = 0.5

    @property
    def model_id(self) -> str:
        """The Hugging Face id of the embedding model."""
        return EMBEDDING_MODELS.get(self.embedding_model, self.embedding_model)

    @property
    def instruction(self) -> str | None:
        return BGE_QUERY_INSTRUCTION if self.query_instruction else None

    def label(self) -> str:
        """A short human-readable name, such as 'bge-small 1000/200 +instruction'."""
        text = f"{self.embedding_model} {self.chunk_size}/{self.chunk_overlap}"
        text += " +instruction" if self.query_instruction else ""
        return text + ("" if self.strip_boilerplate else " +boilerplate")

    def key(self) -> str:
        """A file-name-safe id, such as 'bge-small_1000-200_instr'."""
        name = self.embedding_model.replace("/", "--")
        text = f"{name}_{self.chunk_size}-{self.chunk_overlap}_"
        text += "instr" if self.query_instruction else "plain"
        return text + ("" if self.strip_boilerplate else "_raw")

    def to_dict(self) -> dict:
        return asdict(self)


PRESETS = {
    # The settings of the first version of this repository.
    "original": Settings("bge-large", 1000, 500, query_instruction=False),
    # A 130 MB embedding model; the best bge-small row of the retrieval grid.
    "small": Settings("bge-small", 1000, 200, query_instruction=False),
}
DEFAULT_PRESET = "small"


def preset(name: str) -> Settings:
    try:
        return PRESETS[name]
    except KeyError:
        raise ValueError(f"unknown preset {name!r}; choose from {', '.join(PRESETS)}") from None
