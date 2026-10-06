import os
from dataclasses import asdict, dataclass

OLLAMA_URL = os.environ.get("DOCSRAG_OLLAMA", "http://localhost:11434")

# Embedding dimension per Ollama model; one vector table per embedder.
EMBEDDERS = {"bge-m3": 1024, "nomic-embed-text": 768}


@dataclass(frozen=True)
class PipelineConfig:
    name: str = "baseline"
    embedder: str = "bge-m3"
    chunk_tokens: int = 512
    chunk_overlap: int = 64
    hybrid: bool = False
    rerank: bool = False
    top_k: int = 5
    candidate_k: int = 30  # per-retriever pool before fusion / rerank
    llm: str = "qwen3:8b"

    @property
    def chunker(self) -> str:
        return f"w{self.chunk_tokens}o{self.chunk_overlap}"

    def to_dict(self) -> dict:
        return asdict(self)


PRESETS = {
    "baseline": PipelineConfig(),
    "hybrid": PipelineConfig(name="hybrid", hybrid=True),
    "hybrid_rerank": PipelineConfig(name="hybrid_rerank", hybrid=True, rerank=True),
    "nomic": PipelineConfig(name="nomic", embedder="nomic-embed-text"),
    "chunk256": PipelineConfig(name="chunk256", chunk_tokens=256, chunk_overlap=32),
    "chunk1024": PipelineConfig(name="chunk1024", chunk_tokens=1024, chunk_overlap=128),
}


def get(name: str) -> PipelineConfig:
    try:
        return PRESETS[name]
    except KeyError:
        raise ValueError(f"unknown config {name!r}; choose from {', '.join(PRESETS)}") from None
