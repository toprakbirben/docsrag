from dataclasses import replace
from functools import cache

from stuffrag.retrieve import Hit

MODEL = "BAAI/bge-reranker-v2-m3"


@cache
def _model():
    import torch
    from sentence_transformers import CrossEncoder

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    return CrossEncoder(MODEL, device=device, max_length=1024)


def rerank(query: str, hits: list[Hit]) -> list[Hit]:
    if not hits:
        return []
    scores = _model().predict([(query, h.text) for h in hits])
    return sorted((replace(h, score=float(s)) for h, s in zip(hits, scores, strict=True)),
                  key=lambda h: -h.score)
