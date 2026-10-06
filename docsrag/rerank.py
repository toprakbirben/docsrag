from dataclasses import replace
from functools import cache

from docsrag.retrieve import Hit

MODEL = "BAAI/bge-reranker-v2-m3"
# Pinned to the commit already cached under
# ~/.cache/huggingface/hub/models--BAAI--bge-reranker-v2-m3/snapshots/.
REVISION = "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"


@cache
def _model():
    import torch
    from sentence_transformers import CrossEncoder

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    try:
        return CrossEncoder(
            MODEL, device=device, max_length=1024, revision=REVISION, local_files_only=True
        )
    except OSError:
        # Snapshot not cached yet: the one allowed one-time download.
        return CrossEncoder(MODEL, device=device, max_length=1024, revision=REVISION)


def rerank(query: str, hits: list[Hit]) -> list[Hit]:
    if not hits:
        return []
    scores = _model().predict([(query, h.text) for h in hits])
    return sorted((replace(h, score=float(s)) for h, s in zip(hits, scores, strict=True)),
                  key=lambda h: -h.score)
