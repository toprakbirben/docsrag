from dataclasses import dataclass, replace

import psycopg

from stuffrag.config import PipelineConfig
from stuffrag.embed import embed, table, to_pgvector


@dataclass(frozen=True)
class Hit:
    chunk_id: int
    document_id: str
    text: str
    score: float


def dense(conn: psycopg.Connection, query: str, cfg: PipelineConfig, k: int, prefix: str | None = None) -> list[Hit]:
    qv = to_pgvector(embed([query], cfg.embedder, kind="query")[0])
    rows = conn.execute(
        f"""SELECT c.id, c.document_id, c.text, 1 - (e.embedding <=> %(q)s::vector)
            FROM {table(cfg.embedder)} e JOIN chunks c ON c.id = e.chunk_id
            WHERE c.chunker = %(chunker)s AND (%(prefix)s::text IS NULL OR c.document_id LIKE %(prefix)s)
            ORDER BY e.embedding <=> %(q)s::vector LIMIT %(k)s""",
        {"q": qv, "chunker": cfg.chunker, "k": k, "prefix": prefix},
    ).fetchall()
    return [Hit(*r) for r in rows]


def fulltext(conn: psycopg.Connection, query: str, cfg: PipelineConfig, k: int, prefix: str | None = None) -> list[Hit]:
    # OR the query lexemes: plainto_tsquery ANDs them, which matches almost nothing for full questions.
    rows = conn.execute(
        """WITH q AS (SELECT replace(plainto_tsquery('english', %(q)s)::text, '&', '|')::tsquery AS q)
           SELECT c.id, c.document_id, c.text, ts_rank_cd(c.tsv, q.q)
           FROM chunks c, q WHERE c.chunker = %(chunker)s AND c.tsv @@ q.q
             AND (%(prefix)s::text IS NULL OR c.document_id LIKE %(prefix)s)
           ORDER BY 4 DESC LIMIT %(k)s""",
        {"q": query, "chunker": cfg.chunker, "k": k, "prefix": prefix},
    ).fetchall()
    return [Hit(*r) for r in rows]


def rrf(rankings: list[list[Hit]], k: int = 60) -> list[Hit]:
    scores: dict[int, float] = {}
    first: dict[int, Hit] = {}
    for ranking in rankings:
        for rank, hit in enumerate(ranking, 1):
            scores[hit.chunk_id] = scores.get(hit.chunk_id, 0.0) + 1 / (k + rank)
            first.setdefault(hit.chunk_id, hit)
    order = sorted(scores, key=lambda cid: (-scores[cid], cid))
    return [replace(first[cid], score=scores[cid]) for cid in order]


def rerank(query: str, hits: list[Hit]) -> list[Hit]:
    from stuffrag.rerank import rerank as _rerank  # lazy: loads torch only when rerank is on

    return _rerank(query, hits)


def retrieve(conn: psycopg.Connection, query: str, cfg: PipelineConfig, project: str | None = None) -> list[Hit]:
    prefix = f"projects:{project}/%" if project else None
    pool = cfg.candidate_k if (cfg.hybrid or cfg.rerank) else cfg.top_k
    hits = dense(conn, query, cfg, pool, prefix)
    if cfg.hybrid:
        hits = rrf([hits, fulltext(conn, query, cfg, pool, prefix)])
    if cfg.rerank:
        hits = rerank(query, hits)
    return hits[: cfg.top_k]
