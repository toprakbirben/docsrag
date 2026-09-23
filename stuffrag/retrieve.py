from dataclasses import dataclass

import psycopg

from stuffrag.config import PipelineConfig
from stuffrag.embed import embed, table, to_pgvector


@dataclass(frozen=True)
class Hit:
    chunk_id: int
    document_id: str
    text: str
    score: float


def dense(conn: psycopg.Connection, query: str, cfg: PipelineConfig, k: int) -> list[Hit]:
    qv = to_pgvector(embed([query], cfg.embedder, kind="query")[0])
    rows = conn.execute(
        f"""SELECT c.id, c.document_id, c.text, 1 - (e.embedding <=> %(q)s::vector)
            FROM {table(cfg.embedder)} e JOIN chunks c ON c.id = e.chunk_id
            WHERE c.chunker = %(chunker)s
            ORDER BY e.embedding <=> %(q)s::vector LIMIT %(k)s""",
        {"q": qv, "chunker": cfg.chunker, "k": k},
    ).fetchall()
    return [Hit(*r) for r in rows]


def retrieve(conn: psycopg.Connection, query: str, cfg: PipelineConfig) -> list[Hit]:
    return dense(conn, query, cfg, cfg.top_k)
