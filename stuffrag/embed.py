import re
from collections.abc import Callable

import httpx
import psycopg

from stuffrag.chunk import chunk
from stuffrag.config import EMBEDDERS, OLLAMA_URL, PipelineConfig

# (document prefix, query prefix) for models trained with task prefixes.
PREFIX = {"nomic-embed-text": ("search_document: ", "search_query: ")}
# Model context in tokens. Ollama caps one input at num_batch (default 2048), so raise it to the
# full context or dense code chunks fail with "input length exceeds the context length".
CONTEXT = {"bge-m3": 8192, "nomic-embed-text": 2048}


def table(embedder: str) -> str:
    return "emb_" + re.sub(r"\W", "_", embedder)


def to_pgvector(v: list[float]) -> str:
    return "[" + ",".join(map(str, v)) + "]"


def embed(texts: list[str], embedder: str, kind: str = "document") -> list[list[float]]:
    prefix = PREFIX.get(embedder, ("", ""))[0 if kind == "document" else 1]
    r = httpx.post(
        f"{OLLAMA_URL}/api/embed",
        json={"model": embedder, "input": [prefix + t for t in texts],
              "options": {"num_batch": CONTEXT[embedder]}},
        timeout=600,
    )
    r.raise_for_status()
    return r.json()["embeddings"]


def ensure_table(conn: psycopg.Connection, embedder: str) -> str:
    t = table(embedder)
    conn.execute(
        f"""CREATE TABLE IF NOT EXISTS {t} (
              chunk_id  BIGINT PRIMARY KEY REFERENCES chunks(id) ON DELETE CASCADE,
              embedding vector({EMBEDDERS[embedder]}) NOT NULL)"""
    )
    return t


def chunk_rows(docs: list[tuple], cfg: PipelineConfig) -> list[tuple[str, str, int, str]]:
    """(doc_id, path, label, body) -> chunk rows; a label (project/path) prefixes every chunk."""
    return [
        (doc_id, cfg.chunker, i, f"{label}\n\n{text}" if label else text)
        for doc_id, path, label, body in docs
        for i, text in enumerate(chunk(path or doc_id, body, cfg.chunk_tokens, cfg.chunk_overlap))
    ]


def index(
    conn: psycopg.Connection,
    cfg: PipelineConfig,
    on_progress: Callable[[int, int], None] | None = None,
    batch: int = 32,
) -> dict:
    """Chunk documents lacking chunks for cfg.chunker, then embed chunks lacking vectors. Resumable."""
    docs = conn.execute(
        """SELECT id, metadata->>'path', metadata->>'label', body FROM documents d
           WHERE NOT EXISTS (SELECT 1 FROM chunks c WHERE c.document_id = d.id AND c.chunker = %s)""",
        (cfg.chunker,),
    ).fetchall()
    rows = chunk_rows(docs, cfg)
    with conn.cursor() as cur:
        cur.executemany(
            "INSERT INTO chunks (document_id, chunker, ordinal, text) VALUES (%s, %s, %s, %s)", rows
        )
    conn.commit()

    t = ensure_table(conn, cfg.embedder)
    todo = conn.execute(
        f"""SELECT c.id, c.text FROM chunks c LEFT JOIN {t} e ON e.chunk_id = c.id
            WHERE c.chunker = %s AND e.chunk_id IS NULL ORDER BY c.id""",
        (cfg.chunker,),
    ).fetchall()
    for start in range(0, len(todo), batch):
        part = todo[start : start + batch]
        vecs = embed([text for _, text in part], cfg.embedder)
        with conn.cursor() as cur:
            cur.executemany(
                f"INSERT INTO {t} (chunk_id, embedding) VALUES (%s, %s::vector)",
                [(cid, to_pgvector(v)) for (cid, _), v in zip(part, vecs, strict=True)],
            )
        conn.commit()  # per batch, so an interrupted run resumes where it stopped
        if on_progress:
            on_progress(start + len(part), len(todo))
    return {"chunked_docs": len({doc_id for doc_id, *_ in rows}), "chunks": len(rows), "embedded": len(todo)}
