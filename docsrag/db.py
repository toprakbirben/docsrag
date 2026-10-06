import os

import psycopg
from psycopg.types.json import Jsonb

DSN = os.environ.get("DOCSRAG_DSN", "postgresql://docsrag:docsrag@localhost:5433/docsrag")

# Dimension is per-embedder, so vectors live in one table per embedder (created in embed.py).
SCHEMA = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    id          TEXT PRIMARY KEY,          -- e.g. 'fastapi:docs/en/docs/tutorial/dependencies.md'
    source_type TEXT NOT NULL,             -- 'fastapi' | 'email' | 'job'
    title       TEXT,
    body        TEXT NOT NULL,
    metadata    JSONB NOT NULL DEFAULT '{}',
    created_at  TIMESTAMPTZ,
    ingested_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS chunks (
    id          BIGSERIAL PRIMARY KEY,
    document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunker     TEXT NOT NULL,             -- chunker name + params, so ablations coexist
    ordinal     INT  NOT NULL,
    text        TEXT NOT NULL,
    tsv         TSVECTOR GENERATED ALWAYS AS (to_tsvector('english', text)) STORED,
    UNIQUE (document_id, chunker, ordinal)
);
CREATE INDEX IF NOT EXISTS chunks_tsv_idx ON chunks USING GIN (tsv);

CREATE TABLE IF NOT EXISTS jobs (
    id          TEXT PRIMARY KEY,          -- hash(company, title, location)
    company     TEXT NOT NULL,
    title       TEXT NOT NULL,
    location    TEXT,
    url         TEXT,
    source      TEXT NOT NULL,             -- 'linkedin_alert' | 'greenhouse' | 'lever' | 'ashby'
    score       REAL,
    first_seen  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS sync_state (
    source TEXT PRIMARY KEY,
    cursor TEXT,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS eval_runs (
    id         TEXT PRIMARY KEY,
    config     JSONB NOT NULL,
    git_sha    TEXT,
    metrics    JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
"""


def connect() -> psycopg.Connection:
    return psycopg.connect(DSN)


def init() -> list[str]:
    with connect() as conn:
        conn.execute(SCHEMA)
        rows = conn.execute(
            "SELECT tablename FROM pg_tables WHERE schemaname = 'public' ORDER BY tablename"
        ).fetchall()
    return [r[0] for r in rows]


def diff_documents(existing: dict[str, str], docs: list[dict]) -> tuple[list[str], list[str]]:
    new = [d["id"] for d in docs if d["id"] not in existing]
    changed = [d["id"] for d in docs if d["id"] in existing and existing[d["id"]] != d["body"]]
    return new, changed


def stale_ids(existing: list[str], docs: list[dict]) -> list[str]:
    keep = {d["id"] for d in docs}
    return [i for i in existing if i not in keep]


def delete_missing(conn: psycopg.Connection, source_type: str, docs: list[dict]) -> int:
    """Drop this source's docs that no longer exist (chunks + vectors cascade). Returns count."""
    existing = [r[0] for r in conn.execute("SELECT id FROM documents WHERE source_type = %s", (source_type,))]
    gone = stale_ids(existing, docs)
    conn.execute("DELETE FROM documents WHERE id = ANY(%s)", (gone,))
    conn.commit()
    return len(gone)


def upsert_documents(conn: psycopg.Connection, docs: list[dict]) -> tuple[int, int]:
    """Insert new docs, replace changed ones and drop their stale chunks. Returns (new, changed)."""
    ids = [d["id"] for d in docs]
    existing = dict(conn.execute("SELECT id, body FROM documents WHERE id = ANY(%s)", (ids,)).fetchall())
    new, changed = diff_documents(existing, docs)
    todo = set(new) | set(changed)
    conn.execute("DELETE FROM chunks WHERE document_id = ANY(%s)", (changed,))
    with conn.cursor() as cur:
        cur.executemany(
            """INSERT INTO documents (id, source_type, title, body, metadata)
               VALUES (%(id)s, %(source_type)s, %(title)s, %(body)s, %(metadata)s)
               ON CONFLICT (id) DO UPDATE SET title = EXCLUDED.title, body = EXCLUDED.body,
                   metadata = EXCLUDED.metadata, ingested_at = now()""",
            [{**d, "metadata": Jsonb(d["metadata"])} for d in docs if d["id"] in todo],
        )
    conn.commit()
    return len(new), len(changed)
