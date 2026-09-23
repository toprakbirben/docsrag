import os

import psycopg

DSN = os.environ.get("STUFFRAG_DSN", "postgresql://stuff:stuff@localhost:5433/stuffrag")

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
