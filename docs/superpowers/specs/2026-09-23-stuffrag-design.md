# Plan: `stuffrag` — local "chat with my stuff" RAG + evals (Phase 1)

## Context
Toprak wants a local, open-source, always-on assistant (OpenClaw-like) that knows about their daily life. Phase 1 is the RAG core, measured by an eval set, so later changes are proven to help. Sources in Phase 1: **Gmail** (catch things missed) and **FastAPI docs+code** (framework knowledge). Explicitly **deferred**: job-openings tool (LinkedIn alerts + ATS APIs), personal notes (A), tweets/bookmarks/articles snapshot (D), live X/HN/RSS ingestors, daily digest agent, Telegram bot.

Machine: Apple M4, 16 GB. Ollama, Docker, psql, uv, Python 3.12 present. New project — no existing code to reuse.

## Decisions (with reasons)
- **Framework corpus: FastAPI** (docs `docs/en/**/*.md` + `fastapi/` source, pinned to a release tag). Mid-sized, stable, Markdown — fast to index, unambiguous eval answers. Pinning the tag keeps evals reproducible.
- **Vector store: Postgres 16 + pgvector (Docker)**. One DB gives vectors, `tsvector` full-text for hybrid search, and SQL filters (email date/sender, source type). Chroma can't do the last two cleanly.
- **Embeddings:** `bge-m3` (default) and `nomic-embed-text` via Ollama — both compared in evals. **Reranker:** `BAAI/bge-reranker-v2-m3` via `sentence-transformers` CrossEncoder on MPS. **LLM:** `qwen3:8b` via Ollama.
- **Gmail:** Gmail API, `gmail.readonly` scope, OAuth desktop flow, incremental sync via `historyId`. Token stored locally in `~/.stuffrag/`; email never leaves the machine.
- **Projects (`stuff sync projects`, added 2026-10-05):** every own folder in `~/projects` (third-party clones and backups skipped). Git repos list files via `git ls-files`, others by walk with `node_modules`/`vendor`/venvs pruned; code/doc extension allow-list, <=100 KB. **Secrets are never indexed:** `.env*`, keys and credential files are rejected by name, and any file whose content matches a secret pattern is skipped whole (paths reported, never content). Each project also gets a deterministic `__overview__` doc (README, deps, languages, layout, git history) for big-picture questions; chunks are prefixed with `project/path`. `stuff ask --project X` restricts retrieval to one project.
- **Interface:** Typer CLI (`stuff sync|ask|eval`) + small local FastAPI web chat on `localhost`. Background sync via a `launchd` plist.

## Architecture
```
sources ──► ingest/ (gmail, fastapi_repo)    ──► documents table (raw + metadata)
                                                     │
                                  chunk/ (md-heading, code-AST, email-thread)
                                                     │
                                  embed/ (ollama) ──► chunks table (vector + tsvector)
query ─► retrieve/ (dense ∪ BM25 → RRF fuse) ─► rerank/ (cross-encoder top-k) ─► generate/ (qwen3, cited answer)
evals/ runs the same pipeline with a config and records metrics per run
```
Every pipeline stage is configured by one `PipelineConfig` (chunk size/overlap, embedder, hybrid on/off, rerank on/off, top-k). The eval harness takes a config, so each ablation is one flag change.

## Project layout — `~/projects/stuffrag/`
- `pyproject.toml` (uv), `docker-compose.yml` (pgvector/pgvector:pg16), `config.toml` (model names)
- `stuffrag/db.py` — schema + migrations (`documents`, `chunks`, `sync_state`, `eval_runs`)
- `stuffrag/ingest/{gmail.py, fastapi_repo.py}`
- `stuffrag/chunk.py` — Markdown by heading, Python by function/class (`ast`), email by thread with quoted-reply stripping
- `stuffrag/embed.py`, `retrieve.py`, `rerank.py`, `generate.py` (answers cite chunk ids; refuse with "not found" when context is weak)
- `stuffrag/cli.py`, `stuffrag/web.py` (one chat page)
- `evals/questions.yaml`, `evals/run.py`, `evals/runs/*.json`
- `tests/` — unit tests for chunkers, RRF fusion
- `docs/superpowers/specs/2026-09-23-stuffrag-design.md` — this plan copied in as the spec

## Evals (the core deliverable)
- **40 questions** in `questions.yaml`, each with `question`, `expected_answer` (key facts), `gold_sources` (doc/file ids), `source_type`:
  - 25 FastAPI — I draft from the pinned docs; each gold source is checked against the file.
  - 12 email — I generate *candidates* from sampled emails locally; **you verify/edit** (only you know the ground truth). 3 of them are "should answer not found" to test refusal.
  - 3 cross-source / date-filtered (e.g. "what did X email about last month").
- **Metrics per run:** retrieval Recall@5, MRR@10 (vs gold_sources); answer correctness (qwen3-as-judge against key facts, plus a key-fact substring check as a sanity floor); refusal accuracy; p50/p95 latency. Stored with config + git SHA; `stuff eval compare <runA> <runB>` prints the deltas.
- **Planned ablations:** dense-only → +BM25 hybrid → +reranker; bge-m3 vs nomic-embed; chunk 256/512/1024 tokens.
- Known limitation: using a local 8B model as the judge is noisy. I'll spot-check 10 judgments by hand in the first run and report how often it agrees with me.

## Build order (each step ends verified)
1. Scaffold, docker pgvector up, schema — `stuff db init` works.
2. FastAPI ingest + chunk + embed — row counts reported, chunker tests pass.
3. Dense retrieval + generate — `stuff ask` answers a FastAPI question with citations.
4. Eval harness + 25 FastAPI questions — **baseline run recorded.**
5. Hybrid + rerank — eval run proves/disproves the gain.
6. Gmail OAuth + incremental sync + email chunking — you run the OAuth once (`! stuff auth gmail`).
7. Email eval questions (you verify) — run evals again.
8. Web chat + launchd sync job.

## Verification
- `uv run pytest` passes (chunkers, fusion).
- `stuff eval run --config baseline` and `--config hybrid_rerank` both produce run files, and `stuff eval compare` shows the metric deltas.
- `stuff ask "How do I declare a dependency with yield in FastAPI?"` returns a cited answer that matches the docs.
- After `stuff sync gmail` twice, the second run only fetches new messages (logged count).
- `curl localhost:8000/` serves the chat page, and `launchctl list | grep stuffrag` shows the sync job.

## Open items
- The email eval answers need your verification (step 7).
