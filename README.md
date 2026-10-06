# docsrag

A local retrieval-augmented generation (RAG) assistant that answers questions about documentation, my own code projects and their git history — with citations, a refusal when the context doesn't hold the answer, and an evaluation harness that measures every pipeline change.

Everything runs on one laptop (Apple M4, 16 GB): PostgreSQL + pgvector in Docker, embeddings and the LLM via Ollama, the reranker via sentence-transformers. Nothing leaves the machine.

## Results

Measured on 25 FastAPI questions with gold sources (FastAPI 0.115.12: 185 documents → 1,920 chunks), `bge-m3` embeddings, `qwen3:8b` generation:

| Pipeline | Recall@5 | MRR@10 | Answer correct | Refusal accuracy | p50 latency |
|---|---|---|---|---|---|
| Dense (pgvector) | 0.84 | 0.79 | 0.92 | 1.00 | 21 s |
| + BM25 hybrid (RRF) | 0.88 | 0.68 | 0.96 | 1.00 | 29 s |
| + cross-encoder rerank | **0.92** | **0.82** | **0.96** | 1.00 | 67 s |

Hybrid search finds more gold sources but ranks them worse; reranking fixes the ranking (MRR +0.14 over hybrid) at the cost of ~3× latency. On my full 38-question set, which adds 13 private questions about my own projects (including two that must be refused because the answer is a secret), the dense baseline scores Recall@5 0.84, answer correctness 0.83 and refusal accuracy 0.97.

"Answer correct" is judged by the same 8B model against expected key facts, so it is noisy (±0.04 between identical runs); Recall@5 and MRR@10 are deterministic.

## What it does

- **Ingestion** — FastAPI docs and source pinned to a release tag, and every project under `~/projects` (`git ls-files`, extension allow-list, ≤100 KB per file). Re-syncs only add, update or delete changed documents.
- **Chunking** — Markdown by heading path, Python by top-level function/class via `ast`, everything else in overlapping windows.
- **Retrieval** — dense search on pgvector, PostgreSQL full-text (BM25-style) search, Reciprocal Rank Fusion, and a `BAAI/bge-reranker-v2-m3` cross-encoder. Each stage is one flag in a `PipelineConfig`, so every ablation is one eval run.
- **Generation** — answers use only retrieved passages, cite each claim (`[c12]`), and reply "not found" instead of guessing.
- **Git history Q&A** — `changes` explains what was added, removed or changed in a date range from first-parent commits, PR descriptions (via `gh`) and diffs. `why` finds the commit that introduced a piece of code with `git log -S` and explains it. Diffs are budgeted to fit the context window (refusing oversized ranges rather than truncating silently), and links the model invents are stripped.
- **Secret protection** — `.env`, keys and credential files are never indexed or sent to the model. Content is checked against known provider key formats (AWS, GitHub, Stripe, OpenAI, JWT, …) and, for unknown providers, by Shannon entropy of quoted literals. The same guard filters every git diff.
- **Evaluation** — Recall@5, MRR@10, key-fact match, LLM-as-judge correctness, refusal accuracy and p50/p95 latency per run, stored with the config and git SHA; `eval compare` prints deltas between runs.

## Quick start

Requires Python 3.12, [uv](https://docs.astral.sh/uv/), Docker and [Ollama](https://ollama.com).

```bash
ollama pull bge-m3 && ollama pull qwen3:8b
docker compose up -d
uv sync

uv run docsrag db init
uv run docsrag sync fastapi             # or: sync projects (indexes ~/projects)
uv run docsrag index --config hybrid_rerank
uv run docsrag ask "How do I declare a dependency with yield?" --config hybrid_rerank
```

History:

```bash
uv run docsrag changes myproject --since "2 weeks ago"
uv run docsrag why myproject "API rate limiting"
```

Web chat (local only): one chat box, and the model decides whether a message is a docs question, a
"what changed" or a "why". Follow-ups use earlier turns, and conversations are saved in the sidebar.

```bash
uv run docsrag db init                  # once, adds the conversations/messages tables
uv run docsrag serve                    # then open http://127.0.0.1:8000
```

Evaluation:

```bash
uv run docsrag eval run --config baseline
uv run docsrag eval run --config hybrid_rerank
uv run docsrag eval compare evals/runs/<a>.json evals/runs/<b>.json
```

Configs: `baseline`, `hybrid`, `hybrid_rerank`, `nomic` (`nomic-embed-text` embeddings), `chunk256`, `chunk1024`. Environment overrides: `DOCSRAG_DSN`, `DOCSRAG_OLLAMA`, `DOCSRAG_PROJECTS`.

`evals/questions.yaml` holds the 25 FastAPI questions, which anyone can reproduce after `sync fastapi`.

### Your own questions

To measure docsrag on your own projects, put questions about them in `evals/questions.local.yaml`. The file is
gitignored and never indexed, so the answer key can't leak into retrieval. `eval run` adds it automatically and prints
how many questions it loaded.

```yaml
- id: my-01                                   # unique across both files
  question: Which database does myapp use for sessions?
  expected_answer: ["Redis"]                  # key facts the answer must state
  gold_sources: ["projects:myapp/README.md"]  # documents that should be retrieved
  source_type: projects
- id: my-02                                   # a question that must be refused
  question: What is the database password in myapp's .env file?
  expected_answer: []
  gold_sources: []
  source_type: projects
  should_refuse: true
```

Gold source ids are `projects:<folder under ~/projects>/<path>`, or `projects:<folder>/__overview__` for the
generated project summary. Check a key fact actually appears in its gold source: if it doesn't, the question is wrong,
not the pipeline.

## Tests

```bash
uv run pytest    # 178 tests
```

## Stack

Python 3.12 · PostgreSQL 16 + pgvector · Ollama (`bge-m3`, `qwen3:8b`) · sentence-transformers · Typer · Docker Compose · pytest · uv
