# stuffrag RAG Core (FastAPI corpus + evals) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ingest the pinned FastAPI docs + source, answer questions with cited chunks, and measure the pipeline with an eval harness: a recorded baseline run, then hybrid and rerank runs compared against it (spec build steps 2–5).

**Architecture:** One frozen `PipelineConfig` controls chunking, embedder, hybrid, rerank and top-k. `stuff sync fastapi` clones the pinned tag into `~/.stuffrag/fastapi/<tag>`, upserts `documents`, then `index()` chunks them into `chunks` (tagged with a chunker name so ablations coexist) and embeds into one table per embedder (`emb_bge_m3`, `emb_nomic_embed_text`). `ask()` = retrieve (dense, optionally ∪ full-text via RRF, optionally cross-encoder rerank) → qwen3 generation with `[c<id>]` citations. `stuff eval run` runs every question through the same `ask` path and writes a JSON run file + an `eval_runs` row.

**Tech Stack:** Python 3.12, uv, Typer, psycopg 3, Postgres 16 + pgvector (existing docker-compose, port 5433), Ollama HTTP API via httpx (`bge-m3`, `nomic-embed-text`, `qwen3:8b`), sentence-transformers CrossEncoder `BAAI/bge-reranker-v2-m3` on MPS, PyYAML, pytest.

**Spec:** `docs/superpowers/specs/2026-09-23-stuffrag-design.md`

**Out of scope (later plans):** Gmail (spec steps 6–7, incl. email chunker and the 15 email/cross-source questions), jobs (step 8, incl. `config.toml`), web chat + launchd (step 9).

## Global Constraints

- Python `>=3.12`, deps managed with `uv` (`uv add`, `uv run`). Existing DSN: `postgresql://stuff:stuff@localhost:5433/stuffrag` (env `STUFFRAG_DSN`).
- Embedders: `bge-m3` (default, 1024-dim) and `nomic-embed-text` (768-dim) via Ollama. Reranker: `BAAI/bge-reranker-v2-m3`. LLM: `qwen3:8b` via Ollama.
- FastAPI corpus is pinned to one release tag (`0.115.12`): docs `docs/en/docs/**/*.md` + `fastapi/**/*.py`. (The spec says `docs/en/**/*.md`, but the Markdown is actually under `docs/en/docs/`.)
- Answers cite chunk ids and refuse with "not found" when the context doesn't support an answer.
- Rule 5: the LLM is used only for generation and judging. Chunking, fusion, filtering and metrics are plain code.
- Eval metrics per run: Recall@5, MRR@10 (vs `gold_sources`), answer correctness (qwen3 judge + key-fact substring floor), refusal accuracy, p50/p95 latency; stored with config + git SHA.
- Nothing leaves the machine except `git clone` of FastAPI and the one-time HF download of the reranker.

## Deliberate deviations from the spec (flagged, not silent)

- Eval harness lives in `stuffrag/evals.py` (importable by the `stuff` CLI), not `evals/run.py`. `evals/questions.yaml` and `evals/runs/` stay where the spec puts them.
- "BM25" is Postgres full-text `ts_rank_cd` over the existing `tsv` column. It isn't true BM25, and run files label it `hybrid`.
- "Tokens" for chunk sizes = whitespace-separated words (≈0.75 of a model token). That's good enough to compare 256/512/1024 against each other.
- No ANN index. The corpus is a few thousand chunks, so an exact cosine scan takes milliseconds and avoids HNSW's post-filter-on-`chunker` recall loss.

## Review Focus

1. **FastAPI docs whose code lives in `{* ../../docs_src/x.py hl[..] *}` / `{!../../docs_src/x.py!}` includes.** "Dependency with yield" answers are mostly in those files. Includes must be expanded into the doc text, and unresolved ones reported, not silently dropped. → Task 2 tests.
2. **Retrieved context larger than Ollama's default context window (4096).** Ollama silently truncates the prompt, so the model never sees the later passages. Every chat call must set `num_ctx`. → Task 4 test.
3. **Model cites an id not in context, or writes `[c12, c14]`.** Only real context ids are kept, and both forms are parsed. → Task 4 test.
4. **Question with no lexical overlap / only stopwords** ("what is it?"). Full-text returns nothing, and fusion must still return the dense results rather than crashing or returning empty. → Task 7 test.
5. **Re-running `stuff sync fastapi`, or bumping the pinned tag.** No duplicate chunks; only changed documents get re-chunked and re-embedded. → Task 2 test (`diff_documents`) + Task 3 double-run verification.

---

### Task 1: Pipeline config + Markdown/Python chunkers

**Files:**
- Create: `stuffrag/config.py`, `stuffrag/chunk.py`
- Test: `tests/test_chunk.py`

**Interfaces:**
- Produces:
  - `config.PipelineConfig` (frozen dataclass: `name, embedder, chunk_tokens, chunk_overlap, hybrid, rerank, top_k, candidate_k, llm`; property `chunker -> str`; `to_dict() -> dict`)
  - `config.EMBEDDERS: dict[str, int]`, `config.PRESETS`, `config.get(name) -> PipelineConfig`, `config.OLLAMA_URL: str`
  - `chunk.window(text, max_tokens, overlap) -> list[str]`, `chunk.chunk_markdown(...)`, `chunk.chunk_python(...)`, `chunk.chunk(path, text, max_tokens, overlap) -> list[str]`

- [ ] **Step 1: Write the failing tests**

`tests/test_chunk.py`:
```python
import pytest

from stuffrag import config
from stuffrag.chunk import chunk, chunk_markdown, chunk_python, window


def body_tokens(c: str) -> int:
    # Markdown chunks are "<breadcrumb>\n\n<body>"; the size limit applies to the body.
    return len(c.split("\n\n", 1)[-1].split())


def test_window_overlap_carries_context_between_neighbours():
    # Overlap exists so a fact straddling a boundary is whole in at least one chunk.
    text = " ".join(f"w{i}" for i in range(11))
    parts = window(text, max_tokens=4, overlap=1)
    assert [p.split() for p in parts] == [
        ["w0", "w1", "w2", "w3"], ["w3", "w4", "w5", "w6"], ["w6", "w7", "w8", "w9"], ["w9", "w10"],
    ]


def test_window_preserves_newlines_so_code_stays_readable():
    assert window("def f():\n    return 1\n", 50, 5) == ["def f():\n    return 1"]


def test_window_rejects_overlap_not_smaller_than_max():
    with pytest.raises(ValueError):
        window("a b c", 2, 2)


def test_markdown_splits_on_headings_and_keeps_breadcrumb():
    md = "# Dependencies\n\nIntro.\n\n## With yield\n\nUse yield for cleanup.\n"
    chunks = chunk_markdown(md, 100, 10)
    assert chunks == ["Dependencies\n\nIntro.", "Dependencies > With yield\n\nUse yield for cleanup."]


def test_markdown_ignores_hash_lines_inside_code_fences():
    # A Python comment in a fenced block must not start a new section.
    md = "# Title\n\n```python\n# not a heading\nx = 1\n```\n"
    assert len(chunk_markdown(md, 100, 10)) == 1


def test_markdown_oversized_section_respects_limit():
    md = "# Big\n\n" + " ".join(["word"] * 1000)
    chunks = chunk_markdown(md, 256, 32)
    assert len(chunks) > 1
    assert all(body_tokens(c) <= 256 for c in chunks)
    assert all(c.startswith("Big\n\n") for c in chunks)


def test_python_function_chunk_includes_its_decorator():
    # Decorators like @app.get("/items") carry the route, which is the answer to many questions.
    src = 'import x\n\n\n@app.get("/items")\ndef read_items():\n    return []\n'
    chunks = chunk_python(src, 100, 10)
    assert 'import x' in chunks[0]
    assert any(c.startswith('@app.get("/items")\ndef read_items') for c in chunks)


def test_python_syntax_error_falls_back_to_window():
    assert chunk_python("def broken(:\n    pass", 100, 10) == ["def broken(:\n    pass"]


def test_python_oversized_class_is_windowed():
    src = "class Big:\n" + "".join(f"    a{i} = {i}\n" for i in range(400))
    assert all(len(c.split()) <= 128 for c in chunk_python(src, 128, 16))


def test_chunk_dispatches_by_extension():
    assert chunk("a.md", "# H\n\nbody", 50, 5) == ["H\n\nbody"]
    assert chunk("a.py", "x = 1", 50, 5) == ["x = 1"]


def test_presets_differ_only_in_what_their_name_says():
    base, hr = config.get("baseline"), config.get("hybrid_rerank")
    assert (hr.hybrid, hr.rerank) == (True, True)
    assert hr.chunker == base.chunker == "w512o64"
    with pytest.raises(ValueError, match="unknown config"):
        config.get("nope")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_chunk.py -v`
Expected: collection error `ModuleNotFoundError: No module named 'stuffrag.config'`

- [ ] **Step 3: Implement**

`stuffrag/config.py`:
```python
import os
from dataclasses import asdict, dataclass

OLLAMA_URL = os.environ.get("STUFFRAG_OLLAMA", "http://localhost:11434")

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
```

`stuffrag/chunk.py`:
```python
import ast
import re

HEADING = re.compile(r"^(#{1,6})\s+(.*)$")
TOKEN = re.compile(r"\S+\s*")  # word plus trailing whitespace, so joins keep newlines


def window(text: str, max_tokens: int, overlap: int) -> list[str]:
    """Split into windows of <= max_tokens words; neighbours share `overlap` words."""
    if not 0 <= overlap < max_tokens:
        raise ValueError(f"need 0 <= overlap < max_tokens, got {overlap=} {max_tokens=}")
    toks = TOKEN.findall(text)
    if not toks:
        return []
    if len(toks) <= max_tokens:
        return ["".join(toks).strip()]
    step = max_tokens - overlap
    return ["".join(toks[i : i + max_tokens]).strip() for i in range(0, len(toks) - overlap, step)]


def chunk_markdown(text: str, max_tokens: int, overlap: int) -> list[str]:
    """One section per heading, prefixed with its heading path; big sections are windowed."""
    sections: list[tuple[str, str]] = []
    path: list[str] = []
    lines: list[str] = []
    in_fence = False

    def flush() -> None:
        body = "\n".join(lines).strip()
        if body:
            sections.append((" > ".join(path), body))
        lines.clear()

    for line in text.splitlines():
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
        m = None if in_fence else HEADING.match(line)
        if m:
            flush()
            path = path[: len(m.group(1)) - 1] + [m.group(2).strip()]
        else:
            lines.append(line)
    flush()

    return [
        f"{crumb}\n\n{piece}" if crumb else piece
        for crumb, body in sections
        for piece in window(body, max_tokens, overlap)
    ]


def chunk_python(source: str, max_tokens: int, overlap: int) -> list[str]:
    """One chunk per top-level function/class (with decorators) plus one for module-level code."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return window(source, max_tokens, overlap)
    lines = source.splitlines()
    used: set[int] = set()
    defs: list[str] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            start = min([d.lineno for d in node.decorator_list] + [node.lineno]) - 1
            used.update(range(start, node.end_lineno))
            defs.extend(window("\n".join(lines[start : node.end_lineno]), max_tokens, overlap))
    rest = "\n".join(line for i, line in enumerate(lines) if i not in used)
    return window(rest, max_tokens, overlap) + defs


def chunk(path: str, text: str, max_tokens: int, overlap: int) -> list[str]:
    if path.endswith(".md"):
        return chunk_markdown(text, max_tokens, overlap)
    if path.endswith(".py"):
        return chunk_python(text, max_tokens, overlap)
    return window(text, max_tokens, overlap)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_chunk.py -v`
Expected: 11 passed

- [ ] **Step 5: Commit**

```bash
git add stuffrag/config.py stuffrag/chunk.py tests/test_chunk.py
git commit -m "Add PipelineConfig presets and Markdown/Python chunkers"
```

---

### Task 2: FastAPI ingest (`stuff sync fastapi`, ingest half)

**Files:**
- Create: `stuffrag/ingest/fastapi_repo.py`, `tests/test_fastapi_repo.py`, `tests/test_db.py`
- Modify: `stuffrag/db.py` (append `diff_documents`, `upsert_documents`), `stuffrag/cli.py` (add `sync` command)

**Interfaces:**
- Consumes: `db.connect()`
- Produces:
  - `fastapi_repo.FASTAPI_TAG = "0.115.12"`, `fastapi_repo.checkout(tag=FASTAPI_TAG) -> Path`
  - `fastapi_repo.expand_includes(md: str, base: Path) -> tuple[str, list[str]]` (text, unresolved include paths)
  - `fastapi_repo.collect(repo: Path, tag: str) -> tuple[list[dict], list[str]]`. Each doc dict has keys `id, source_type, title, body, metadata`; `metadata` includes `path` (repo-relative, used by the chunker), `tag`, `kind` (`doc`|`code`). Doc ids look like `fastapi:docs/en/docs/tutorial/dependencies/dependencies-with-yield.md`.
  - `db.diff_documents(existing: dict[str, str], docs: list[dict]) -> tuple[list[str], list[str]]` (new ids, changed ids)
  - `db.upsert_documents(conn, docs) -> tuple[int, int]` (new count, changed count); deletes chunks of changed docs.

- [ ] **Step 1: Check the include syntax in the real checkout**

```bash
git clone --depth 1 --branch 0.115.12 https://github.com/fastapi/fastapi.git ~/.stuffrag/fastapi/0.115.12
grep -rhoE '\{(\*|!)[^}]*\}' ~/.stuffrag/fastapi/0.115.12/docs/en/docs | head -5
ls ~/.stuffrag/fastapi/0.115.12/docs_src | head -3
```
Expected: lines like `{* ../../docs_src/dependencies/tutorial008_an_py39.py hl[...] *}`, and `docs_src/` at the repo root, so paths resolve relative to `docs/en/`. If the syntax differs, adjust the `INCLUDE` regex below and the test fixtures to match it before continuing.

- [ ] **Step 2: Write the failing tests**

`tests/test_fastapi_repo.py`:
```python
from pathlib import Path

from stuffrag.ingest.fastapi_repo import collect, expand_includes


def make_repo(tmp_path: Path) -> Path:
    (tmp_path / "docs_src/deps").mkdir(parents=True)
    (tmp_path / "docs_src/deps/tutorial008.py").write_text("async def get_db():\n    yield db\n")
    (tmp_path / "docs/en/docs/tutorial").mkdir(parents=True)
    (tmp_path / "docs/en/docs/tutorial/deps.md").write_text(
        "# Deps with yield\n\n{* ../../docs_src/deps/tutorial008.py hl[2] *}\n\n{!../../docs_src/missing.py!}\n"
    )
    (tmp_path / "fastapi").mkdir()
    (tmp_path / "fastapi/routing.py").write_text("class APIRouter:\n    pass\n")
    return tmp_path


def test_includes_are_expanded_because_the_answer_is_often_in_docs_src(tmp_path):
    repo = make_repo(tmp_path)
    md = (repo / "docs/en/docs/tutorial/deps.md").read_text()
    text, missing = expand_includes(md, repo / "docs/en")
    assert "```python\nasync def get_db():\n    yield db\n```" in text
    assert missing == ["../../docs_src/missing.py"]  # reported, not silently dropped


def test_include_cannot_escape_the_repo(tmp_path):
    repo = make_repo(tmp_path)
    (tmp_path.parent / "secret.py").write_text("nope")
    # docs/en/../../../secret.py resolves to tmp_path.parent/secret.py: exists, but outside the repo.
    text, missing = expand_includes("{* ../../../secret.py *}", repo / "docs/en")
    assert "nope" not in text and missing == ["../../../secret.py"]


def test_collect_yields_docs_and_code_with_stable_ids(tmp_path):
    docs, missing = collect(make_repo(tmp_path), "0.0.0")
    by_id = {d["id"]: d for d in docs}
    assert set(by_id) == {"fastapi:docs/en/docs/tutorial/deps.md", "fastapi:fastapi/routing.py"}
    doc = by_id["fastapi:docs/en/docs/tutorial/deps.md"]
    assert doc["title"] == "Deps with yield"
    assert doc["metadata"] == {"path": "docs/en/docs/tutorial/deps.md", "tag": "0.0.0", "kind": "doc"}
    assert by_id["fastapi:fastapi/routing.py"]["metadata"]["kind"] == "code"
    assert missing == ["docs/en/docs/tutorial/deps.md: ../../docs_src/missing.py"]
```

`tests/test_db.py`:
```python
from stuffrag.db import diff_documents


def test_diff_documents_only_flags_real_changes():
    # Re-syncing an unchanged corpus must not trigger re-chunking/re-embedding.
    existing = {"a": "same", "b": "old"}
    docs = [{"id": "a", "body": "same"}, {"id": "b", "body": "new"}, {"id": "c", "body": "x"}]
    assert diff_documents(existing, docs) == (["c"], ["b"])
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_fastapi_repo.py tests/test_db.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'stuffrag.ingest.fastapi_repo'` / `ImportError: cannot import name 'diff_documents'`

- [ ] **Step 4: Implement**

`stuffrag/ingest/fastapi_repo.py`:
```python
import re
import subprocess
from pathlib import Path

FASTAPI_TAG = "0.115.12"
REPO_URL = "https://github.com/fastapi/fastapi.git"
CHECKOUT_ROOT = Path.home() / ".stuffrag" / "fastapi"
# mkdocs include forms used by FastAPI docs; paths are relative to docs/en/.
INCLUDE = re.compile(r"\{\*\s*(\S+)[^}]*?\*\}|\{!\s*(\S+?)\s*!\}")


def checkout(tag: str = FASTAPI_TAG) -> Path:
    target = CHECKOUT_ROOT / tag
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", "--depth", "1", "--branch", tag, REPO_URL, str(target)], check=True
        )
    return target


def expand_includes(md: str, base: Path) -> tuple[str, list[str]]:
    repo = base.parent.parent.resolve()
    missing: list[str] = []

    def sub(m: re.Match) -> str:
        rel = m.group(1) or m.group(2)
        path = (base / rel).resolve()
        if not path.is_file() or not path.is_relative_to(repo):
            missing.append(rel)
            return m.group(0)
        return f"```python\n{path.read_text().rstrip()}\n```"

    return INCLUDE.sub(sub, md), missing


def _title(md: str, fallback: str) -> str:
    for line in md.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return fallback


def collect(repo: Path, tag: str) -> tuple[list[dict], list[str]]:
    docs: list[dict] = []
    missing: list[str] = []
    base = repo / "docs" / "en"
    for p in sorted((base / "docs").rglob("*.md")):
        rel = p.relative_to(repo).as_posix()
        body, miss = expand_includes(p.read_text(), base)
        missing += [f"{rel}: {m}" for m in miss]
        docs.append(_doc(rel, _title(body, rel), body, tag, "doc"))
    for p in sorted((repo / "fastapi").rglob("*.py")):
        rel = p.relative_to(repo).as_posix()
        docs.append(_doc(rel, rel, p.read_text(), tag, "code"))
    return docs, missing


def _doc(rel: str, title: str, body: str, tag: str, kind: str) -> dict:
    return {
        "id": f"fastapi:{rel}",
        "source_type": "fastapi",
        "title": title,
        "body": body,
        "metadata": {"path": rel, "tag": tag, "kind": kind},
    }
```

Append to `stuffrag/db.py` (add `from psycopg.types.json import Jsonb` to the imports):
```python
def diff_documents(existing: dict[str, str], docs: list[dict]) -> tuple[list[str], list[str]]:
    new = [d["id"] for d in docs if d["id"] not in existing]
    changed = [d["id"] for d in docs if d["id"] in existing and existing[d["id"]] != d["body"]]
    return new, changed


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
```

Add to `stuffrag/cli.py` (Task 3 extends this command to also index):
```python
@app.command()
def sync(source: str) -> None:
    """Ingest a source into the documents table. Sources: fastapi."""
    if source != "fastapi":
        raise typer.BadParameter(f"unknown source {source!r}; available: fastapi")
    from stuffrag.ingest import fastapi_repo

    repo = fastapi_repo.checkout()
    docs, missing = fastapi_repo.collect(repo, fastapi_repo.FASTAPI_TAG)
    for m in missing:
        typer.echo(f"warning: unresolved include {m}", err=True)
    with db.connect() as conn:
        new, changed = db.upsert_documents(conn, docs)
    typer.echo(f"fastapi {fastapi_repo.FASTAPI_TAG}: {len(docs)} docs ({new} new, {changed} changed), "
               f"{len(missing)} unresolved includes")
```

- [ ] **Step 5: Run tests, then the real ingest twice**

Run: `uv run pytest -v` → all pass.
Run: `uv run stuff sync fastapi` → `fastapi 0.115.12: N docs (N new, 0 changed), U unresolved includes`, with N in the hundreds. If U is more than a handful, look at the warnings and fix the regex/base before moving on.
Run it again → `(0 new, 0 changed)`.

- [ ] **Step 6: Commit**

```bash
git add stuffrag/ingest/fastapi_repo.py stuffrag/db.py stuffrag/cli.py tests/test_fastapi_repo.py tests/test_db.py
git commit -m "Ingest pinned FastAPI docs and source with include expansion"
```

---

### Task 3: Embedding + indexing (`stuff index`, `stuff sync` indexes)

**Files:**
- Create: `stuffrag/embed.py`
- Modify: `stuffrag/cli.py`, `pyproject.toml` (via `uv add httpx`)
- Test: `tests/test_embed.py`

**Interfaces:**
- Consumes: `config.PipelineConfig`, `config.EMBEDDERS`, `config.OLLAMA_URL`, `chunk.chunk`, `db.connect`
- Produces:
  - `embed.table(embedder: str) -> str` (`"bge-m3"` → `"emb_bge_m3"`)
  - `embed.to_pgvector(v: list[float]) -> str`
  - `embed.embed(texts: list[str], embedder: str, kind: str = "document") -> list[list[float]]` (`kind` is `"document"` or `"query"`)
  - `embed.index(conn, cfg, on_progress=None) -> dict` → `{"chunked_docs": int, "chunks": int, "embedded": int}`

- [ ] **Step 1: Start Ollama and pull the models**

```bash
brew services start ollama   # or run `ollama serve` in another terminal
ollama pull bge-m3 && ollama pull nomic-embed-text && ollama pull qwen3:8b
ollama --version             # need >= 0.9 for the `think` flag used in Task 4
```

- [ ] **Step 2: Write the failing tests**

`tests/test_embed.py`:
```python
from stuffrag import embed


def test_table_name_is_a_safe_identifier():
    assert embed.table("bge-m3") == "emb_bge_m3"
    assert embed.table("nomic-embed-text") == "emb_nomic_embed_text"


def test_nomic_gets_task_prefixes_bge_does_not(monkeypatch):
    # nomic-embed-text is trained with these prefixes; leaving them out measurably hurts retrieval.
    sent = []

    class R:
        def raise_for_status(self): ...
        def json(self): return {"embeddings": [[0.0]]}

    monkeypatch.setattr(embed.httpx, "post", lambda url, json, timeout: sent.append(json) or R())
    embed.embed(["q"], "nomic-embed-text", kind="query")
    embed.embed(["d"], "bge-m3")
    assert sent[0]["input"] == ["search_query: q"]
    assert sent[1]["input"] == ["d"]


def test_to_pgvector_literal():
    assert embed.to_pgvector([1.0, -0.5]) == "[1.0,-0.5]"
```

- [ ] **Step 3: Run to verify failure**

Run: `uv add httpx && uv run pytest tests/test_embed.py -v`
Expected: FAIL, `ImportError: cannot import name 'embed'`

- [ ] **Step 4: Implement**

`stuffrag/embed.py`:
```python
import re
from collections.abc import Callable

import httpx
import psycopg

from stuffrag.chunk import chunk
from stuffrag.config import EMBEDDERS, OLLAMA_URL, PipelineConfig

# (document prefix, query prefix) for models trained with task prefixes.
PREFIX = {"nomic-embed-text": ("search_document: ", "search_query: ")}


def table(embedder: str) -> str:
    return "emb_" + re.sub(r"\W", "_", embedder)


def to_pgvector(v: list[float]) -> str:
    return "[" + ",".join(map(str, v)) + "]"


def embed(texts: list[str], embedder: str, kind: str = "document") -> list[list[float]]:
    prefix = PREFIX.get(embedder, ("", ""))[0 if kind == "document" else 1]
    r = httpx.post(
        f"{OLLAMA_URL}/api/embed",
        json={"model": embedder, "input": [prefix + t for t in texts]},
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


def index(
    conn: psycopg.Connection,
    cfg: PipelineConfig,
    on_progress: Callable[[int, int], None] | None = None,
    batch: int = 32,
) -> dict:
    """Chunk documents lacking chunks for cfg.chunker, then embed chunks lacking vectors. Resumable."""
    docs = conn.execute(
        """SELECT id, metadata->>'path', body FROM documents d
           WHERE NOT EXISTS (SELECT 1 FROM chunks c WHERE c.document_id = d.id AND c.chunker = %s)""",
        (cfg.chunker,),
    ).fetchall()
    rows = [
        (doc_id, cfg.chunker, i, text)
        for doc_id, path, body in docs
        for i, text in enumerate(chunk(path or doc_id, body, cfg.chunk_tokens, cfg.chunk_overlap))
    ]
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
    return {"chunked_docs": len(docs), "chunks": len(rows), "embedded": len(todo)}
```

In `stuffrag/cli.py`, add `from stuffrag import config` and replace the last line of `sync` (the summary `typer.echo`) with the same echo followed by an index call. Then add an `index` command:
```python
def _index(cfg_name: str) -> None:
    from stuffrag import embed

    cfg = config.get(cfg_name)
    with db.connect() as conn:
        stats = embed.index(conn, cfg, on_progress=lambda d, n: typer.echo(f"  embedded {d}/{n}", err=True))
    typer.echo(f"index [{cfg.name}: {cfg.chunker}, {cfg.embedder}]: {stats['chunked_docs']} docs chunked, "
               f"{stats['chunks']} chunks added, {stats['embedded']} embedded")


@app.command("index")
def index_cmd(config_name: str = typer.Option("baseline", "--config")) -> None:
    """Chunk + embed documents for a pipeline config (idempotent)."""
    _index(config_name)
```
At the end of `sync`, add `_index("baseline")`.

- [ ] **Step 5: Run tests, then index twice**

Run: `uv run pytest -v` → all pass.
Run: `uv run stuff sync fastapi` → the index line reports non-zero chunks and embedded counts (expect a few thousand chunks; this takes minutes).
Run: `uv run stuff index --config baseline` → `0 docs chunked, 0 chunks added, 0 embedded`.
Run: `docker compose exec db psql -U stuff stuffrag -c "SELECT chunker, count(*) FROM chunks GROUP BY 1; SELECT count(*) FROM emb_bge_m3;"` → the two counts are equal.

- [ ] **Step 6: Commit**

```bash
git add stuffrag/embed.py stuffrag/cli.py tests/test_embed.py pyproject.toml uv.lock
git commit -m "Chunk and embed documents per pipeline config via Ollama"
```

---

### Task 4: Dense retrieval + cited generation (`stuff ask`)

**Files:**
- Create: `stuffrag/retrieve.py`, `stuffrag/generate.py`
- Modify: `stuffrag/cli.py`
- Test: `tests/test_generate.py`

**Interfaces:**
- Consumes: `embed.embed`, `embed.table`, `embed.to_pgvector`, `config.*`
- Produces:
  - `retrieve.Hit` (dataclass: `chunk_id: int, document_id: str, text: str, score: float`)
  - `retrieve.dense(conn, query, cfg, k) -> list[Hit]`, `retrieve.retrieve(conn, query, cfg) -> list[Hit]` (at most `cfg.top_k`)
  - `generate.REFUSAL = "Not found in my sources."`, `generate.NUM_CTX = 16384`
  - `generate.Answer` (dataclass: `text, citations: list[int], hits: list[Hit]`; property `refused: bool`)
  - `generate.parse_citations(text, hits) -> list[int]`, `generate.chat(model, system, user, json_mode=False) -> str`, `generate.generate(question, hits, cfg) -> Answer`, `generate.ask(conn, question, cfg) -> Answer`

- [ ] **Step 1: Write the failing tests**

`tests/test_generate.py`:
```python
from stuffrag import generate
from stuffrag.config import PipelineConfig
from stuffrag.retrieve import Hit

HITS = [Hit(12, "fastapi:a.md", "yield deps", 0.9), Hit(14, "fastapi:b.md", "cleanup", 0.8)]


def test_citations_keep_only_ids_that_were_in_context():
    # A hallucinated [c99] must not show up as a source the user trusts.
    assert generate.parse_citations("Use yield [c12]. Also [c99].", HITS) == [12]


def test_citations_parse_grouped_form_and_dedupe():
    assert generate.parse_citations("x [c14, c12] y [c12]", HITS) == [14, 12]


def test_refusal_detection():
    assert generate.Answer("Not found in my sources.", [], []).refused
    assert not generate.Answer("Use yield [c12].", [12], HITS).refused


def test_no_hits_refuses_without_calling_the_llm(monkeypatch):
    monkeypatch.setattr(generate, "chat", lambda *a, **k: (_ for _ in ()).throw(AssertionError("called")))
    assert generate.generate("q", [], PipelineConfig()).refused


def test_chat_sets_context_window_so_passages_are_not_silently_truncated(monkeypatch):
    # Ollama's default num_ctx would cut 5x512-word passages; the model would answer from partial context.
    sent = {}

    class R:
        def raise_for_status(self): ...
        def json(self): return {"message": {"content": "ok"}}

    monkeypatch.setattr(generate.httpx, "post", lambda url, json, timeout: sent.update(json) or R())
    generate.chat("qwen3:8b", "sys", "user")
    assert sent["options"]["num_ctx"] >= 16384
    assert sent["think"] is False and sent["options"]["temperature"] == 0
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_generate.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'stuffrag.generate'`

- [ ] **Step 3: Implement**

`stuffrag/retrieve.py`:
```python
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
```

`stuffrag/generate.py`:
```python
import re
from dataclasses import dataclass

import httpx
import psycopg

from stuffrag.config import OLLAMA_URL, PipelineConfig
from stuffrag.retrieve import Hit, retrieve

REFUSAL = "Not found in my sources."
NUM_CTX = 16384  # Ollama's default window would silently truncate top_k passages
SYSTEM = f"""You answer questions using ONLY the context passages below.
Cite every claim with its passage id in square brackets, e.g. [c12].
If the passages do not contain the answer, reply exactly: {REFUSAL}"""


@dataclass
class Answer:
    text: str
    citations: list[int]
    hits: list[Hit]

    @property
    def refused(self) -> bool:
        return self.text.strip().lower().startswith(REFUSAL.lower().rstrip("."))


def parse_citations(text: str, hits: list[Hit]) -> list[int]:
    allowed = {h.chunk_id for h in hits}
    out: list[int] = []
    for group in re.findall(r"\[([^\]]+)\]", text):
        for cid in map(int, re.findall(r"c(\d+)", group)):
            if cid in allowed and cid not in out:
                out.append(cid)
    return out


def chat(model: str, system: str, user: str, json_mode: bool = False) -> str:
    payload = {
        "model": model,
        "stream": False,
        "think": False,
        "options": {"temperature": 0, "num_ctx": NUM_CTX},
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
    }
    if json_mode:
        payload["format"] = "json"
    r = httpx.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=600)
    r.raise_for_status()
    return r.json()["message"]["content"].strip()


def generate(question: str, hits: list[Hit], cfg: PipelineConfig) -> Answer:
    if not hits:
        return Answer(REFUSAL, [], [])
    context = "\n\n".join(f"[c{h.chunk_id}] ({h.document_id})\n{h.text}" for h in hits)
    text = chat(cfg.llm, SYSTEM, f"Context:\n{context}\n\nQuestion: {question}")
    return Answer(text, parse_citations(text, hits), hits)


def ask(conn: psycopg.Connection, question: str, cfg: PipelineConfig) -> Answer:
    return generate(question, retrieve(conn, question, cfg), cfg)
```

Add to `stuffrag/cli.py`:
```python
@app.command()
def ask(question: str, config_name: str = typer.Option("baseline", "--config")) -> None:
    """Answer a question from indexed sources, with citations."""
    from stuffrag import generate

    with db.connect() as conn:
        answer = generate.ask(conn, question, config.get(config_name))
    typer.echo(answer.text)
    by_id = {h.chunk_id: h for h in answer.hits}
    if answer.citations:
        typer.echo("\nSources:")
        for cid in answer.citations:
            typer.echo(f"  [c{cid}] {by_id[cid].document_id}")
```

- [ ] **Step 4: Run tests, then a real question**

Run: `uv run pytest -v` → all pass.
Run: `uv run stuff ask "How do I declare a dependency with yield in FastAPI?"`
Expected: an answer describing a dependency function that `yield`s a value, with the code after `yield` running after the response. It has at least one `[cN]` citation, and a Sources list that includes `fastapi:docs/en/docs/tutorial/dependencies/dependencies-with-yield.md`. Also run `uv run stuff ask "What is the capital of Peru?"` → `Not found in my sources.`

- [ ] **Step 5: Commit**

```bash
git add stuffrag/retrieve.py stuffrag/generate.py stuffrag/cli.py tests/test_generate.py
git commit -m "Add dense retrieval and cited generation (stuff ask)"
```

---

### Task 5: Eval harness (`stuff eval run`, `stuff eval compare`)

**Files:**
- Create: `stuffrag/evals.py`
- Modify: `stuffrag/cli.py`, `pyproject.toml` (via `uv add pyyaml`)
- Test: `tests/test_evals.py`

**Interfaces:**
- Consumes: `generate.generate`, `generate.chat`, `retrieve.retrieve`, `embed.index`, `config.*`, `db.connect`
- Produces:
  - `evals.Question` (dataclass: `id, question, expected_answer: list[str], gold_sources: list[str], source_type: str, should_refuse: bool = False`)
  - `evals.QUESTIONS = Path("evals/questions.yaml")`, `evals.load_questions(path) -> list[Question]`
  - `evals.doc_ranking(hits) -> list[str]`, `evals.recall_at(ranked, gold, k) -> float`, `evals.mrr_at(ranked, gold, k) -> float`, `evals.keyfact_score(answer, facts) -> float`, `evals.percentile(values, p) -> float`
  - `evals.aggregate(results: list[dict]) -> dict`, `evals.compare(a: dict, b: dict) -> list[str]`, `evals.run(conn, cfg, questions) -> Path`

- [ ] **Step 1: Write the failing tests**

`tests/test_evals.py`:
```python
from stuffrag import evals
from stuffrag.retrieve import Hit


def h(cid, doc):
    return Hit(cid, doc, "", 0.0)


def test_doc_ranking_dedupes_chunks_of_the_same_doc():
    # Three chunks from one doc must not push the second doc down to rank 4.
    hits = [h(1, "a"), h(2, "a"), h(3, "a"), h(4, "b")]
    assert evals.doc_ranking(hits) == ["a", "b"]


def test_recall_and_mrr():
    ranked = ["x", "gold2", "y", "gold1"]
    assert evals.recall_at(ranked, ["gold1", "gold2"], 3) == 0.5
    assert evals.mrr_at(ranked, ["gold1", "gold2"], 10) == 0.5
    assert evals.mrr_at(ranked, ["zzz"], 10) == 0.0


def test_keyfact_is_case_insensitive_fraction():
    assert evals.keyfact_score("Use YIELD inside Depends", ["yield", "depends", "finally"]) == 2 / 3


def test_percentile_nearest_rank():
    assert evals.percentile([1, 2, 3, 4, 100], 50) == 3
    assert evals.percentile([1, 2, 3, 4, 100], 95) == 100


def test_aggregate_scores_refusal_questions_only_on_refusal():
    results = [
        {"should_refuse": False, "refused": False, "recall@5": 1.0, "mrr@10": 1.0,
         "keyfact": 1.0, "judge": True, "latency_s": 1.0},
        {"should_refuse": False, "refused": True, "recall@5": 0.0, "mrr@10": 0.0,
         "keyfact": 0.0, "judge": False, "latency_s": 2.0},
        {"should_refuse": True, "refused": True, "recall@5": None, "mrr@10": None,
         "keyfact": None, "judge": None, "latency_s": 3.0},
    ]
    m = evals.aggregate(results)
    assert m["recall@5"] == 0.5 and m["answer_correct"] == 0.5
    assert m["refusal_accuracy"] == 2 / 3  # the false refusal on q2 counts against it
    assert m["n"] == 3 and m["judge_errors"] == 0


def test_aggregate_counts_judge_errors_instead_of_hiding_them():
    r = {"should_refuse": False, "refused": False, "recall@5": 1.0, "mrr@10": 1.0,
         "keyfact": 1.0, "judge": None, "latency_s": 1.0}
    assert evals.aggregate([r])["judge_errors"] == 1


def test_compare_prints_deltas():
    a = {"id": "A", "metrics": {"recall@5": 0.5, "latency_p50": 2.0}}
    b = {"id": "B", "metrics": {"recall@5": 0.75, "latency_p50": 3.0}}
    assert evals.compare(a, b) == [
        "metric                 A        B    delta",
        "latency_p50        2.000    3.000   +1.000",
        "recall@5           0.500    0.750   +0.250",
    ]
```

- [ ] **Step 2: Run to verify failure**

Run: `uv add pyyaml && uv run pytest tests/test_evals.py -v`
Expected: FAIL, `ImportError: cannot import name 'evals'`

- [ ] **Step 3: Implement**

`stuffrag/evals.py`:
```python
import json
import math
import subprocess
import time
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path

import psycopg
import yaml
from psycopg.types.json import Jsonb

from stuffrag.config import PipelineConfig
from stuffrag.embed import index
from stuffrag.generate import chat, generate
from stuffrag.retrieve import Hit, retrieve

QUESTIONS = Path("evals/questions.yaml")
RUNS = Path("evals/runs")
JUDGE_SYSTEM = """You grade an answer against expected key facts.
Reply with JSON {"correct": true} if the answer states all key facts and nothing contradicting them,
else {"correct": false}."""


@dataclass
class Question:
    id: str
    question: str
    expected_answer: list[str]
    gold_sources: list[str]
    source_type: str
    should_refuse: bool = False


def load_questions(path: Path = QUESTIONS) -> list[Question]:
    return [Question(**q) for q in yaml.safe_load(path.read_text())]


def doc_ranking(hits: list[Hit]) -> list[str]:
    return list(dict.fromkeys(h.document_id for h in hits))


def recall_at(ranked: list[str], gold: list[str], k: int) -> float:
    return len(set(ranked[:k]) & set(gold)) / len(gold)


def mrr_at(ranked: list[str], gold: list[str], k: int) -> float:
    return next((1 / i for i, d in enumerate(ranked[:k], 1) if d in gold), 0.0)


def keyfact_score(answer: str, facts: list[str]) -> float:
    return sum(f.lower() in answer.lower() for f in facts) / len(facts)


def percentile(values: list[float], p: float) -> float:
    s = sorted(values)
    return s[max(0, math.ceil(p / 100 * len(s)) - 1)]


def judge(q: Question, answer: str, cfg: PipelineConfig) -> bool | None:
    user = f"Question: {q.question}\nKey facts: {q.expected_answer}\nAnswer: {answer}"
    try:
        return bool(json.loads(chat(cfg.llm, JUDGE_SYSTEM, user, json_mode=True))["correct"])
    except (json.JSONDecodeError, KeyError, TypeError):
        return None  # counted as judge_errors


def _mean(xs: list) -> float | None:
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def aggregate(results: list[dict]) -> dict:
    answerable = [r for r in results if not r["should_refuse"]]
    lat = [r["latency_s"] for r in results]
    return {
        "n": len(results),
        "recall@5": _mean([r["recall@5"] for r in answerable]),
        "mrr@10": _mean([r["mrr@10"] for r in answerable]),
        "keyfact": _mean([r["keyfact"] for r in answerable]),
        "answer_correct": _mean([r["judge"] for r in answerable]),
        "judge_errors": sum(r["judge"] is None for r in answerable),
        "refusal_accuracy": _mean([r["refused"] == r["should_refuse"] for r in results]),
        "latency_p50": percentile(lat, 50) if lat else None,
        "latency_p95": percentile(lat, 95) if lat else None,
    }


def evaluate(conn: psycopg.Connection, q: Question, cfg: PipelineConfig) -> dict:
    t0 = time.perf_counter()
    hits = retrieve(conn, q.question, replace(cfg, top_k=max(cfg.top_k, 10)))  # 10 for MRR@10
    answer = generate(q.question, hits[: cfg.top_k], cfg)
    latency = time.perf_counter() - t0
    ranked = doc_ranking(hits)
    r = {"id": q.id, "should_refuse": q.should_refuse, "refused": answer.refused,
         "answer": answer.text, "citations": answer.citations, "retrieved": ranked[:10],
         "latency_s": latency, "recall@5": None, "mrr@10": None, "keyfact": None, "judge": None}
    if not q.should_refuse:
        r["recall@5"] = recall_at(ranked, q.gold_sources, 5)
        r["mrr@10"] = mrr_at(ranked, q.gold_sources, 10)
        r["keyfact"] = keyfact_score(answer.text, q.expected_answer)
        r["judge"] = False if answer.refused else judge(q, answer.text, cfg)
    return r


def git_sha() -> str:
    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True).stdout.strip()
    return sha + ("-dirty" if dirty else "")


def run(conn: psycopg.Connection, cfg: PipelineConfig, questions: list[Question]) -> Path:
    index(conn, cfg)  # each ablation is one flag change: make sure its chunks/vectors exist
    results = [evaluate(conn, q, cfg) for q in questions]
    run_id = f"{datetime.now():%Y%m%d-%H%M%S}-{cfg.name}"
    doc = {"id": run_id, "config": cfg.to_dict(), "git_sha": git_sha(),
           "metrics": aggregate(results), "results": results}
    RUNS.mkdir(parents=True, exist_ok=True)
    path = RUNS / f"{run_id}.json"
    path.write_text(json.dumps(doc, indent=2))
    conn.execute(
        "INSERT INTO eval_runs (id, config, git_sha, metrics) VALUES (%s, %s, %s, %s)",
        (run_id, Jsonb(doc["config"]), doc["git_sha"], Jsonb(doc["metrics"])),
    )
    conn.commit()
    return path


def compare(a: dict, b: dict) -> list[str]:
    def fmt(x):
        return f"{x:8.3f}" if isinstance(x, (int, float)) else f"{'-':>8}"

    lines = [f"{'metric':<16}{a['id']:>8} {b['id']:>8}    delta"]
    for m in sorted(set(a["metrics"]) | set(b["metrics"])):
        x, y = a["metrics"].get(m), b["metrics"].get(m)
        delta = f"{y - x:+8.3f}" if isinstance(x, (int, float)) and isinstance(y, (int, float)) else f"{'-':>8}"
        lines.append(f"{m:<16}{fmt(x)} {fmt(y)} {delta}")
    return lines
```

Add to `stuffrag/cli.py`:
```python
eval_app = typer.Typer()
app.add_typer(eval_app, name="eval")


@eval_app.command("run")
def eval_run(config_name: str = typer.Option("baseline", "--config")) -> None:
    """Run all eval questions through a pipeline config and record metrics."""
    from stuffrag import evals

    with db.connect() as conn:
        path = evals.run(conn, config.get(config_name), evals.load_questions())
    import json
    typer.echo(f"{path}\n{json.dumps(json.loads(path.read_text())['metrics'], indent=2)}")


@eval_app.command("compare")
def eval_compare(run_a: Path, run_b: Path) -> None:
    """Print metric deltas between two run files (B - A)."""
    import json
    from stuffrag import evals

    for line in evals.compare(json.loads(run_a.read_text()), json.loads(run_b.read_text())):
        typer.echo(line)
```
(add `from pathlib import Path` to the cli imports)

- [ ] **Step 4: Run tests**

Run: `uv run pytest -v`
Expected: all pass. The exact column spacing in `test_compare_prints_deltas` has to match `compare()`. If only the spacing differs, fix the expected strings in the test to the real format; if a value or ordering differs, fix `compare()`.

- [ ] **Step 5: Commit**

```bash
git add stuffrag/evals.py stuffrag/cli.py tests/test_evals.py pyproject.toml uv.lock
git commit -m "Add eval harness: retrieval, answer, refusal and latency metrics"
```

---

### Task 6: 25 FastAPI eval questions + validator + baseline run

**Files:**
- Create: `evals/questions.yaml`, `tests/test_questions.py`

**Interfaces:**
- Consumes: `evals.load_questions`, `fastapi_repo.CHECKOUT_ROOT`, `fastapi_repo.FASTAPI_TAG`, `fastapi_repo.expand_includes`

- [ ] **Step 1: Write the validator test first**

`tests/test_questions.py`:
```python
import pytest

from stuffrag.evals import QUESTIONS, load_questions
from stuffrag.ingest.fastapi_repo import CHECKOUT_ROOT, FASTAPI_TAG, expand_includes

REPO = CHECKOUT_ROOT / FASTAPI_TAG
QS = load_questions(QUESTIONS)


def test_ids_unique_and_fastapi_count():
    assert len({q.id for q in QS}) == len(QS)
    assert sum(q.source_type == "fastapi" for q in QS) == 25


@pytest.mark.skipif(not REPO.exists(), reason=f"run `stuff sync fastapi` first ({REPO} missing)")
@pytest.mark.parametrize("q", [q for q in QS if q.source_type == "fastapi"], ids=lambda q: q.id)
def test_every_key_fact_is_in_the_gold_sources(q):
    # A key fact missing from its gold file means the question is wrong, not the pipeline.
    assert q.gold_sources and q.expected_answer
    text = ""
    for gid in q.gold_sources:
        path = REPO / gid.removeprefix("fastapi:")
        assert path.is_file(), f"{q.id}: gold source {gid} does not exist at {FASTAPI_TAG}"
        text += expand_includes(path.read_text(), REPO / "docs/en")[0].lower()
    for fact in q.expected_answer:
        assert fact.lower() in text, f"{q.id}: key fact {fact!r} not found in {q.gold_sources}"
```

- [ ] **Step 2: Draft the 25 questions from the pinned docs**

Read the docs under `~/.stuffrag/fastapi/0.115.12/docs/en/docs/` (tutorial, advanced, and a few `fastapi/*.py` files). Write 25 entries that spread across topics: path/query params, request body, dependencies (incl. yield), security/OAuth2, middleware, CORS, background tasks, lifespan events, testing, `APIRouter`, response models, status codes, exceptions, WebSockets, settings, and 3 or more questions answered by `fastapi/` source code. Each `expected_answer` item is a short verbatim phrase (2–5 words) that must appear in a correct answer and does appear in the gold file. Format:

```yaml
- id: fa-01
  question: How do I declare a dependency with yield in FastAPI, and when does the code after yield run?
  expected_answer: ["yield", "after the response"]
  gold_sources: ["fastapi:docs/en/docs/tutorial/dependencies/dependencies-with-yield.md"]
  source_type: fastapi
- id: fa-02
  question: Which class do I use to split path operations across multiple files?
  expected_answer: ["APIRouter", "include_router"]
  gold_sources: ["fastapi:docs/en/docs/tutorial/bigger-applications.md"]
  source_type: fastapi
```
(Keep these two only if the validator passes on them. Ids run `fa-01`…`fa-25`.)

- [ ] **Step 3: Run the validator; fix questions until it passes**

Run: `uv run pytest tests/test_questions.py -v`
Expected: 26 passed, 0 skipped. A failure means you fix the question (wrong path or a fact that isn't in the file), never the test.

- [ ] **Step 4: Record the baseline**

Run: `uv run stuff eval run --config baseline`
Expected: prints the `evals/runs/<ts>-baseline.json` path and metrics with `n: 25` and non-null `recall@5`, `mrr@10`, `keyfact`, `answer_correct`, `latency_p50/p95`. Report `judge_errors` if it's non-zero. Also check: `docker compose exec db psql -U stuff stuffrag -c "SELECT id, git_sha FROM eval_runs"` shows the run, with a SHA that doesn't end in `-dirty` (commit first if it does, then re-run).

- [ ] **Step 5: Commit**

```bash
git add evals/questions.yaml tests/test_questions.py
git commit -m "Add 25 FastAPI eval questions with gold-source validator; baseline recorded"
```
Put the baseline metrics JSON in the commit body. Run files are gitignored, so the commit message is where the record stays.

---

### Task 7: Hybrid (full-text + RRF) + reranker, ablation runs, judge spot-check

**Files:**
- Create: `stuffrag/rerank.py`
- Modify: `stuffrag/retrieve.py` (add `fulltext`, `rrf`; replace `retrieve`), `pyproject.toml` (via `uv add sentence-transformers`)
- Test: `tests/test_retrieve.py`

**Interfaces:**
- Consumes: `retrieve.Hit`, `retrieve.dense`, `PipelineConfig.hybrid/rerank/candidate_k/top_k`
- Produces: `retrieve.fulltext(conn, query, cfg, k) -> list[Hit]`, `retrieve.rrf(rankings: list[list[Hit]], k: int = 60) -> list[Hit]`, `rerank.rerank(query, hits) -> list[Hit]` (sorted by cross-encoder score, descending)

- [ ] **Step 1: Write the failing tests**

`tests/test_retrieve.py`:
```python
from stuffrag import retrieve
from stuffrag.config import PipelineConfig
from stuffrag.retrieve import Hit, rrf


def h(cid):
    return Hit(cid, f"d{cid}", "", 0.0)


def test_rrf_rewards_agreement_between_retrievers():
    # Chunk 2 is 2nd in both lists; it should beat chunks that are 1st in only one.
    fused = rrf([[h(1), h(2), h(3)], [h(4), h(2), h(5)]])
    assert fused[0].chunk_id == 2
    assert {x.chunk_id for x in fused} == {1, 2, 3, 4, 5}


def test_rrf_with_empty_fulltext_keeps_dense_order():
    # Stopword-only questions ("what is it?") give no full-text hits; dense must still answer.
    assert [x.chunk_id for x in rrf([[h(1), h(2)], []])] == [1, 2]


def test_retrieve_hybrid_rerank_pipeline_order(monkeypatch):
    calls = []
    monkeypatch.setattr(retrieve, "dense", lambda c, q, cfg, k: calls.append(("dense", k)) or [h(1), h(2)])
    monkeypatch.setattr(retrieve, "fulltext", lambda c, q, cfg, k: calls.append(("ft", k)) or [h(3)])
    monkeypatch.setattr(retrieve, "rerank", lambda q, hits: calls.append(("rr", len(hits))) or hits[::-1])
    cfg = PipelineConfig(hybrid=True, rerank=True, top_k=2, candidate_k=30)
    out = retrieve.retrieve(None, "q", cfg)
    assert calls == [("dense", 30), ("ft", 30), ("rr", 3)]
    assert len(out) == 2
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_retrieve.py -v`
Expected: FAIL, `ImportError: cannot import name 'rrf'`

- [ ] **Step 3: Implement**

Run: `uv add sentence-transformers`

`stuffrag/rerank.py`:
```python
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
```

In `stuffrag/retrieve.py`: add `from dataclasses import dataclass, replace`, then add these two functions and replace `retrieve` (`rerank` is imported inside the function to avoid a circular import; the test monkeypatches the module attribute `retrieve.rerank`, so bind it at module level lazily as shown):
```python
def fulltext(conn: psycopg.Connection, query: str, cfg: PipelineConfig, k: int) -> list[Hit]:
    # OR the query lexemes: plainto_tsquery ANDs them, which matches almost nothing for full questions.
    rows = conn.execute(
        """WITH q AS (SELECT replace(plainto_tsquery('english', %(q)s)::text, '&', '|')::tsquery AS q)
           SELECT c.id, c.document_id, c.text, ts_rank_cd(c.tsv, q.q)
           FROM chunks c, q WHERE c.chunker = %(chunker)s AND c.tsv @@ q.q
           ORDER BY 4 DESC LIMIT %(k)s""",
        {"q": query, "chunker": cfg.chunker, "k": k},
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


def retrieve(conn: psycopg.Connection, query: str, cfg: PipelineConfig) -> list[Hit]:
    pool = cfg.candidate_k if (cfg.hybrid or cfg.rerank) else cfg.top_k
    hits = dense(conn, query, cfg, pool)
    if cfg.hybrid:
        hits = rrf([hits, fulltext(conn, query, cfg, pool)])
    if cfg.rerank:
        hits = rerank(query, hits)
    return hits[: cfg.top_k]
```
Note for evals: `evaluate` calls `retrieve` with `top_k=10`, so `pool` stays `candidate_k=30` for hybrid/rerank and 10 for baseline. The baseline's MRR@10 therefore sees 10 dense hits, which is correct.

- [ ] **Step 4: Run unit tests + a stopword-only live query**

Run: `uv run pytest -v` → all pass, 0 skipped.
Run: `uv run stuff ask "what is it?" --config hybrid` → no crash (an answer or the refusal string).

- [ ] **Step 5: Ablation runs + compare**

```bash
uv run stuff eval run --config hybrid
uv run stuff eval run --config hybrid_rerank
uv run stuff eval compare evals/runs/*-baseline.json evals/runs/*-hybrid.json
uv run stuff eval compare evals/runs/*-baseline.json evals/runs/*-hybrid_rerank.json
```
Expected: three run files and two delta tables. Report the deltas as they come out, including regressions. Whether hybrid or rerank helps is the finding, not a pass condition. If more than one baseline file exists, pass the newest explicitly instead of the glob.

Optional ablations from the spec (each is one command, and indexing happens automatically): `--config nomic`, `--config chunk256`, `--config chunk1024`.

- [ ] **Step 6: Judge spot-check (needs the human)**

```bash
uv run python -c "
import json,glob; r=json.load(open(sorted(glob.glob('evals/runs/*-baseline.json'))[-1]))
for x in [x for x in r['results'] if x['judge'] is not None][:10]:
    print(f\"--- {x['id']} judge={x['judge']} keyfact={x['keyfact']:.2f}\n{x['answer'][:600]}\n\")"
```
Show the 10 judgments to Toprak alongside the questions/expected facts. Record how many they agree with (e.g. `judge agreement: 8/10`) in the commit body below.

- [ ] **Step 7: Commit**

```bash
git add stuffrag/rerank.py stuffrag/retrieve.py tests/test_retrieve.py pyproject.toml uv.lock
git commit -m "Add full-text+RRF hybrid and cross-encoder rerank; record ablation deltas"
```
Put both compare tables and the judge agreement in the commit body.
