import re
import subprocess
from pathlib import Path

FASTAPI_TAG = "0.115.12"
REPO_URL = "https://github.com/fastapi/fastapi.git"
CHECKOUT_ROOT = Path.home() / ".stuffrag" / "fastapi"
# mkdocs include forms used by FastAPI docs; paths are relative to docs/en/.
# {* path hl[...] *}   -- the primary "snippet" form
# {!path!} / {!> path!} -- mkdocs-include-markdown forms (the '>' variant is common in the repo)
INCLUDE = re.compile(r"\{\*\s*(\S+)[^}]*?\*\}|\{!\s*>?\s*(\S+?)\s*!\}")


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
        snippet_form = m.group(1) is not None
        rel = m.group(1) or m.group(2)
        path = (base / rel).resolve()
        if not path.is_file() or not path.is_relative_to(repo):
            missing.append(rel)
            return m.group(0)
        content = path.read_text().rstrip()
        if snippet_form:
            # {* path *} is a standalone mkdocs "snippet" -- not inside a fence.
            return f"```python\n{content}\n```"
        # {!path!} / {!> path!} are mkdocs-include-markdown inserts that FastAPI's
        # docs always place inside an existing ``` fence -- insert raw.
        return content

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
