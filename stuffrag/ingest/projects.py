import json
import os
import re
import subprocess
import tomllib
from collections import Counter
from fnmatch import fnmatch
from pathlib import Path

ROOT = Path(os.environ.get("STUFFRAG_PROJECTS", Path.home() / "projects"))
# Third-party clones and backups: they'd crowd my own projects out of retrieval.
SKIP_PROJECTS = {"agency-agents", "career-ops", "LLaVA", "llama-vision-boilerplate", "morethantasks backup"}

EXTS = {".md", ".txt", ".rst", ".py", ".ipynb", ".ts", ".tsx", ".js", ".jsx", ".mjs", ".cs", ".gd", ".go",
        ".rs", ".java", ".kt", ".swift", ".c", ".cpp", ".h", ".sql", ".sh", ".yaml", ".yml", ".toml",
        ".html", ".css", ".php", ".vue", ".mdc"}
MANIFESTS = {"package.json", "Dockerfile"}
LOCKFILES = {"package-lock.json", "yarn.lock", "pnpm-lock.yaml", "poetry.lock", "uv.lock", "Cargo.lock"}
PRUNE = {"docs", "node_modules", "vendor", ".venv", "venv", ".next", "dist", "build", "__pycache__", ".git", "site-packages"}
DOC_EXTS = {".md", ".txt", ".rst"}
MAX_BYTES = 100_000

# Secrets are never indexed: a match skips the whole file (no redaction to get wrong).
SECRET_NAMES = [".env", ".env.*", ".envrc", "*.pem", "*.key", "*.p12", "*.pfx", "id_rsa*", "id_ed25519*",
                "*credentials*", "*secret*", "token*.json", "service-account*.json", ".npmrc", ".netrc", ".pypirc"]
SECRET_TEXT = re.compile("|".join([
    r"-----BEGIN [A-Z ]*PRIVATE KEY-----",
    r"AKIA[0-9A-Z]{16}",
    r"sk-(?:ant-)?[A-Za-z0-9_-]{20,}",
    r"gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{20,}",
    r"xox[bpas]-[A-Za-z0-9-]{10,}",
    r"AIza[0-9A-Za-z_-]{35}",
    r"(?i:api[_-]?key|secret|passw(?:or)?d|token)[\"']?\s*[:=]\s*[\"'][^\"'\s]{12,}[\"']",
]))


def is_secret(path: str, text: str) -> bool:
    name = Path(path).name.lower()
    return any(fnmatch(name, p) for p in SECRET_NAMES) or bool(SECRET_TEXT.search(text))


def notebook_text(raw: str) -> str:
    """Markdown + code cell sources only; outputs are bulky and can echo data."""
    try:
        cells = json.loads(raw).get("cells", [])
    except (json.JSONDecodeError, AttributeError):
        return ""
    return "\n\n".join("".join(c.get("source", [])) for c in cells if c.get("cell_type") in ("markdown", "code"))


def _allowed(rel: str) -> bool:
    p = Path(rel)
    if p.name in LOCKFILES or p.name.endswith(".min.js") or PRUNE & set(p.parts[:-1]):
        return False
    return p.suffix in EXTS or p.name in MANIFESTS or p.name.startswith("docker-compose")


def _list_files(project: Path) -> list[str]:
    if (project / ".git").exists():
        out = subprocess.run(["git", "-C", str(project), "ls-files", "-z", "--cached", "--others",
                              "--exclude-standard"], capture_output=True, check=True).stdout
        return sorted({f for f in out.decode().split("\0") if f})
    files = []
    for dirpath, dirnames, filenames in os.walk(project):
        dirnames[:] = [d for d in dirnames if d not in PRUNE and not d.startswith(".")]
        files += [(Path(dirpath) / f).relative_to(project).as_posix() for f in filenames]
    return sorted(files)


def _read(path: Path) -> str | None:
    if not path.is_file() or path.stat().st_size > MAX_BYTES:
        return None
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return None
    return notebook_text(text) if path.suffix == ".ipynb" else text


def _git_history(project: Path) -> list[str]:
    if not (project / ".git").exists():
        return []
    r = subprocess.run(["git", "-C", str(project), "log", "--format=%ad %s", "--date=short"],
                       capture_output=True, text=True)
    return r.stdout.splitlines() if r.returncode == 0 else []


def _dependencies(files: dict[str, str]) -> list[str]:
    deps: list[str] = []
    for rel, body in files.items():
        name = Path(rel).name
        try:
            if name == "package.json":
                pkg = json.loads(body)
                deps += [*pkg.get("dependencies", {}), *pkg.get("devDependencies", {})]
            elif name == "pyproject.toml":
                deps += tomllib.loads(body).get("project", {}).get("dependencies", [])
            elif name == "requirements.txt":
                deps += [ln.strip() for ln in body.splitlines() if ln.strip() and not ln.startswith("#")]
        except (json.JSONDecodeError, tomllib.TOMLDecodeError, AttributeError):
            continue
    return sorted(set(deps))


def overview(name: str, files: dict[str, str], history: list[str]) -> str:
    """Deterministic project summary for big-picture questions (README, stack, layout, history)."""
    readme = next((files[r] for r in files if r.lower() in ("readme.md", "readme.rst", "readme.txt")), "")
    langs = Counter(Path(r).suffix or Path(r).name for r in files)
    layout = sorted({"/".join(Path(r).parts[:2]) + ("/" if len(Path(r).parts) > 2 else "") for r in files})
    parts = [f"# Project: {name}", "## README", readme.strip()[:6000] or "(none)",
             "## Dependencies", ", ".join(_dependencies(files)) or "(none found)",
             "## Languages", ", ".join(f"{ext}: {n}" for ext, n in langs.most_common()),
             "## Layout", "\n".join(layout[:80])]
    if history:
        parts += ["## Git history", f"first commit {history[-1][:10]}, last commit {history[0][:10]}, "
                  f"{len(history)} commits. Recent:", "\n".join(f"- {h}" for h in history[:15])]
    return "\n\n".join(parts)


def collect(root: Path = ROOT, skip: set[str] = SKIP_PROJECTS) -> tuple[list[dict], list[str]]:
    """Returns (documents, paths skipped as secrets). Only paths are reported, never content."""
    docs: list[dict] = []
    skipped: list[str] = []
    for project in sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")):
        if project.name in skip:
            continue
        kept: dict[str, str] = {}
        for rel in _list_files(project):
            label = f"{project.name}/{rel}"
            if is_secret(rel, ""):
                skipped.append(label)
                continue
            if not _allowed(rel) or (body := _read(project / rel)) is None or not body.strip():
                continue
            if is_secret(rel, body):
                skipped.append(label)
                continue
            kept[rel] = body
            kind = "doc" if Path(rel).suffix in DOC_EXTS else "code"
            docs.append(_doc(project.name, rel, rel, body, kind, label))
        if kept:
            ov = overview(project.name, kept, _git_history(project))
            if is_secret("__overview__.md", ov):
                skipped.append(f"{project.name}/__overview__")
            else:
                docs.append(_doc(project.name, "__overview__", "__overview__.md", ov, "overview",
                                 f"{project.name} (project overview)"))
    return docs, skipped


def _doc(project: str, id_rel: str, path: str, body: str, kind: str, label: str) -> dict:
    return {
        "id": f"projects:{project}/{id_rel}",
        "source_type": "projects",
        "title": label,
        "body": body,
        "metadata": {"project": project, "path": path, "kind": kind, "label": label},
    }
