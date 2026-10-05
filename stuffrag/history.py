import json
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from stuffrag.config import PipelineConfig
from stuffrag.generate import chat
from stuffrag.ingest.projects import ROOT, SKIP_PROJECTS, _allowed, is_secret
from stuffrag.retrieve import retrieve

COMMIT_CAP = 6_000   # floor of filtered-diff chars per commit (a commit gets more if the total allows)
TOTAL_CAP = 40_000   # chars of diff across all commits sent to the LLM
OMITTED = "(omitted: matched a secret pattern)"
TRUNCATED = "\n... (diff truncated: see the full file list above)\n"
FILE_HEADER = re.compile(r"^diff --git a/.* b/(.*)$", re.M)


class HistoryError(Exception):
    pass


@dataclass
class Commit:
    sha: str
    date: str
    subject: str
    body: str
    pr: dict | None = None
    diff: str = ""
    stat: str = ""
    secret_files: list[str] = field(default_factory=list)
    hidden_files: int = 0
    reduced: bool = False


def _git(repo: Path, *args: str) -> str:
    r = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    if r.returncode:
        raise HistoryError(f"git {args[0]} failed: {r.stderr.strip()}")
    return r.stdout


def repo_for(project: str, root: Path = ROOT) -> Path:
    path = root / project
    if project in SKIP_PROJECTS or not path.is_dir():
        raise HistoryError(f"unknown project {project!r}")
    if not (path / ".git").exists():
        raise HistoryError(f"{project} has no git history")
    return path


LOG_FORMAT = "--format=%H%x1f%ad%x1f%s%x1f%b%x1e"


def _parse(out: str) -> list[Commit]:
    commits = []
    for rec in out.split("\x1e"):
        if rec.strip():
            sha, date, subject, body = rec.strip("\n").split("\x1f")
            commits.append(Commit(sha, date, subject, body.strip()))
    return commits


BARE_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")


def commits(repo: Path, since: str, until: str | None = None) -> list[Commit]:
    """First-parent only: a merged PR is one unit, not its merge plus every branch commit."""
    # git reads a bare date as that day at the *current* time of day; mean whole days instead.
    if BARE_DATE.fullmatch(since):
        since += " 00:00:00"
    if until and BARE_DATE.fullmatch(until):
        until += " 23:59:59"
    args = ["log", "--first-parent", LOG_FORMAT, "--date=short", f"--since={since}"]
    if until:
        args.append(f"--until={until}")
    return _parse(_git(repo, *args))


def commit_info(repo: Path, sha: str) -> Commit:
    return _parse(_git(repo, "log", "-1", LOG_FORMAT, "--date=short", sha))[0]


def _range(repo: Path, sha: str) -> list[str]:
    parents = _git(repo, "rev-list", "--parents", "-n1", sha).split()[1:]
    return [parents[0], sha] if parents else ["4b825dc642cb6eb9a060e54bf8d69288fbee4904", sha]  # empty tree


def _scrub(text: str) -> str:
    return OMITTED if text and is_secret("message", text) else text


def load(repo: Path, c: Commit, prs: dict[str, dict], path: str | None = None) -> None:
    """Fill the filtered diff, stat and PR; secret files are dropped whole and only their paths kept."""
    rng = _range(repo, c.sha)
    scope = ["--", path] if path else []
    raw = _git(repo, "diff", "--no-color", *rng, *scope)
    c.stat = _git(repo, "diff", "--stat", *rng, *scope).strip()
    kept = []
    starts = [m.start() for m in FILE_HEADER.finditer(raw)] + [len(raw)]
    for a, b in zip(starts, starts[1:]):
        chunk = raw[a:b]
        fpath = FILE_HEADER.match(chunk).group(1)
        if is_secret(fpath, chunk):
            c.secret_files.append(fpath)
        elif not _allowed(fpath) or "\nBinary files " in chunk:
            c.hidden_files += 1
        else:
            kept.append(chunk)
    c.diff = "".join(kept)  # capped later by budget(), which knows how many commits share the context
    c.subject, c.body = _scrub(c.subject), _scrub(c.body)
    pr = prs.get(c.sha)
    c.pr = pr and {**pr, "title": _scrub(pr.get("title", "")), "body": _scrub(pr.get("body") or "")}


def budget(commits: list[Commit], notes: list[str]) -> None:
    """Cap each diff at its share of the context, then reduce the largest to --stat until it fits."""
    cap = max(COMMIT_CAP, TOTAL_CAP // max(len(commits), 1))
    for c in commits:
        if len(c.diff) > cap:  # marker inside the cap, or a lone commit overshoots and gets reduced
            c.diff = c.diff[: cap - len(TRUNCATED)] + TRUNCATED
    while sum(len(c.diff) for c in commits) > TOTAL_CAP:
        full = [c for c in commits if not c.reduced]
        if not full:
            break
        big = max(full, key=lambda c: len(c.diff))
        big.diff, big.reduced = big.stat, True
        notes.append(f"{big.sha[:7]} too large: shown as file stats only")


GITHUB = re.compile(r"github\.com[:/](.+?)(?:\.git)?/?$")
NO_PRS = "using commit messages only"


def _gh(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["gh", *args], capture_output=True, text=True, timeout=60)


def pr_map(repo: Path) -> tuple[dict[str, dict], str | None]:
    """Merged PRs keyed by merge commit SHA (works for merge and squash). Read-only; never fails."""
    try:
        origin = _git(repo, "remote", "get-url", "origin").strip()
    except HistoryError:
        return {}, f"no origin remote; {NO_PRS}"
    m = GITHUB.search(origin)
    if not m:
        return {}, f"origin is not on GitHub; {NO_PRS}"
    try:
        r = _gh("pr", "list", "-R", m.group(1), "--state", "merged", "--limit", "500",
                "--json", "number,title,body,mergeCommit")
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        return {}, f"gh unavailable ({type(e).__name__}); {NO_PRS}"
    if r.returncode:
        return {}, f"gh failed ({r.stderr.strip()[:200]}); {NO_PRS}"
    return {p["mergeCommit"]["oid"]: p for p in json.loads(r.stdout) if p.get("mergeCommit")}, None


def merge_of(repo: Path, sha: str) -> str | None:
    """The first-parent merge that brought `sha` into the current branch (None for direct commits)."""
    mainline = _git(repo, "rev-list", "--first-parent", "HEAD").split()
    if sha in mainline:
        return None  # on the mainline itself: any later merge is unrelated
    # Not `log --first-parent --ancestry-path`: that only follows first-parent edges, so a branch
    # commit more than one step below its merge is never seen as reaching it.
    merges = set(_git(repo, "rev-list", "--ancestry-path", "--merges", f"{sha}..HEAD").split())
    return next((m for m in reversed(mainline) if m in merges), None)


CHANGES_SYSTEM = """You explain what changed in a software project, using ONLY the commits below
(commit message, PR description, diff). Write three sections: Added, Removed, Changed.
Cite every point with its commit id in square brackets, e.g. [abc1234].
If a message or PR description claims something its diff does not show, or the diff does something
the message does not mention, list it under a fourth section "Message vs diff". Do not invent changes.
A diff marked truncated (or "file list only") is incomplete: the file list is complete, so never claim
a message is unsupported because its change is missing from a truncated diff."""


@dataclass
class Report:
    text: str
    commits: list[Commit]
    notes: list[str]


def render(commits: list[Commit]) -> str:
    parts = []
    for c in commits:
        head = f"[{c.sha[:7]}] {c.date} {c.subject}"
        if c.pr:
            head += f"\nPR #{c.pr['number']}: {c.pr['title']}\n{c.pr['body']}"
        if c.body:
            head += f"\nMessage:\n{c.body}"
        diff = "(too large: file list only)" if c.reduced else (c.diff or "(no code changes shown)")
        parts.append(f"{head}\nFiles changed:\n{c.stat or '(none)'}\nDiff:\n{diff}")
    return "\n\n---\n\n".join(parts)


def omission_notes(commits: list[Commit]) -> list[str]:
    secret = sorted({p for c in commits for p in c.secret_files})
    hidden = sum(c.hidden_files for c in commits)
    notes = [f"{len(secret)} files omitted as secret: {', '.join(secret)}"] if secret else []
    return notes + ([f"{hidden} hidden/non-code file diffs not shown"] if hidden else [])


def changes(project: str, since: str, until: str | None, cfg: PipelineConfig, root: Path | None = None) -> Report:
    repo = repo_for(project, root or ROOT)
    cs = commits(repo, since, until)
    if not cs:
        raise HistoryError(f"no commits in {project} for since={since!r} until={until!r}")
    prs, warning = pr_map(repo)
    for c in cs:
        load(repo, c, prs)
    notes = [warning] if warning else []
    budget(cs, notes)
    text = chat(cfg.llm, CHANGES_SYSTEM, f"Project: {project}\n\n{render(cs)}")
    return Report(text, cs, notes + omission_notes(cs))


COMMENT = ("#", "//", "/*", "*", "<!--", "--")
WHY_SYSTEM = """You explain when and why a piece of code was added, using ONLY the commits below
(commit message, PR description, diff of the file where the code lives). Say when it was added, why
(from the message/PR; say so if they give no reason), and what the change did. Cite commits as [abc1234].
A diff marked truncated is incomplete; do not draw conclusions from what it does not show."""
NOT_FOUND = "Not found: could not identify the commit that introduced this code, so I won't guess."


def introducing_line(text: str, document_id: str) -> str | None:
    """Most distinctive line of a chunk (label stripped, comments skipped, >=20 non-space chars)."""
    label = document_id.removeprefix("projects:")
    body = text.removeprefix(label + "\n\n")
    lines = [ln.strip() for ln in body.splitlines()]
    good = [ln for ln in lines if not ln.startswith(COMMENT) and len(ln.replace(" ", "")) >= 20]
    return max(good, key=len) if good else None


def introducing_commit(repo: Path, path: str, line: str) -> str | None:
    shas = _git(repo, "log", "-S", line, "--reverse", "--format=%H", "--", path).split()
    return shas[0] if shas else None


def why(conn, project: str, topic: str, cfg: PipelineConfig, root: Path | None = None) -> Report:
    repo = repo_for(project, root or ROOT)
    prs, warning = pr_map(repo)
    notes = [warning] if warning else []
    found: list[Commit] = []
    for h in retrieve(conn, topic, cfg, project)[:3]:
        if h.document_id.endswith("/__overview__"):
            continue
        path = h.document_id.removeprefix(f"projects:{project}/")
        line = introducing_line(h.text, h.document_id)
        sha = line and introducing_commit(repo, path, line)
        if not sha or sha in {c.sha for c in found} or len(found) == 3:
            continue
        c = commit_info(repo, sha)
        load(repo, c, prs, path=path)
        if not c.pr and (m := merge_of(repo, sha)) and m in prs:
            c.pr = {**prs[m], "title": _scrub(prs[m].get("title", "")), "body": _scrub(prs[m].get("body") or "")}
        c.body = f"(found via {path})\n{c.body}".strip()
        found.append(c)
    if not found:
        return Report(NOT_FOUND, [], notes)
    budget(found, notes)
    text = chat(cfg.llm, WHY_SYSTEM, f"Project: {project}\nQuestion: when and why was this added: {topic}\n\n{render(found)}")
    return Report(text, found, notes + omission_notes(found))
