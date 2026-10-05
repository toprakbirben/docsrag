import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from stuffrag.ingest.projects import ROOT, SKIP_PROJECTS, _allowed, is_secret

COMMIT_CAP = 6_000   # chars of filtered diff per commit
TOTAL_CAP = 40_000   # chars of diff across all commits sent to the LLM
OMITTED = "(omitted: matched a secret pattern)"
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
    diff = "".join(kept)
    c.diff = diff if len(diff) <= COMMIT_CAP else diff[:COMMIT_CAP] + "\n... (diff truncated)\n"
    c.subject, c.body = _scrub(c.subject), _scrub(c.body)
    pr = prs.get(c.sha)
    c.pr = pr and {**pr, "title": _scrub(pr.get("title", "")), "body": _scrub(pr.get("body") or "")}


def budget(commits: list[Commit], notes: list[str]) -> None:
    """Reduce the largest diffs to --stat until the total fits; say which ones."""
    while sum(len(c.diff) for c in commits) > TOTAL_CAP:
        full = [c for c in commits if not c.reduced]
        if not full:
            break
        big = max(full, key=lambda c: len(c.diff))
        big.diff, big.reduced = big.stat, True
        notes.append(f"{big.sha[:7]} too large: shown as file stats only")
