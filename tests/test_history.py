import os
import subprocess
from pathlib import Path

import pytest

from stuffrag import history

AWS = "AKIA" + "ABCDEFGHIJKLMNOP"


def git(repo: Path, *args: str, date: str | None = None) -> str:
    env = {**os.environ}
    if date:
        env |= {"GIT_AUTHOR_DATE": f"{date}T12:00:00", "GIT_COMMITTER_DATE": f"{date}T12:00:00"}
    return subprocess.run(["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t", *args],
                          check=True, capture_output=True, text=True, env=env).stdout.strip()


def commit(repo: Path, msg: str, date: str, files: dict[str, str | None]) -> str:
    for p, t in files.items():
        f = repo / p
        if t is None:
            f.unlink()
        else:
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text(t)
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", msg, date=date)
    return git(repo, "rev-parse", "HEAD")


@pytest.fixture
def repo(tmp_path):
    """main: c1 (09-01) -> merge of feat (09-12) -> c2 (09-20); feat: f1 (09-10), f2 (09-11)."""
    r = tmp_path / "app"
    r.mkdir()
    git(r, "init", "-q", "-b", "main")
    commit(r, "Add app", "2026-09-01", {"app.py": "limiter = Limiter(key_func=session_key_func)\n"})
    git(r, "checkout", "-q", "-b", "feat")
    commit(r, "Add stamina", "2026-09-10", {"stamina.py": "def recover(p):\n    return p.stamina + 10\n"})
    commit(r, "Tune stamina", "2026-09-11", {"stamina.py": "def recover(p):\n    return p.stamina + 15\n"})
    git(r, "checkout", "-q", "main")
    git(r, "merge", "-q", "--no-ff", "feat", "-m", "Merge pull request #7 from me/feat", date="2026-09-12")
    commit(r, "Fix typo", "2026-09-20", {"README.md": "# App\n"})
    return r


def test_range_is_honoured_and_a_merged_branch_is_one_unit(repo):
    # A PR is what I think of as "one change"; its branch commits must not be listed twice.
    assert [c.subject for c in history.commits(repo, "2026-09-05", "2026-09-15")] == \
        ["Merge pull request #7 from me/feat"]
    assert [c.subject for c in history.commits(repo, "2026-09-05")] == \
        ["Fix typo", "Merge pull request #7 from me/feat"]


def test_merge_diff_covers_the_whole_branch(repo):
    (merge,) = history.commits(repo, "2026-09-05", "2026-09-15")
    history.load(repo, merge, {})
    assert "stamina.py" in merge.diff and "stamina + 15" in merge.diff and "stamina + 10" not in merge.diff


def test_secret_added_then_removed_is_omitted_from_both(repo):
    # History can hold a key the current files no longer have; it must never reach the LLM.
    commit(repo, "Add config", "2026-09-21", {"config.py": f"KEY = '{AWS}'\n", ".env": "X=1\n"})
    commit(repo, "Remove key", "2026-09-22", {"config.py": "KEY = None\n"})
    cs = history.commits(repo, "2026-09-21")
    for c in cs:
        history.load(repo, c, {})
        assert AWS not in c.diff
    assert {p for c in cs for p in c.secret_files} == {"config.py", ".env"}


def test_secret_in_commit_message_is_replaced(repo):
    commit(repo, f"Rotate key {AWS}", "2026-09-21", {"a.py": "x = 1\n"})
    (c,) = history.commits(repo, "2026-09-21")
    history.load(repo, c, {})
    assert c.subject == history.OMITTED


def test_hidden_files_are_counted_not_shown(repo):
    commit(repo, "Docs and lock", "2026-09-21", {"docs/plan.md": "plan\n", "package-lock.json": "{}\n",
                                                 "b.py": "y = 2\n"})
    (c,) = history.commits(repo, "2026-09-21")
    history.load(repo, c, {})
    assert "b.py" in c.diff and "plan.md" not in c.diff and c.hidden_files == 2


def test_budget_reduces_largest_commit_and_reports_it(repo, monkeypatch):
    monkeypatch.setattr(history, "TOTAL_CAP", 300)
    commit(repo, "Big", "2026-09-21", {"big.py": "".join(f"v{i} = {i}\n" for i in range(200))})
    cs = history.commits(repo, "2026-09-05")
    for c in cs:
        history.load(repo, c, {})
    notes: list[str] = []
    history.budget(cs, notes)
    big = next(c for c in cs if c.subject == "Big")
    assert big.reduced and big.diff == big.stat
    assert any(big.sha[:7] in n for n in notes)


def test_non_utf8_diff_does_not_crash(repo):
    (repo / "latin.txt").write_bytes("caf\xe9\n".encode("latin-1"))
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "Latin", date="2026-09-21")
    (c,) = history.commits(repo, "2026-09-21")
    history.load(repo, c, {})
    assert "latin.txt" in c.diff


def test_repo_for_rejects_non_git_and_skipped(tmp_path):
    (tmp_path / "point cloud").mkdir()
    with pytest.raises(history.HistoryError, match="no git history"):
        history.repo_for("point cloud", tmp_path)
    with pytest.raises(history.HistoryError, match="unknown project"):
        history.repo_for("LLaVA", tmp_path)


import json


def test_pr_text_attaches_to_its_merge_commit(repo, monkeypatch):
    git(repo, "remote", "add", "origin", "https://github.com/me/app.git")
    merge = history.commits(repo, "2026-09-05", "2026-09-15")[0].sha
    out = json.dumps([{"number": 7, "title": "Stamina", "body": "Rest days recover stamina",
                       "mergeCommit": {"oid": merge}}])
    calls = []
    monkeypatch.setattr(history, "_gh", lambda *a: calls.append(a) or subprocess.CompletedProcess(a, 0, out, ""))
    prs, warning = history.pr_map(repo)
    assert warning is None and prs[merge]["number"] == 7
    assert "me/app" in calls[0]


def test_gh_failure_warns_and_continues(repo, monkeypatch):
    git(repo, "remote", "add", "origin", "git@github.com:me/app.git")

    def boom(*a):
        raise FileNotFoundError("gh")

    monkeypatch.setattr(history, "_gh", boom)
    prs, warning = history.pr_map(repo)
    assert prs == {} and "gh" in warning


def test_no_github_origin_warns(repo):
    prs, warning = history.pr_map(repo)
    assert prs == {} and "commit messages only" in warning


def test_merge_of_finds_the_merge_that_brought_a_branch_commit_in(repo):
    # `why` finds the branch commit that introduced a line; its PR text hangs off the merge.
    f1 = git(repo, "log", "--all", "--format=%H", "--grep", "Add stamina")
    merge = history.commits(repo, "2026-09-05", "2026-09-15")[0].sha
    assert history.merge_of(repo, f1) == merge
    # Direct mainline commits must not be attributed to a later PR merge.
    assert history.merge_of(repo, git(repo, "log", "--format=%H", "--grep", "Add app")) is None
    assert history.merge_of(repo, git(repo, "rev-parse", "HEAD")) is None


from typer.testing import CliRunner

from stuffrag.config import PipelineConfig


def test_changes_prompt_has_messages_diffs_and_never_the_secret(repo, monkeypatch):
    commit(repo, "Add config", "2026-09-21", {"config.py": f"KEY = '{AWS}'\n", "b.py": "rest_days = 2\n"})
    sent = {}
    monkeypatch.setattr(history, "pr_map", lambda r: ({}, "no origin remote; using commit messages only"))
    monkeypatch.setattr(history, "chat", lambda m, s, u: sent.update(system=s, user=u) or "Added: rest days [abc1234]")
    rep = history.changes("app", "2026-09-05", None, PipelineConfig(), root=repo.parent)
    assert "Merge pull request #7" in sent["user"] and "rest_days" in sent["user"]
    assert AWS not in sent["user"]
    assert "Message vs diff" in sent["system"]
    assert any("config.py" in n for n in rep.notes) and any("commit messages only" in n for n in rep.notes)


def test_cli_changes_empty_range_exits_1(repo, monkeypatch):
    from stuffrag import cli
    monkeypatch.setattr(history, "ROOT", repo.parent)
    out = CliRunner().invoke(cli.app, ["changes", "app", "--since", "2030-01-01"])
    assert out.exit_code == 1 and "no commits" in out.output
