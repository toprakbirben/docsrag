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
    monkeypatch.setattr(history, "chat", lambda m, s, u, options=None: sent.update(system=s, user=u) or "Added: rest days [abc1234]")
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


from stuffrag.retrieve import Hit


def test_introducing_commit_is_the_first_not_the_latest(repo):
    commit(repo, "Touch app", "2026-09-21", {"app.py": "limiter = Limiter(key_func=session_key_func)\nx = 1\n"})
    first = git(repo, "log", "--format=%H", "--grep", "Add app")
    assert history.introducing_commit(repo, "app.py", "limiter = Limiter(key_func=session_key_func)") == first


def test_introducing_line_skips_label_and_comments():
    text = "app/app.py\n\n# a very long comment line that should never be picked\nlimiter = Limiter(key_func=f)\nx = 1"
    assert history.introducing_line(text, "projects:app/app.py") == "limiter = Limiter(key_func=f)"
    assert history.introducing_line("app/a.py\n\nx = 1", "projects:app/a.py") is None


def test_why_explains_from_the_introducing_commit(repo, monkeypatch):
    hits = [Hit(1, "projects:app/__overview__", "overview", 1.0),
            Hit(2, "projects:app/app.py", "app/app.py\n\nlimiter = Limiter(key_func=session_key_func)\n", 0.9)]
    monkeypatch.setattr(history, "retrieve", lambda conn, q, cfg, project: hits)
    monkeypatch.setattr(history, "pr_map", lambda r: ({}, None))
    sent = {}
    monkeypatch.setattr(history, "chat", lambda m, s, u, options=None: sent.update(user=u) or "Added in [x]")
    rep = history.why(None, "app", "rate limiting", PipelineConfig(), root=repo.parent)
    assert [c.subject for c in rep.commits] == ["Add app"]
    assert "app.py" in sent["user"] and "Limiter" in sent["user"]


def test_why_without_introducing_commit_does_not_guess(repo, monkeypatch):
    monkeypatch.setattr(history, "retrieve", lambda conn, q, cfg, project: [Hit(1, "projects:app/__overview__", "o", 1.0)])
    monkeypatch.setattr(history, "chat", lambda *a: pytest.fail("LLM must not be asked to guess"))
    rep = history.why(None, "app", "anything", PipelineConfig(), root=repo.parent)
    assert rep.commits == [] and "not found" in rep.text.lower()


def test_truncated_commit_still_lists_every_changed_file(repo, monkeypatch):
    # Real run: PR #123's TacticsPage.tsx fell past the cap and the model claimed the
    # "tactic" part of the message wasn't in the diff. The full file list must always be visible.
    monkeypatch.setattr(history, "COMMIT_CAP", 1000)
    monkeypatch.setattr(history, "TOTAL_CAP", 2000)
    commit(repo, "Tactics", "2026-09-21", {"a.py": "".join(f"v{i} = {i}\n" for i in range(400)),
                                           "b.tsx": "export const tactic = 1\n"})
    cs = history.commits(repo, "2026-09-21")
    for c in cs:
        history.load(repo, c, {})
    history.budget(cs, [])
    assert "b.tsx" not in cs[0].diff and "b.tsx" in history.render(cs)
    assert "truncated" in history.CHANGES_SYSTEM and "truncated" in history.WHY_SYSTEM


def test_single_commit_gets_the_whole_budget(repo):
    # One PR asked about on its own should be shown in full, not cut at the per-commit floor.
    commit(repo, "Big", "2026-09-21", {"big.py": "".join(f"value_{i} = {i}\n" for i in range(800))})
    cs = history.commits(repo, "2026-09-21")
    history.load(repo, cs[0], {})
    history.budget(cs, [])
    assert len(cs[0].diff) > history.COMMIT_CAP and "truncated" not in cs[0].diff


def test_invented_links_are_stripped_but_citations_kept():
    # Real run: the model wrapped citations in made-up github.com/your-repo URLs.
    text = "Added in [697b2ea](https://github.com/your-repo/fuutball/commit/697b2ea) on 2026-02-28."
    assert history.strip_invented_links(text, "context mentions nothing") == "Added in [697b2ea] on 2026-02-28."


def test_links_present_in_the_source_survive():
    text = "See [docs](https://slowapi.readthedocs.io) for limits."
    assert history.strip_invented_links(text, "PR body: https://slowapi.readthedocs.io") == text


def test_history_calls_cap_output_and_penalise_repetition(repo, monkeypatch):
    # Real run on mtt: at temperature 0 the model looped on the same bullets until the 600s timeout.
    seen = {}
    monkeypatch.setattr(history, "pr_map", lambda r: ({}, None))
    monkeypatch.setattr(history, "chat", lambda m, s, u, options=None: seen.update(options or {}) or "ok")
    history.changes("app", "2026-09-05", None, PipelineConfig(), root=repo.parent)
    assert seen["num_predict"] <= 2000 and seen["repeat_penalty"] > 1.1
