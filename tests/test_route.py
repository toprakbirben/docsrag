import json
from datetime import date

from docsrag import generate, route
from docsrag.config import PipelineConfig

PROJECTS = ["docsrag", "fuutball", "mtt"]


def parse(d, message="msg"):
    return route.parse(json.dumps(d) if not isinstance(d, str) else d, message, PROJECTS)


def test_broken_json_falls_back_to_plain_ask_with_the_original_message():
    # A router failure must never lose the user's question; worst case it's answered as a normal search.
    r = parse("not json {", "how do deps work?")
    assert (r.mode, r.question, r.project) == ("ask", "how do deps work?", None)


def test_unknown_mode_falls_back_to_ask():
    assert parse({"mode": "delete_everything", "question": "q"}).mode == "ask"


def test_invented_project_is_dropped():
    # The model may guess a name; searching a project that doesn't exist would return nothing silently.
    r = parse({"mode": "ask", "question": "q", "project": "zzz"})
    assert r.project is None


def test_project_match_is_case_insensitive_and_uses_the_real_name():
    assert parse({"mode": "ask", "question": "q", "project": "MTT"}).project == "mtt"


def test_history_modes_without_a_project_ask_which_project():
    # changes/why need a git repo; guessing one would explain the wrong project's history.
    assert parse({"mode": "changes", "project": None, "since": "1 week ago"}).mode == "clarify"
    assert parse({"mode": "why", "project": "zzz", "topic": "auth"}).mode == "clarify"


def test_changes_defaults_since_and_why_defaults_topic_to_the_question():
    assert parse({"mode": "changes", "project": "mtt"}).since == "1 week ago"
    assert parse({"mode": "why", "project": "mtt", "question": "why redis?"}).topic == "why redis?"


def test_until_equal_to_since_is_dropped():
    # Seen live: "what changed last week?" -> since=until="last week", an empty range with no commits.
    r = parse({"mode": "changes", "project": "mtt", "since": "last week", "until": "Last Week"})
    assert (r.since, r.until) == ("last week", None)


def test_empty_question_falls_back_to_the_message():
    assert parse({"mode": "ask", "question": "  "}, "original").question == "original"


def test_router_sees_earlier_turns_and_known_projects(monkeypatch):
    # Follow-ups like "and in mtt?" only work if the router sees what came before.
    sent = {}
    monkeypatch.setattr(generate, "chat", lambda model, system, user, json_mode=False, options=None:
                        sent.update(system=system, user=user, json_mode=json_mode) or '{"mode": "ask", "question": "q"}')
    prior = [{"role": "user", "content": "what changed in fuutball last week?", "payload": {}},
             {"role": "assistant", "content": "Added X",
              "payload": {"route": {"mode": "changes", "project": "fuutball", "since": "1 week ago"}}}]
    route.route("and in mtt?", prior, PROJECTS, PipelineConfig())
    assert "what changed in fuutball last week?" in sent["user"] and "and in mtt?" in sent["user"]
    assert "mode=changes" in sent["user"] and "project=fuutball" in sent["user"]
    assert "mtt" in sent["system"] and sent["json_mode"] is True


def test_router_is_told_todays_date(monkeypatch):
    # Without it, "this past week" became 2026-09-01..07 a month late, and changes found no commits.
    sent = {}
    monkeypatch.setattr(generate, "chat", lambda model, system, user, json_mode=False, options=None:
                        sent.update(system=system) or "{}")
    route.route("what changed this week?", [], PROJECTS, PipelineConfig(), today=date(2026, 10, 6))
    assert "2026-10-06" in sent["system"]


def test_known_projects_skips_third_party_and_hidden_dirs(tmp_path):
    for name in ["mine", "LLaVA", ".cache"]:
        (tmp_path / name).mkdir()
    (tmp_path / "file.txt").write_text("x")
    assert route.known_projects(tmp_path) == ["mine"]
