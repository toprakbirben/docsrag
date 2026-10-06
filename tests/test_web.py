from contextlib import nullcontext

import pytest
from fastapi.testclient import TestClient

from docsrag import chatlog, db, generate, history, route, web
from docsrag.generate import Answer
from docsrag.history import Commit, Report
from docsrag.retrieve import Hit

HITS = [Hit(12, "fastapi:a.md", "yield deps", 0.9), Hit(14, "fastapi:b.md", "cleanup", 0.8)]
PROJECTS = ["docsrag", "mtt"]


class FakeConn:
    def commit(self): ...


class FakeLog:
    """In-memory stand-in for the conversations/messages tables."""

    def __init__(self):
        self.convs, self.msgs = {}, {}

    def create(self, conn, title):
        cid = len(self.convs) + 1
        self.convs[cid], self.msgs[cid] = title, []
        return cid

    def messages(self, conn, cid):
        return list(self.msgs[cid]) if cid in self.msgs else None

    def add(self, conn, cid, role, content, payload):
        self.msgs[cid].append({"role": role, "content": content, "payload": payload})


@pytest.fixture
def log(monkeypatch):
    fake = FakeLog()
    for name in ("create", "messages", "add"):
        monkeypatch.setattr(chatlog, name, getattr(fake, name))
    monkeypatch.setattr(db, "connect", lambda: nullcontext(FakeConn()))
    monkeypatch.setattr(route, "known_projects", lambda: PROJECTS)
    return fake


@pytest.fixture
def client(log):
    return TestClient(web.app)


def routes_to(monkeypatch, r: route.Route, seen: dict | None = None):
    def fake(message, prior, projects, cfg):
        if seen is not None:
            seen.update(message=message, prior=prior, cfg=cfg.name)
        return r
    monkeypatch.setattr(route, "route", fake)


def test_new_chat_saves_question_and_reply_under_a_title(client, log, monkeypatch):
    # Reopening a chat from the sidebar must show both sides of the conversation.
    routes_to(monkeypatch, route.Route("ask", "How do deps work?"))
    monkeypatch.setattr(generate, "ask", lambda conn, q, cfg, project: Answer("Use yield [c12].", [12], HITS))
    body = client.post("/api/chat", json={"message": "How do deps work?"}).json()
    cid = body["conversation_id"]
    assert log.convs[cid] == "How do deps work?"
    assert [m["role"] for m in log.msgs[cid]] == ["user", "assistant"]
    assert body["message"]["content"] == "Use yield [c12]."


def test_long_first_message_is_cut_for_the_title(client, log, monkeypatch):
    routes_to(monkeypatch, route.Route("ask", "q"))
    monkeypatch.setattr(generate, "ask", lambda *a: Answer("a", [], []))
    cid = client.post("/api/chat", json={"message": "word " * 40}).json()["conversation_id"]
    assert len(log.convs[cid]) <= 60 and log.convs[cid].endswith("…")


def test_ask_reply_lists_only_cited_sources_and_shows_the_route(client, monkeypatch):
    # The page must show what the answer relies on, not every passage retrieval returned,
    # and what the router decided, so a misroute is visible.
    routes_to(monkeypatch, route.Route("ask", "deps in mtt?", project="mtt"))
    monkeypatch.setattr(generate, "ask", lambda conn, q, cfg, project: Answer("Use yield [c12].", [12], HITS))
    payload = client.post("/api/chat", json={"message": "deps?"}).json()["message"]["payload"]
    assert payload["sources"] == [{"chunk_id": 12, "document_id": "fastapi:a.md"}]
    assert payload["route"]["mode"] == "ask" and payload["route"]["project"] == "mtt"
    assert payload["refused"] is False


def test_ask_uses_the_standalone_rewrite_not_the_raw_follow_up(client, monkeypatch):
    # "and in mtt?" alone retrieves nothing useful; the router's rewrite carries the context.
    routes_to(monkeypatch, route.Route("ask", "How are background jobs run in mtt?", project="mtt"))
    seen = {}
    monkeypatch.setattr(generate, "ask",
                        lambda conn, q, cfg, project: seen.update(q=q, p=project) or Answer("a", [], []))
    client.post("/api/chat", json={"message": "and in mtt?"})
    assert seen == {"q": "How are background jobs run in mtt?", "p": "mtt"}


def test_follow_up_passes_earlier_turns_to_the_router(client, log, monkeypatch):
    routes_to(monkeypatch, route.Route("ask", "q"))
    monkeypatch.setattr(generate, "ask", lambda *a: Answer("first answer", [], []))
    cid = client.post("/api/chat", json={"message": "first"}).json()["conversation_id"]
    seen = {}
    routes_to(monkeypatch, route.Route("ask", "q"), seen)
    client.post("/api/chat", json={"conversation_id": cid, "message": "second", "config": "hybrid"})
    assert [m["content"] for m in seen["prior"]] == ["first", "first answer"]
    assert seen["message"] == "second" and seen["cfg"] == "hybrid"


def test_changes_route_runs_history_with_its_range(client, monkeypatch):
    routes_to(monkeypatch, route.Route("changes", "q", project="mtt", since="2 weeks ago", until="2026-10-02"))
    seen = {}
    commit = Commit("abcdef1234", "2026-10-01", "Add x", "", pr={"number": 7, "title": "t", "body": ""})

    def fake(project, since, until, cfg):
        seen.update(project=project, since=since, until=until)
        return Report("Added x [abcdef1]", [commit], ["note"])
    monkeypatch.setattr(history, "changes", fake)
    msg = client.post("/api/chat", json={"message": "what changed?"}).json()["message"]
    assert seen == {"project": "mtt", "since": "2 weeks ago", "until": "2026-10-02"}
    assert msg["content"] == "Added x [abcdef1]"
    assert msg["payload"]["commits"] == [{"sha": "abcdef1234", "date": "2026-10-01", "subject": "Add x", "pr_number": 7}]
    assert msg["payload"]["notes"] == ["note"]


def test_why_route_passes_project_and_topic(client, monkeypatch):
    routes_to(monkeypatch, route.Route("why", "q", project="mtt", topic="background jobs"))
    seen = {}
    monkeypatch.setattr(history, "why",
                        lambda conn, project, topic, cfg: seen.update(p=project, t=topic) or Report("r", [], []))
    client.post("/api/chat", json={"message": "why background jobs?"})
    assert seen == {"p": "mtt", "t": "background jobs"}


def test_missing_project_gets_a_question_back_listing_known_projects(client, monkeypatch):
    routes_to(monkeypatch, route.Route("clarify", "q"))
    msg = client.post("/api/chat", json={"message": "what changed last week?"}).json()["message"]
    assert "Which project" in msg["content"] and "docsrag, mtt" in msg["content"]


def test_history_error_becomes_a_readable_reply_and_is_kept(client, log, monkeypatch):
    # The reason (e.g. no git history) must reach the user in the chat, not as a crash.
    routes_to(monkeypatch, route.Route("why", "q", project="mtt", topic="t"))

    def boom(*a, **k):
        raise history.HistoryError("mtt has no git history")
    monkeypatch.setattr(history, "why", boom)
    body = client.post("/api/chat", json={"message": "why t?"}).json()
    assert body["message"]["content"] == "mtt has no git history" and body["message"]["payload"]["error"] is True
    assert len(log.msgs[body["conversation_id"]]) == 2


def test_unknown_conversation_is_a_404(client):
    assert client.post("/api/chat", json={"conversation_id": 99, "message": "x"}).status_code == 404
    assert client.get("/api/conversations/99").status_code == 404


def test_unknown_config_is_a_readable_400(client):
    r = client.post("/api/chat", json={"message": "q", "config": "nope"})
    assert r.status_code == 400 and "unknown config" in r.json()["detail"]


def test_page_has_a_single_message_box_and_a_sidebar(client):
    r = client.get("/")
    assert r.status_code == 200 and "text/html" in r.headers["content-type"]
    assert 'id="sidebar"' in r.text and r.text.count("<textarea") == 1
    assert client.get("/static/marked.min.js").status_code == 200
