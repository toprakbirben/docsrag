from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from docsrag import chatlog, config, db, generate, history, route

STATIC = Path(__file__).parent / "static"
TITLE_CHARS = 60

app = FastAPI()
app.mount("/static", StaticFiles(directory=STATIC), name="static")


class ChatIn(BaseModel):
    message: str
    conversation_id: int | None = None
    config: str = "baseline"


def _cfg(name: str) -> config.PipelineConfig:
    try:
        return config.get(name)
    except ValueError as e:
        raise HTTPException(400, str(e))


def _title(message: str) -> str:
    text = " ".join(message.split())
    return text if len(text) <= TITLE_CHARS else text[:TITLE_CHARS - 1].rstrip() + "…"


def _report(rep: history.Report) -> tuple[str, dict]:
    return rep.text, {"notes": rep.notes,
                      "commits": [{"sha": c.sha, "date": c.date, "subject": c.subject,
                                   "pr_number": c.pr["number"] if c.pr else None} for c in rep.commits]}


def _answer(conn, r: route.Route, projects: list[str], cfg: config.PipelineConfig) -> tuple[str, dict]:
    if r.mode == "clarify":
        return f"Which project do you mean? I know: {', '.join(projects)}", {}
    if r.mode == "changes":
        return _report(history.changes(r.project, r.since, r.until, cfg))
    if r.mode == "why":
        return _report(history.why(conn, r.project, r.topic, cfg))
    answer = generate.ask(conn, r.question, cfg, r.project)
    by_id = {h.chunk_id: h for h in answer.hits}
    return answer.text, {"refused": answer.refused,
                         "sources": [{"chunk_id": cid, "document_id": by_id[cid].document_id}
                                     for cid in answer.citations]}


@app.get("/", response_class=HTMLResponse)
def page() -> str:
    return (STATIC / "index.html").read_text()


@app.get("/api/conversations")
def list_conversations() -> list[dict]:
    with db.connect() as conn:
        return chatlog.conversations(conn)


@app.get("/api/conversations/{cid}")
def get_conversation(cid: int) -> dict:
    with db.connect() as conn:
        msgs = chatlog.messages(conn, cid)
        if msgs is None:
            raise HTTPException(404, f"no conversation {cid}")
        return {"id": cid, "title": chatlog.title(conn, cid), "messages": msgs}


# Plain `def`: FastAPI runs it in a threadpool, so a 60s LLM call doesn't block the server.
@app.post("/api/chat")
def chat(body: ChatIn) -> dict:
    cfg = _cfg(body.config)
    with db.connect() as conn:
        cid = body.conversation_id or chatlog.create(conn, _title(body.message))
        prior = chatlog.messages(conn, cid)
        if prior is None:
            raise HTTPException(404, f"no conversation {cid}")
        chatlog.add(conn, cid, "user", body.message, {})
        conn.commit()  # keep the question even if answering crashes
        projects = route.known_projects()
        r = route.route(body.message, prior, projects, cfg)
        try:
            text, payload = _answer(conn, r, projects, cfg)
        except history.HistoryError as e:
            text, payload = str(e), {"error": True}
        payload = {"route": r.to_dict(), "config": cfg.name, **payload}
        chatlog.add(conn, cid, "assistant", text, payload)
    return {"conversation_id": cid, "message": {"role": "assistant", "content": text, "payload": payload}}
