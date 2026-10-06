import json
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

from docsrag import generate
from docsrag.config import PipelineConfig
from docsrag.ingest.projects import ROOT, SKIP_PROJECTS

MODES = {"ask", "changes", "why"}
PRIOR_TURNS = 6  # enough to resolve "and in mtt?" without flooding the prompt with old answers
PRIOR_CHARS = 400

SYSTEM = """You route one chat message about the user's software projects and the FastAPI docs.
Reply with JSON only:
{{"mode": "ask" | "changes" | "why", "question": string, "project": string | null,
  "since": string | null, "until": string | null, "topic": string | null}}
- changes: what was added, removed or changed in a project over a period. since/until are git dates.
  Prefer relative ones such as "1 week ago", "3 days ago" or "yesterday"; until is null unless the
  period has a stated end.
- why: when or why something was added to a project. topic is that thing, in a few words.
- ask: everything else (how something works, where something is, FastAPI questions).
- question: the new message rewritten to stand on its own, resolving references to earlier turns.
- project: one of the known projects named or implied by the conversation, otherwise null.
Today is {today}.
Known projects: {projects}"""


@dataclass
class Route:
    mode: str  # ask | changes | why | clarify (a history question with no known project)
    question: str
    project: str | None = None
    since: str | None = None
    until: str | None = None
    topic: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def known_projects(root: Path = ROOT) -> list[str]:
    return sorted(p.name for p in root.iterdir()
                  if p.is_dir() and not p.name.startswith(".") and p.name not in SKIP_PROJECTS)


def _str(v) -> str | None:
    return v.strip() if isinstance(v, str) and v.strip() else None


def parse(raw: str, message: str, projects: list[str]) -> Route:
    """Validate the model's routing JSON in code; anything unusable degrades to a plain ask."""
    try:
        d = json.loads(raw)
    except json.JSONDecodeError:
        d = None
    if not isinstance(d, dict):
        return Route("ask", message)
    mode = d.get("mode") if d.get("mode") in MODES else "ask"
    question = _str(d.get("question")) or message
    by_lower = {p.lower(): p for p in projects}
    project = by_lower.get((_str(d.get("project")) or "").lower())
    r = Route(mode, question, project, _str(d.get("since")), _str(d.get("until")), _str(d.get("topic")))
    if mode in ("changes", "why") and not project:
        r.mode = "clarify"
    if mode == "changes":
        r.since = r.since or "1 week ago"
        if r.until and r.until.lower() == r.since.lower():  # echoed, not a stated end: empty range
            r.until = None
    if mode == "why":
        r.topic = r.topic or question
    return r


def _turn(m: dict) -> str:
    rt = m["payload"].get("route")
    tag = f" [{' '.join(f'{k}={v}' for k, v in rt.items() if v and k != 'question')}]" if rt else ""
    return f"{m['role']}{tag}: {m['content'][:PRIOR_CHARS]}"


def route(message: str, prior: list[dict], projects: list[str], cfg: PipelineConfig,
          today: date | None = None) -> Route:
    convo = "\n".join(_turn(m) for m in prior[-PRIOR_TURNS:]) or "(none)"
    system = SYSTEM.format(projects=", ".join(projects), today=(today or date.today()).isoformat())
    raw = generate.chat(cfg.llm, system,
                        f"Conversation so far:\n{convo}\n\nNew message: {message}", json_mode=True)
    return parse(raw, message, projects)
