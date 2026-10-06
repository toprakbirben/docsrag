import re
from dataclasses import dataclass

import httpx
import psycopg

from docsrag.config import OLLAMA_URL, PipelineConfig
from docsrag.retrieve import Hit, retrieve

REFUSAL = "Not found in my sources."
NUM_CTX = 16384  # Ollama's default window would silently truncate top_k passages
SYSTEM = f"""You answer questions using ONLY the context passages below.
Cite every claim with its passage id in square brackets, e.g. [c12].
If the passages do not contain the answer, reply exactly: {REFUSAL}"""


@dataclass
class Answer:
    text: str
    citations: list[int]
    hits: list[Hit]

    @property
    def refused(self) -> bool:
        return self.text.strip().lower().startswith(REFUSAL.lower().rstrip("."))


def parse_citations(text: str, hits: list[Hit]) -> list[int]:
    allowed = {h.chunk_id for h in hits}
    out: list[int] = []
    for group in re.findall(r"\[([^\]]+)\]", text):
        for cid in map(int, re.findall(r"c(\d+)", group)):
            if cid in allowed and cid not in out:
                out.append(cid)
    return out


def chat(model: str, system: str, user: str, json_mode: bool = False, options: dict | None = None) -> str:
    payload = {
        "model": model,
        "stream": False,
        "think": False,
        "options": {"temperature": 0, "num_ctx": NUM_CTX, **(options or {})},
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
    }
    if json_mode:
        payload["format"] = "json"
    r = httpx.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=600)
    r.raise_for_status()
    body = r.json()
    text = body["message"]["content"].strip()
    if body.get("done_reason") == "length":  # hit num_predict/num_ctx: never pass a partial answer off as whole
        text += "\n\n(answer cut off at the output limit)"
    return text


def generate(question: str, hits: list[Hit], cfg: PipelineConfig) -> Answer:
    if not hits:
        return Answer(REFUSAL, [], [])
    context = "\n\n".join(f"[c{h.chunk_id}] ({h.document_id})\n{h.text}" for h in hits)
    text = chat(cfg.llm, SYSTEM, f"Context:\n{context}\n\nQuestion: {question}")
    return Answer(text, parse_citations(text, hits), hits)


def ask(conn: psycopg.Connection, question: str, cfg: PipelineConfig, project: str | None = None) -> Answer:
    return generate(question, retrieve(conn, question, cfg, project), cfg)
