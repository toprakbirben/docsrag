import json
import math
import subprocess
import time
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path

import psycopg
import yaml
from psycopg.types.json import Jsonb

from docsrag import rerank as rerank_mod
from docsrag.config import PipelineConfig
from docsrag.embed import index
from docsrag.generate import chat, generate
from docsrag.retrieve import Hit, retrieve

QUESTIONS = Path("evals/questions.yaml")
LOCAL_QUESTIONS = Path("evals/questions.local.yaml")  # about your own ~/projects; gitignored
RUNS = Path("evals/runs")
JUDGE_SYSTEM = """You grade an answer against expected key facts.
Reply with JSON {"correct": true} if the answer states all key facts and nothing contradicting them,
else {"correct": false}."""


@dataclass
class Question:
    id: str
    question: str
    expected_answer: list[str]
    gold_sources: list[str]
    source_type: str
    should_refuse: bool = False


def load_questions(path: Path = QUESTIONS, local: Path | None = LOCAL_QUESTIONS) -> list[Question]:
    qs = [Question(**q) for q in yaml.safe_load(path.read_text()) or []]
    if local and local.exists():
        qs += [Question(**q) for q in yaml.safe_load(local.read_text()) or []]
    dupes = sorted({q.id for q in qs if sum(o.id == q.id for o in qs) > 1})
    if dupes:
        raise ValueError(f"duplicate question ids: {', '.join(dupes)}")
    return qs


def doc_ranking(hits: list[Hit]) -> list[str]:
    return list(dict.fromkeys(h.document_id for h in hits))


def recall_at(ranked: list[str], gold: list[str], k: int) -> float:
    return len(set(ranked[:k]) & set(gold)) / len(gold)


def mrr_at(ranked: list[str], gold: list[str], k: int) -> float:
    return next((1 / i for i, d in enumerate(ranked[:k], 1) if d in gold), 0.0)


def keyfact_score(answer: str, facts: list[str]) -> float:
    return sum(f.lower() in answer.lower() for f in facts) / len(facts)


def percentile(values: list[float], p: float) -> float:
    s = sorted(values)
    return s[max(0, math.ceil(p / 100 * len(s)) - 1)]


def judge(q: Question, answer: str, cfg: PipelineConfig) -> bool | None:
    user = f"Question: {q.question}\nKey facts: {q.expected_answer}\nAnswer: {answer}"
    try:
        return bool(json.loads(chat(cfg.llm, JUDGE_SYSTEM, user, json_mode=True))["correct"])
    except (json.JSONDecodeError, KeyError, TypeError):
        return None  # counted as judge_errors


def _mean(xs: list) -> float | None:
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def aggregate(results: list[dict]) -> dict:
    answerable = [r for r in results if not r["should_refuse"]]
    lat = [r["latency_s"] for r in results]
    return {
        "n": len(results),
        "recall@5": _mean([r["recall@5"] for r in answerable]),
        "mrr@10": _mean([r["mrr@10"] for r in answerable]),
        "keyfact": _mean([r["keyfact"] for r in answerable]),
        "answer_correct": _mean([r["judge"] for r in answerable]),
        "judge_errors": sum(r["judge"] is None for r in answerable),
        "refusal_accuracy": _mean([r["refused"] == r["should_refuse"] for r in results]),
        "latency_p50": percentile(lat, 50) if lat else None,
        "latency_p95": percentile(lat, 95) if lat else None,
    }


def evaluate(conn: psycopg.Connection, q: Question, cfg: PipelineConfig) -> dict:
    t0 = time.perf_counter()
    hits = retrieve(conn, q.question, replace(cfg, top_k=max(cfg.top_k, 10)))  # 10 for MRR@10
    answer = generate(q.question, hits[: cfg.top_k], cfg)
    latency = time.perf_counter() - t0
    ranked = doc_ranking(hits)
    r = {"id": q.id, "should_refuse": q.should_refuse, "refused": answer.refused,
         "answer": answer.text, "citations": answer.citations, "retrieved": ranked[:10],
         "latency_s": latency, "context_chars": sum(len(h.text) for h in hits[: cfg.top_k]),
         "recall@5": None, "mrr@10": None, "keyfact": None, "judge": None}
    if not q.should_refuse:
        r["recall@5"] = recall_at(ranked, q.gold_sources, 5)
        r["mrr@10"] = mrr_at(ranked, q.gold_sources, 10)
        r["keyfact"] = keyfact_score(answer.text, q.expected_answer)
        r["judge"] = False if answer.refused else judge(q, answer.text, cfg)
    return r


def git_sha() -> str:
    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True).stdout.strip()
    return sha + ("-dirty" if dirty else "")


def run(conn: psycopg.Connection, cfg: PipelineConfig, questions: list[Question]) -> Path:
    index(conn, cfg)  # each ablation is one flag change: make sure its chunks/vectors exist
    if cfg.rerank:
        rerank_mod._model()  # warm up so the first question's latency excludes model load
    results = [evaluate(conn, q, cfg) for q in questions]
    run_id = f"{datetime.now():%Y%m%d-%H%M%S}-{cfg.name}"
    doc = {"id": run_id, "config": cfg.to_dict(), "git_sha": git_sha(),
           "metrics": aggregate(results), "results": results}
    RUNS.mkdir(parents=True, exist_ok=True)
    path = RUNS / f"{run_id}.json"
    path.write_text(json.dumps(doc, indent=2))
    conn.execute(
        "INSERT INTO eval_runs (id, config, git_sha, metrics) VALUES (%s, %s, %s, %s)",
        (run_id, Jsonb(doc["config"]), doc["git_sha"], Jsonb(doc["metrics"])),
    )
    conn.commit()
    return path


def compare(a: dict, b: dict) -> list[str]:
    def fmt(x):
        return f"{x:8.3f}" if isinstance(x, (int, float)) else f"{'-':>8}"

    lines = [f"{'metric':<16}{a['id']:>8} {b['id']:>8}    delta"]
    for m in sorted(set(a["metrics"]) | set(b["metrics"])):
        x, y = a["metrics"].get(m), b["metrics"].get(m)
        delta = f"{y - x:+8.3f}" if isinstance(x, (int, float)) and isinstance(y, (int, float)) else f"{'-':>8}"
        lines.append(f"{m:<16}{fmt(x)} {fmt(y)} {delta}")
    return lines
