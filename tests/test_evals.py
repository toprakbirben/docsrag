from stuffrag import evals
from stuffrag.retrieve import Hit


def h(cid, doc):
    return Hit(cid, doc, "", 0.0)


def test_doc_ranking_dedupes_chunks_of_the_same_doc():
    # Three chunks from one doc must not push the second doc down to rank 4.
    hits = [h(1, "a"), h(2, "a"), h(3, "a"), h(4, "b")]
    assert evals.doc_ranking(hits) == ["a", "b"]


def test_recall_and_mrr():
    ranked = ["x", "gold2", "y", "gold1"]
    assert evals.recall_at(ranked, ["gold1", "gold2"], 3) == 0.5
    assert evals.mrr_at(ranked, ["gold1", "gold2"], 10) == 0.5
    assert evals.mrr_at(ranked, ["zzz"], 10) == 0.0


def test_keyfact_is_case_insensitive_fraction():
    assert evals.keyfact_score("Use YIELD inside Depends", ["yield", "depends", "finally"]) == 2 / 3


def test_percentile_nearest_rank():
    assert evals.percentile([1, 2, 3, 4, 100], 50) == 3
    assert evals.percentile([1, 2, 3, 4, 100], 95) == 100


def test_aggregate_scores_refusal_questions_only_on_refusal():
    results = [
        {"should_refuse": False, "refused": False, "recall@5": 1.0, "mrr@10": 1.0,
         "keyfact": 1.0, "judge": True, "latency_s": 1.0},
        {"should_refuse": False, "refused": True, "recall@5": 0.0, "mrr@10": 0.0,
         "keyfact": 0.0, "judge": False, "latency_s": 2.0},
        {"should_refuse": True, "refused": True, "recall@5": None, "mrr@10": None,
         "keyfact": None, "judge": None, "latency_s": 3.0},
    ]
    m = evals.aggregate(results)
    assert m["recall@5"] == 0.5 and m["answer_correct"] == 0.5
    assert m["refusal_accuracy"] == 2 / 3  # the false refusal on q2 counts against it
    assert m["n"] == 3 and m["judge_errors"] == 0


def test_aggregate_counts_judge_errors_instead_of_hiding_them():
    r = {"should_refuse": False, "refused": False, "recall@5": 1.0, "mrr@10": 1.0,
         "keyfact": 1.0, "judge": None, "latency_s": 1.0}
    assert evals.aggregate([r])["judge_errors"] == 1


def test_compare_prints_deltas():
    a = {"id": "A", "metrics": {"recall@5": 0.5, "latency_p50": 2.0}}
    b = {"id": "B", "metrics": {"recall@5": 0.75, "latency_p50": 3.0}}
    assert evals.compare(a, b) == [
        "metric                 A        B    delta",
        "latency_p50        2.000    3.000   +1.000",
        "recall@5           0.500    0.750   +0.250",
    ]


def test_run_warms_up_reranker_before_the_timed_loop(tmp_path, monkeypatch):
    # cfg.rerank=True must load the model once before evaluate() is called for any
    # question, so the first question's latency doesn't include model load time.
    from stuffrag.config import PipelineConfig
    from stuffrag import rerank as rerank_mod

    calls = []
    monkeypatch.setattr(evals, "index", lambda conn, cfg: None)
    monkeypatch.setattr(evals, "git_sha", lambda: "deadbee")
    monkeypatch.setattr(evals, "RUNS", tmp_path)

    def fake_evaluate(conn, q, cfg):
        calls.append("evaluate")
        return {"id": q.id, "should_refuse": False, "refused": False, "recall@5": 1.0,
                "mrr@10": 1.0, "keyfact": 1.0, "judge": True, "latency_s": 0.0}

    monkeypatch.setattr(evals, "evaluate", fake_evaluate)

    def fake_model():
        calls.append("warmup")

    monkeypatch.setattr(rerank_mod, "_model", fake_model)

    class FakeConn:
        def execute(self, *a, **k):
            pass

        def commit(self):
            pass

    q = evals.Question(id="q1", question="?", expected_answer=[], gold_sources=[], source_type="fastapi")
    evals.run(FakeConn(), PipelineConfig(name="hybrid_rerank", rerank=True), [q])
    assert calls[0] == "warmup"
    assert calls[1] == "evaluate"


def test_run_skips_warmup_when_rerank_is_off(tmp_path, monkeypatch):
    from stuffrag.config import PipelineConfig
    from stuffrag import rerank as rerank_mod

    monkeypatch.setattr(evals, "index", lambda conn, cfg: None)
    monkeypatch.setattr(evals, "git_sha", lambda: "deadbee")
    monkeypatch.setattr(evals, "RUNS", tmp_path)
    monkeypatch.setattr(evals, "evaluate", lambda conn, q, cfg: {
        "id": q.id, "should_refuse": False, "refused": False, "recall@5": 1.0,
        "mrr@10": 1.0, "keyfact": 1.0, "judge": True, "latency_s": 0.0})

    def boom():
        raise AssertionError("must not warm up the reranker when cfg.rerank is False")

    monkeypatch.setattr(rerank_mod, "_model", boom)

    class FakeConn:
        def execute(self, *a, **k):
            pass

        def commit(self):
            pass

    q = evals.Question(id="q1", question="?", expected_answer=[], gold_sources=[], source_type="fastapi")
    evals.run(FakeConn(), PipelineConfig(name="baseline", rerank=False), [q])


def test_evaluate_records_context_chars_from_top_k_hits(monkeypatch):
    from stuffrag.config import PipelineConfig
    from stuffrag.generate import Answer
    from stuffrag.retrieve import Hit

    hits = [Hit(1, "docA", "aaaa", 0.0), Hit(2, "docB", "bb", 0.0), Hit(3, "docC", "ccccccc", 0.0)]
    monkeypatch.setattr(evals, "retrieve", lambda conn, question, cfg: hits)
    monkeypatch.setattr(
        evals, "generate",
        lambda question, hits, cfg: Answer(text="ans", citations=[], hits=hits),
    )
    cfg = PipelineConfig(top_k=2)
    q = evals.Question(id="q1", question="?", expected_answer=["x"], gold_sources=["docA"],
                        source_type="fastapi")
    r = evals.evaluate(object(), q, cfg)
    # only the top_k=2 hits actually go into the prompt; the 3rd must not count.
    assert r["context_chars"] == len(hits[0].text) + len(hits[1].text)
