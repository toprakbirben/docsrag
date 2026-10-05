from stuffrag import retrieve
from stuffrag.config import PipelineConfig
from stuffrag.retrieve import Hit, rrf


def h(cid):
    return Hit(cid, f"d{cid}", "", 0.0)


def test_rrf_rewards_agreement_between_retrievers():
    # Chunk 2 is 2nd in both lists; it should beat chunks that are 1st in only one.
    fused = rrf([[h(1), h(2), h(3)], [h(4), h(2), h(5)]])
    assert fused[0].chunk_id == 2
    assert {x.chunk_id for x in fused} == {1, 2, 3, 4, 5}


def test_rrf_with_empty_fulltext_keeps_dense_order():
    # Stopword-only questions ("what is it?") give no full-text hits; dense must still answer.
    assert [x.chunk_id for x in rrf([[h(1), h(2)], []])] == [1, 2]


def test_retrieve_hybrid_rerank_pipeline_order(monkeypatch):
    calls = []
    monkeypatch.setattr(retrieve, "dense", lambda c, q, cfg, k, prefix=None: calls.append(("dense", k)) or [h(1), h(2)])
    monkeypatch.setattr(retrieve, "fulltext", lambda c, q, cfg, k, prefix=None: calls.append(("ft", k)) or [h(3)])
    monkeypatch.setattr(retrieve, "rerank", lambda q, hits: calls.append(("rr", len(hits))) or hits[::-1])
    cfg = PipelineConfig(hybrid=True, rerank=True, top_k=2, candidate_k=30)
    out = retrieve.retrieve(None, "q", cfg)
    assert calls == [("dense", 30), ("ft", 30), ("rr", 3)]
    assert len(out) == 2


def test_project_filter_reaches_both_retrievers(monkeypatch):
    # If only dense were filtered, BM25 would leak other projects back in via fusion.
    seen = []
    monkeypatch.setattr(retrieve, "dense", lambda c, q, cfg, k, prefix=None: seen.append(prefix) or [h(1)])
    monkeypatch.setattr(retrieve, "fulltext", lambda c, q, cfg, k, prefix=None: seen.append(prefix) or [])
    retrieve.retrieve(None, "q", PipelineConfig(hybrid=True), project="collabdocs")
    assert seen == ["projects:collabdocs/%", "projects:collabdocs/%"]
