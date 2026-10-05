from stuffrag import generate
from stuffrag.config import PipelineConfig
from stuffrag.retrieve import Hit

HITS = [Hit(12, "fastapi:a.md", "yield deps", 0.9), Hit(14, "fastapi:b.md", "cleanup", 0.8)]


def test_citations_keep_only_ids_that_were_in_context():
    # A hallucinated [c99] must not show up as a source the user trusts.
    assert generate.parse_citations("Use yield [c12]. Also [c99].", HITS) == [12]


def test_citations_parse_grouped_form_and_dedupe():
    assert generate.parse_citations("x [c14, c12] y [c12]", HITS) == [14, 12]


def test_refusal_detection():
    assert generate.Answer("Not found in my sources.", [], []).refused
    assert not generate.Answer("Use yield [c12].", [12], HITS).refused


def test_no_hits_refuses_without_calling_the_llm(monkeypatch):
    monkeypatch.setattr(generate, "chat", lambda *a, **k: (_ for _ in ()).throw(AssertionError("called")))
    assert generate.generate("q", [], PipelineConfig()).refused


def test_chat_sets_context_window_so_passages_are_not_silently_truncated(monkeypatch):
    # Ollama's default num_ctx would cut 5x512-word passages; the model would answer from partial context.
    sent = {}

    class R:
        def raise_for_status(self): ...
        def json(self): return {"message": {"content": "ok"}}

    monkeypatch.setattr(generate.httpx, "post", lambda url, json, timeout: sent.update(json) or R())
    generate.chat("qwen3:8b", "sys", "user")
    assert sent["options"]["num_ctx"] >= 16384
    assert sent["think"] is False and sent["options"]["temperature"] == 0


def test_chat_merges_options_and_says_when_an_answer_was_cut(monkeypatch):
    # A capped answer must not look complete: the reader should know it stopped at the limit.
    sent = {}

    class R:
        def raise_for_status(self): ...
        def json(self): return {"message": {"content": "partial"}, "done_reason": "length"}

    monkeypatch.setattr(generate.httpx, "post", lambda url, json, timeout: sent.update(json) or R())
    out = generate.chat("qwen3:8b", "sys", "user", options={"num_predict": 5})
    assert sent["options"]["num_predict"] == 5 and sent["options"]["num_ctx"] >= 16384
    assert out.startswith("partial") and "cut off" in out
