from stuffrag import embed


def test_table_name_is_a_safe_identifier():
    assert embed.table("bge-m3") == "emb_bge_m3"
    assert embed.table("nomic-embed-text") == "emb_nomic_embed_text"


def test_nomic_gets_task_prefixes_bge_does_not(monkeypatch):
    # nomic-embed-text is trained with these prefixes; leaving them out measurably hurts retrieval.
    sent = []

    class R:
        def raise_for_status(self): ...
        def json(self): return {"embeddings": [[0.0]]}

    monkeypatch.setattr(embed.httpx, "post", lambda url, json, timeout: sent.append(json) or R())
    embed.embed(["q"], "nomic-embed-text", kind="query")
    embed.embed(["d"], "bge-m3")
    assert sent[0]["input"] == ["search_query: q"]
    assert sent[1]["input"] == ["d"]


def test_to_pgvector_literal():
    assert embed.to_pgvector([1.0, -0.5]) == "[1.0,-0.5]"
