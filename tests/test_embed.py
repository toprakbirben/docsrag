from docsrag import embed


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


def test_label_prefixes_every_chunk_but_unlabelled_docs_are_unchanged():
    # A code window from src/auth.ts is unfindable without its project/path; FastAPI docs have
    # no label, so their chunks (and the recorded eval baselines) must stay byte-identical.
    cfg = embed.PipelineConfig(chunk_tokens=4, chunk_overlap=0)
    rows = embed.chunk_rows([
        ("projects:app/a.ts", "a.ts", "myapp/a.ts", "one two three four five six"),
        ("fastapi:x.txt", "x.txt", None, "one two three four five six"),
    ], cfg)
    labelled = [t for d, _, _, t in rows if d.startswith("projects:")]
    plain = [t for d, _, _, t in rows if d.startswith("fastapi:")]
    assert labelled == ["myapp/a.ts\n\none two three four", "myapp/a.ts\n\nfive six"]
    assert plain == ["one two three four", "five six"]


def test_ollama_batch_covers_the_models_full_context(monkeypatch):
    # Ollama caps one input at num_batch (2048) tokens, below bge-m3's 8192 context, so dense
    # code chunks (512 words of TSX ~ 2.1k tokens) failed with "input length exceeds the context length".
    sent = []

    class R:
        def raise_for_status(self): ...
        def json(self): return {"embeddings": [[0.0]]}

    monkeypatch.setattr(embed.httpx, "post", lambda url, json, timeout: sent.append(json) or R())
    embed.embed(["d"], "bge-m3")
    embed.embed(["d"], "nomic-embed-text")
    assert sent[0]["options"] == {"num_batch": 8192}
    assert sent[1]["options"] == {"num_batch": 2048}
