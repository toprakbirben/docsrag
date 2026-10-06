import sentence_transformers

from docsrag import rerank as rerank_mod


def test_model_loads_local_files_only_with_pinned_revision(monkeypatch):
    calls = []

    class FakeCrossEncoder:
        def __init__(self, model_name_or_path, **kwargs):
            calls.append(kwargs)

    monkeypatch.setattr(sentence_transformers, "CrossEncoder", FakeCrossEncoder)
    rerank_mod._model.cache_clear()
    try:
        rerank_mod._model()
        assert calls[0]["local_files_only"] is True
        assert calls[0]["revision"] == rerank_mod.REVISION
    finally:
        rerank_mod._model.cache_clear()


def test_model_falls_back_to_download_only_when_local_snapshot_missing(monkeypatch):
    calls = []

    class FakeCrossEncoder:
        def __init__(self, model_name_or_path, **kwargs):
            calls.append(kwargs)
            if kwargs.get("local_files_only"):
                raise OSError("no cached snapshot")

    monkeypatch.setattr(sentence_transformers, "CrossEncoder", FakeCrossEncoder)
    rerank_mod._model.cache_clear()
    try:
        rerank_mod._model()
        assert len(calls) == 2, "must not fall back unless the local-only load actually failed"
        assert calls[0]["local_files_only"] is True
        assert not calls[1].get("local_files_only")
        assert calls[1]["revision"] == rerank_mod.REVISION
    finally:
        rerank_mod._model.cache_clear()
