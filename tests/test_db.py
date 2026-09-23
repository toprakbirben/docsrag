from stuffrag.db import diff_documents


def test_diff_documents_only_flags_real_changes():
    # Re-syncing an unchanged corpus must not trigger re-chunking/re-embedding.
    existing = {"a": "same", "b": "old"}
    docs = [{"id": "a", "body": "same"}, {"id": "b", "body": "new"}, {"id": "c", "body": "x"}]
    assert diff_documents(existing, docs) == (["c"], ["b"])
