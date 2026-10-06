from docsrag.db import diff_documents


def test_diff_documents_only_flags_real_changes():
    # Re-syncing an unchanged corpus must not trigger re-chunking/re-embedding.
    existing = {"a": "same", "b": "old"}
    docs = [{"id": "a", "body": "same"}, {"id": "b", "body": "new"}, {"id": "c", "body": "x"}]
    assert diff_documents(existing, docs) == (["c"], ["b"])


def test_stale_ids_drops_only_deleted_files_of_that_source():
    # A deleted/renamed project file must stop answering questions; other sources are untouched
    # because they aren't in `existing` (the caller scopes it by source_type).
    from docsrag.db import stale_ids
    assert stale_ids(["projects:a/x.py", "projects:a/gone.py"], [{"id": "projects:a/x.py"}]) == ["projects:a/gone.py"]
