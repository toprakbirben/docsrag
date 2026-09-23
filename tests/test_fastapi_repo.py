from pathlib import Path

from stuffrag.ingest.fastapi_repo import collect, expand_includes


def make_repo(tmp_path: Path) -> Path:
    (tmp_path / "docs_src/deps").mkdir(parents=True)
    (tmp_path / "docs_src/deps/tutorial008.py").write_text("async def get_db():\n    yield db\n")
    (tmp_path / "docs/en/docs/tutorial").mkdir(parents=True)
    (tmp_path / "docs/en/docs/tutorial/deps.md").write_text(
        "# Deps with yield\n\n"
        "{* ../../docs_src/deps/tutorial008.py hl[2] *}\n\n"
        # mkdocs-include-markdown variant with a space after {*, seen in real FastAPI docs
        "{!../../docs_src/deps/tutorial008.py!}\n\n"
        # the {!> ...!} variant (note the '>'), also used throughout the real docs
        "{!> ../../docs_src/deps/tutorial008.py!}\n\n"
        "{!../../docs_src/missing.py!}\n"
    )
    (tmp_path / "fastapi").mkdir()
    (tmp_path / "fastapi/routing.py").write_text("class APIRouter:\n    pass\n")
    return tmp_path


def test_includes_are_expanded_because_the_answer_is_often_in_docs_src(tmp_path):
    repo = make_repo(tmp_path)
    md = (repo / "docs/en/docs/tutorial/deps.md").read_text()
    text, missing = expand_includes(md, repo / "docs/en")
    assert "```python\nasync def get_db():\n    yield db\n```" in text
    # all three resolvable forms ({* *}, {! !}, {!> !}) must expand, not just the first
    assert text.count("```python\nasync def get_db():\n    yield db\n```") == 3
    assert missing == ["../../docs_src/missing.py"]  # reported, not silently dropped


def test_include_cannot_escape_the_repo(tmp_path):
    repo = make_repo(tmp_path)
    (tmp_path.parent / "secret.py").write_text("nope")
    # docs/en/../../../secret.py resolves to tmp_path.parent/secret.py: exists, but outside the repo.
    text, missing = expand_includes("{* ../../../secret.py *}", repo / "docs/en")
    assert "nope" not in text and missing == ["../../../secret.py"]


def test_collect_yields_docs_and_code_with_stable_ids(tmp_path):
    docs, missing = collect(make_repo(tmp_path), "0.0.0")
    by_id = {d["id"]: d for d in docs}
    assert set(by_id) == {"fastapi:docs/en/docs/tutorial/deps.md", "fastapi:fastapi/routing.py"}
    doc = by_id["fastapi:docs/en/docs/tutorial/deps.md"]
    assert doc["title"] == "Deps with yield"
    assert doc["metadata"] == {"path": "docs/en/docs/tutorial/deps.md", "tag": "0.0.0", "kind": "doc"}
    assert by_id["fastapi:fastapi/routing.py"]["metadata"]["kind"] == "code"
    assert missing == ["docs/en/docs/tutorial/deps.md: ../../docs_src/missing.py"]
