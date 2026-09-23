import pytest

from stuffrag.evals import QUESTIONS, load_questions
from stuffrag.ingest.fastapi_repo import CHECKOUT_ROOT, FASTAPI_TAG, expand_includes

REPO = CHECKOUT_ROOT / FASTAPI_TAG
QS = load_questions(QUESTIONS)


def test_ids_unique_and_fastapi_count():
    assert len({q.id for q in QS}) == len(QS)
    assert sum(q.source_type == "fastapi" for q in QS) == 25


@pytest.mark.skipif(not REPO.exists(), reason=f"run `stuff sync fastapi` first ({REPO} missing)")
@pytest.mark.parametrize("q", [q for q in QS if q.source_type == "fastapi"], ids=lambda q: q.id)
def test_every_key_fact_is_in_the_gold_sources(q):
    # A key fact missing from its gold file means the question is wrong, not the pipeline.
    assert q.gold_sources and q.expected_answer
    text = ""
    for gid in q.gold_sources:
        path = REPO / gid.removeprefix("fastapi:")
        assert path.is_file(), f"{q.id}: gold source {gid} does not exist at {FASTAPI_TAG}"
        text += expand_includes(path.read_text(), REPO / "docs/en")[0].lower()
    for fact in q.expected_answer:
        assert fact.lower() in text, f"{q.id}: key fact {fact!r} not found in {q.gold_sources}"
