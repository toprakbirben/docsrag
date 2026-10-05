import json

from typer.testing import CliRunner

from stuffrag.cli import app


def test_spotcheck_counts_agreement_and_skips_unjudged(tmp_path):
    # Unjudged (refusal) rows can't be agreed with, so they must not inflate the denominator;
    # the ids you disagree with are what goes into the commit body.
    run = {"results": [
        {"id": "fa-01", "judge": True, "answer": "Use yield in a Depends function."},
        {"id": "fa-02", "judge": None, "answer": "not found"},
        {"id": "fa-03", "judge": False, "answer": "Use a query param."},
    ]}
    path = tmp_path / "run.json"
    path.write_text(json.dumps(run))

    # An invalid reply is re-asked instead of silently counting as "no".
    out = CliRunner().invoke(app, ["eval", "spotcheck", str(path)], input="y\nmaybe\nn\n")

    assert out.exit_code == 0, out.output
    assert "Judge says correct: Yes" in out.output
    assert "Judge says correct: No" in out.output
    assert "fa-02" not in out.output
    assert "judge agreement: 1/2" in out.output
    assert "disagreed on: fa-03" in out.output
