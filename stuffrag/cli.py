import typer
from pathlib import Path

from stuffrag import config, db

app = typer.Typer(no_args_is_help=True)
db_app = typer.Typer()
app.add_typer(db_app, name="db")
eval_app = typer.Typer()
app.add_typer(eval_app, name="eval")


@db_app.command("init")
def db_init() -> None:
    """Create the schema (idempotent)."""
    tables = db.init()
    typer.echo(f"tables: {', '.join(tables)}")


@app.command()
def sync(source: str) -> None:
    """Ingest a source into the documents table. Sources: fastapi, projects."""
    if source == "projects":
        return _sync_projects()
    if source != "fastapi":
        raise typer.BadParameter(f"unknown source {source!r}; available: fastapi, projects")
    from stuffrag.ingest import fastapi_repo

    repo = fastapi_repo.checkout()
    docs, missing = fastapi_repo.collect(repo, fastapi_repo.FASTAPI_TAG)
    for m in missing:
        typer.echo(f"warning: unresolved include {m}", err=True)
    with db.connect() as conn:
        new, changed = db.upsert_documents(conn, docs)
    typer.echo(f"fastapi {fastapi_repo.FASTAPI_TAG}: {len(docs)} docs ({new} new, {changed} changed), "
               f"{len(missing)} unresolved includes")
    _index("baseline")


def _sync_projects() -> None:
    from stuffrag.ingest import projects

    docs, skipped = projects.collect()
    for path in skipped:
        typer.echo(f"skipped (secret): {path}", err=True)
    with db.connect() as conn:
        new, changed = db.upsert_documents(conn, docs)
        deleted = db.delete_missing(conn, "projects", docs)
    n_proj = len({d["metadata"]["project"] for d in docs})
    typer.echo(f"projects {projects.ROOT}: {n_proj} projects, {len(docs)} docs ({new} new, {changed} changed, "
               f"{deleted} deleted), {len(skipped)} secret files skipped")
    _index("baseline")


def _index(cfg_name: str) -> None:
    from stuffrag import embed

    cfg = config.get(cfg_name)
    with db.connect() as conn:
        stats = embed.index(conn, cfg, on_progress=lambda d, n: typer.echo(f"  embedded {d}/{n}", err=True))
    typer.echo(f"index [{cfg.name}: {cfg.chunker}, {cfg.embedder}]: {stats['chunked_docs']} docs chunked, "
               f"{stats['chunks']} chunks added, {stats['embedded']} embedded")


@app.command("index")
def index_cmd(config_name: str = typer.Option("baseline", "--config")) -> None:
    """Chunk + embed documents for a pipeline config (idempotent)."""
    _index(config_name)


@app.command()
def ask(question: str, config_name: str = typer.Option("baseline", "--config"),
        project: str = typer.Option(None, "--project", help="Search only this ~/projects folder.")) -> None:
    """Answer a question from indexed sources, with citations."""
    from stuffrag import generate

    with db.connect() as conn:
        answer = generate.ask(conn, question, config.get(config_name), project)
    typer.echo(answer.text)
    by_id = {h.chunk_id: h for h in answer.hits}
    if answer.citations:
        typer.echo("\nSources:")
        for cid in answer.citations:
            typer.echo(f"  [c{cid}] {by_id[cid].document_id}")


def _print_report(rep) -> None:
    typer.echo(rep.text)
    if rep.commits:
        typer.echo("\nCommits:")
        for c in rep.commits:
            pr = f" [PR #{c.pr['number']}]" if c.pr else ""
            typer.echo(f"  {c.sha[:7]} {c.date} {c.subject}{pr}")
    for n in rep.notes:
        typer.echo(f"note: {n}", err=True)


@app.command()
def changes(project: str, since: str = typer.Option("1 week ago", "--since"),
            until: str = typer.Option(None, "--until"),
            config_name: str = typer.Option("baseline", "--config")) -> None:
    """Explain what was added/removed/changed in a project, from diffs + commit/PR messages."""
    from stuffrag import history

    try:
        rep = history.changes(project, since, until, config.get(config_name))
    except history.HistoryError as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(1)
    _print_report(rep)


@app.command()
def why(project: str, topic: str, config_name: str = typer.Option("baseline", "--config")) -> None:
    """Find when and why something was added to a project, from the introducing commit/PR."""
    from stuffrag import history

    try:
        with db.connect() as conn:
            rep = history.why(conn, project, topic, config.get(config_name))
    except history.HistoryError as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(1)
    _print_report(rep)


@eval_app.command("run")
def eval_run(config_name: str = typer.Option("baseline", "--config")) -> None:
    """Run all eval questions through a pipeline config and record metrics."""
    from stuffrag import evals
    import json

    with db.connect() as conn:
        path = evals.run(conn, config.get(config_name), evals.load_questions())
    typer.echo(f"{path}\n{json.dumps(json.loads(path.read_text())['metrics'], indent=2)}")


@eval_app.command("compare")
def eval_compare(run_a: Path, run_b: Path) -> None:
    """Print metric deltas between two run files (B - A)."""
    import json
    from stuffrag import evals

    for line in evals.compare(json.loads(run_a.read_text()), json.loads(run_b.read_text())):
        typer.echo(line)


@eval_app.command("spotcheck")
def eval_spotcheck(run_file: Path, n: int = typer.Option(10, "--n")) -> None:
    """Show the first N judged answers and ask whether you agree with the judge (y/n)."""
    import json
    from stuffrag import evals

    questions = {q.id: q for q in evals.load_questions()}
    judged = [r for r in json.loads(run_file.read_text())["results"] if r["judge"] is not None][:n]
    disagreed = []
    for i, r in enumerate(judged, 1):
        q = questions[r["id"]]
        typer.echo(f"\n--- {i}/{len(judged)}  {r['id']}\nQ: {q.question}\n"
                   f"Expected facts: {', '.join(q.expected_answer)}\nAnswer:\n{r['answer']}\n"
                   f"Judge says correct: {'Yes' if r['judge'] else 'No'}")
        if not typer.confirm("Agree with the judge?", default=None):
            disagreed.append(r["id"])
    typer.echo(f"\njudge agreement: {len(judged) - len(disagreed)}/{len(judged)}")
    if disagreed:
        typer.echo(f"disagreed on: {', '.join(disagreed)}")
