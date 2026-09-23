import typer

from stuffrag import config, db

app = typer.Typer(no_args_is_help=True)
db_app = typer.Typer()
app.add_typer(db_app, name="db")


@db_app.command("init")
def db_init() -> None:
    """Create the schema (idempotent)."""
    tables = db.init()
    typer.echo(f"tables: {', '.join(tables)}")


@app.command()
def sync(source: str) -> None:
    """Ingest a source into the documents table. Sources: fastapi."""
    if source != "fastapi":
        raise typer.BadParameter(f"unknown source {source!r}; available: fastapi")
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
def ask(question: str, config_name: str = typer.Option("baseline", "--config")) -> None:
    """Answer a question from indexed sources, with citations."""
    from stuffrag import generate

    with db.connect() as conn:
        answer = generate.ask(conn, question, config.get(config_name))
    typer.echo(answer.text)
    by_id = {h.chunk_id: h for h in answer.hits}
    if answer.citations:
        typer.echo("\nSources:")
        for cid in answer.citations:
            typer.echo(f"  [c{cid}] {by_id[cid].document_id}")
